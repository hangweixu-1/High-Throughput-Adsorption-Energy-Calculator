import os, json, time, glob, shutil, random
from collections import deque, defaultdict

from .util import ensure_dir, now_iso, run_cmd
from .query_library import FAMILIES
from .active_learning import make_agent
from .log_parser import classify, suggest_overrides
from .cif_filter import count_sites_cif
from .ht_cli import build_download_cmd, build_ht_cmd
from .results import read_global_checkpoint, load_structure_records
from .ranking import rank_structures
from .selection import select_topk, select_diverse
from .dashboard import write_dashboard
from .calibration import calibrate_stage1_with_stage2, apply_calibration
from .deepseek_expand import propose_queries
from .slurm import slurm_available, make_sbatch_script, submit_job, job_done

def _repo_root():
    return os.getcwd()

def _safe_name(s):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)

def _load_state(path):
    if os.path.exists(path):
        return json.load(open(path,"r",encoding="utf-8"))
    return {}

def _save_state(path, state):
    json.dump(state, open(path,"w",encoding="utf-8"), ensure_ascii=False, indent=2)

def _list_cifs(folder):
    return sorted(glob.glob(os.path.join(folder,"*.cif")))

def run_pipeline(cfg, use_slurm=False):
    repo=_repo_root()
    for req in ("download_cif.py","High_throughput.py"):
        if not os.path.exists(os.path.join(repo, req)):
            raise RuntimeError(f"Repo root must contain {req}. Current={repo}")

    ws=cfg.get("workspace","workspace_budget")
    ensure_dir(ws)
    logs=os.path.join(ws,"logs"); ensure_dir(logs)
    reports=os.path.join(ws,"reports"); ensure_dir(reports)
    state_path=os.path.join(ws,"pipeline_state.json")

    budget=cfg.get("budget",{}) or {}
    max_successes=int(budget.get("max_successes",50))
    max_attempts=int(budget.get("max_attempts",max_successes*6))
    batch_size=int(budget.get("batch_size",10))

    dl_cfg=cfg.get("download",{}) or {}
    per_query_limit=int(dl_cfg.get("per_query_limit", 80))

    eng=cfg.get("engineering",{}) or {}
    max_fail=int(eng.get("max_structure_failures",2))
    retry_failed=bool(eng.get("retry_failed_structures",True))
    retry_batch_size=int(eng.get("retry_batch_size",3))
    heal_cfg=(eng.get("log_heal",{}) or {})
    heal_enabled=bool(heal_cfg.get("enabled",True))
    heal_rounds=int(heal_cfg.get("max_heal_rounds",6))

    max_nsites=int((cfg.get("filters",{}) or {}).get("max_nsites",10**9))

    qcfg=cfg.get("queries",{}) or {}
    seed_fams=qcfg.get("seed_families",[]) or []
    initial_queue=qcfg.get("initial_queue",[]) or []
    max_queue_size=int(qcfg.get("max_queue_size",200))

    al_cfg=cfg.get("active_learning",{}) or {}
    expand_cfg=(al_cfg.get("expansion",{}) or {})
    expand_strategy=(expand_cfg.get("strategy","hybrid") or "hybrid").lower()
    max_new_per_expand=int(expand_cfg.get("max_new_queries_per_expand",10))
    min_successes_before_expand=int(al_cfg.get("min_successes_before_expand",12))

    # stage configs
    stage1=dict(cfg.get("stage1",{}) or {})
    stage2=dict(cfg.get("stage2",{}) or {})
    ranking_cfg=cfg.get("ranking",{}) or {}
    score_terms=ranking_cfg.get("score_terms",[]) or []

    # outputs aligned to repo
    s1_out=os.path.join(ws, stage1.get("output","ht_stage1"))
    ensure_dir(s1_out)
    s2_out=os.path.join(ws, stage2.get("output","ht_stage2"))
    ensure_dir(s2_out)

    # state
    st=_load_state(state_path)
    st.setdefault("created_at", now_iso())
    st.setdefault("round", 0)
    st.setdefault("attempts", 0)
    st.setdefault("successes", 0)
    st.setdefault("disabled_queries", [])
    st.setdefault("downloaded_queries", [])
    st.setdefault("query_stats", {})  # q -> {n, mean}
    st.setdefault("failed_structures", {})  # bn -> count
    st.setdefault("queue", [])

    # queue init
    q=deque(st.get("queue") or [])
    if not q:
        for fam in seed_fams:
            for item in FAMILIES.get(fam, []):
                q.append(item)
        for item in initial_queue:
            q.append(item)
        st["queue"]=list(q)

    agent=make_agent(cfg)

    # logs
    progress=os.path.join(logs,"progress.csv")
    if not os.path.exists(progress):
        open(progress,"w",encoding="utf-8").write("ts,round,successes,attempts,queue_len,disabled,downloaded,note\n")

    # SLURM
    slurm_cfg=cfg.get("slurm",{}) or {}
    slurm_on=bool(use_slurm) and bool(slurm_cfg.get("enabled",False))
    if slurm_on and not slurm_available():
        print("[WARN] SLURM requested but sbatch/squeue not found. Use local.")
        slurm_on=False

    # cache of completed files
    completed=read_global_checkpoint(s1_out)
    st["successes"]=len(completed)

    no_progress=0
    forced_expands=0

    while st["successes"] < max_successes and st["attempts"] < max_attempts:
        st["round"] += 1
        # pick query
        if not q:
            # try expand
            _expand_queries(cfg, st, q, expand_strategy, max_new_per_expand, max_queue_size)
            if not q:
                break

        # choose best among a window
        window=list(q)[:min(16, len(q))]
        scored=[(agent.score(x), x) if hasattr(agent,"score") else (0.0,x) for x in window]
        scored.sort(key=lambda t:(-t[0], t[1]))
        query=scored[0][1]
        # rotate
        q.remove(query); q.append(query); st["queue"]=list(q)

        # download once per query (because download script doesn't paginate)
        dl_root=os.path.join(ws,"downloads",_safe_name(query))
        ensure_dir(dl_root)
        if query not in st["downloaded_queries"]:
            api_key=(cfg.get("env",{}) or {}).get("MP_API_KEY","") or os.environ.get("MP_API_KEY","")
            cmd=build_download_cmd(repo, dl_root, query, per_query_limit, dl_cfg, api_key=api_key if api_key else None)
            dl_log=os.path.join(logs,"download.log")
            run_cmd(cmd, stdout_path=dl_log, stderr_path=dl_log)
            st["downloaded_queries"].append(query)

        # list cifs for query
        cifs=_list_cifs(dl_root)
        # size filter
        kept=[]
        for p in cifs:
            n=count_sites_cif(p)
            if n is not None and n>max_nsites:
                continue
            kept.append(p)

        # choose batch: not completed and not too failed
        batch=[]
        for p in kept:
            bn=os.path.basename(p)
            if bn in completed:
                continue
            if int(st["failed_structures"].get(bn,0)) >= max_fail:
                continue
            batch.append(p)
            if len(batch) >= batch_size:
                break

        if not batch:
            # disable query if no usable cifs
            _mark_empty_query(st, q, query)
            no_progress += 1
            _log_progress(progress, st, q, note=f"empty_batch:{query}")
            if st["successes"] >= min_successes_before_expand and no_progress >= int(eng.get("no_progress_rounds_trigger_expand",6)):
                forced_expands += 1
                if forced_expands <= int(eng.get("no_progress_max_forced_expands",4)):
                    _expand_queries(cfg, st, q, expand_strategy, max_new_per_expand, max_queue_size)
                    _log_progress(progress, st, q, note="forced_expand")
            _save_state(state_path, st)
            continue

        # make batch folder (pass to High_throughput as cif_folder)
        batch_dir=os.path.join(ws,"batches",f"round_{st['round']:04d}", "cifs")
        ensure_dir(batch_dir)
        for p in batch:
            dst=os.path.join(batch_dir, os.path.basename(p))
            if not os.path.exists(dst):
                shutil.copy2(p, dst)

        # run HT stage1 with heal
        before=read_global_checkpoint(s1_out)
        ht_cmd=build_ht_cmd(repo, batch_dir, stage1.get("engine","mace"), s1_out, stage1)
        ok, meta = _run_ht_with_heal(ht_cmd, logs, stage1, heal_enabled, heal_rounds) if not slurm_on else _run_ht_slurm(ht_cmd, ws, logs, slurm_cfg, name=f"s1_r{st['round']:04d}")
        after=read_global_checkpoint(s1_out)
        new_done=sorted(list(after-before))
        completed=after

        st["attempts"] += len(batch)
        prev_s=st["successes"]
        st["successes"]=len(completed)
        gained=st["successes"]-prev_s

        # reward per query: best penalty among newly completed structures + small bonus for count
        reward=_reward_from_new_done(s1_out, new_done, ranking_cfg)
        if reward is not None and hasattr(agent,"update"):
            agent.update(query, reward)
            _update_query_stats(st, query, reward)

        if gained<=0:
            # mark failed
            for p in batch:
                bn=os.path.basename(p)
                if bn not in after:
                    st["failed_structures"][bn]=int(st["failed_structures"].get(bn,0))+1
            no_progress += 1
        else:
            no_progress = 0

        _log_progress(progress, st, q, note=f"round_done:{query};gained={gained}")
        _save_state(state_path, st)

        # if stuck, expand (after some successes)
        if st["successes"] >= min_successes_before_expand and no_progress >= int(eng.get("no_progress_rounds_trigger_expand",6)):
            forced_expands += 1
            if forced_expands <= int(eng.get("no_progress_max_forced_expands",4)):
                _expand_queries(cfg, st, q, expand_strategy, max_new_per_expand, max_queue_size)
                _log_progress(progress, st, q, note="forced_expand")
                _save_state(state_path, st)

    # stage1 ranking + final dir
    final_dir=os.path.join(reports,"final"); ensure_dir(final_dir)
    ranked_s1=rank_structures(s1_out, final_dir, ranking_cfg)

    # stage2 refine (optional)
    if stage2.get("enabled"):
        ranked_for_pick=ranked_s1[:]
        k=int(stage2.get("topk",20))
        selection=(stage2.get("selection","top") or "top").lower()
        if selection=="diverse":
            keys=stage2.get("diverse_keys",[]) or []
            picked=select_diverse(ranked_for_pick, k, keys) if keys else select_topk(ranked_for_pick,k)
        else:
            picked=select_topk(ranked_for_pick,k)
        # copy cifs to refine dir
        refine_dir=os.path.join(ws,"stage2_refine_cifs"); ensure_dir(refine_dir)
        _copy_cifs_for_structures(picked, ws, refine_dir)
        # run gpaw
        ht2_cmd=build_ht_cmd(repo, refine_dir, stage2.get("engine","gpaw"), s2_out, stage2)
        _run_ht_with_heal(ht2_cmd, logs, stage2, heal_enabled, heal_rounds) if not slurm_on else _run_ht_slurm(ht2_cmd, ws, logs, slurm_cfg, name="stage2_refine")
        # calibration
        cal_cfg=(stage2.get("calibration",{}) or {})
        if cal_cfg.get("enabled"):
            models=calibrate_stage1_with_stage2(
                s1_out, s2_out, score_terms,
                clamp_slope=tuple(cal_cfg.get("clamp_slope",[0.5,1.5])),
                min_points=int(cal_cfg.get("min_points",6))
            )
            json.dump(models, open(os.path.join(final_dir,"multifidelity_calibration.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=2)
            s1_rec=load_structure_records(s1_out)
            s2_rec=load_structure_records(s2_out)
            merged=apply_calibration(s1_rec, s2_rec, models)
            # re-rank using merged records
            ranked_final=rank_structures(s1_out, final_dir, ranking_cfg, records=merged)
        else:
            ranked_final=ranked_s1
    else:
        ranked_final=ranked_s1

    # dashboard + export
    write_dashboard(final_dir, ranked_final, title="HTAEC Final Candidates")
    _export_top(final_dir, ranked_final, ws, topn=int(ranking_cfg.get("topn_export",20)))
    _save_state(state_path, st)
    print(f"[DONE] successes={st['successes']} attempts={st['attempts']} -> {final_dir}")
    return 0

def _run_ht_slurm(cmd, ws, logs, slurm_cfg, name="ht_job"):
    scripts=os.path.join(ws,"slurm_scripts"); ensure_dir(scripts)
    script_path=os.path.join(scripts, f"{name}.sbatch")
    open(script_path,"w",encoding="utf-8").write(make_sbatch_script(cmd, slurm_cfg, logs, name=name))
    jobid=submit_job(script_path)
    while not job_done(jobid):
        time.sleep(30)
    return True, {"slurm_jobid": jobid}

def _run_ht_with_heal(cmd, logs, stage_cfg, enabled=True, max_rounds=6):
    ht_log=os.path.join(logs,"high_throughput.log")
    ht_err=os.path.join(logs,"high_throughput.err")
    attempts=0
    while True:
        attempts += 1
        rc,out,err=run_cmd(cmd, stdout_path=ht_log, stderr_path=ht_err)
        if rc==0:
            return True, {"attempts": attempts}
        if not enabled or attempts>=max_rounds:
            return False, {"attempts": attempts, "rc": rc}
        cls=classify((err or "") + "\n" + (out or ""))
        ovs=suggest_overrides(cls) if cls else []
        if not ovs:
            return False, {"attempts": attempts, "rc": rc}
        # apply next override by mutating stage_cfg and rebuilding cmd
        ov=ovs[min(attempts-1, len(ovs)-1)]
        stage_cfg.update(ov)
        # update cmd in-place
        cmd[:] = _rebuild_cmd(cmd, stage_cfg)

def _rebuild_cmd(old_cmd, stage_cfg):
    # old_cmd = ['python','.../High_throughput.py', cif, engine, '-o', out, ...]
    cif=old_cmd[2]; engine=old_cmd[3]; out=old_cmd[5]
    repo_root=os.path.dirname(os.path.dirname(old_cmd[1]))  # .../High_throughput.py -> repo
    from .ht_cli import build_ht_cmd
    return build_ht_cmd(repo_root, cif, engine, out, stage_cfg)

def _reward_from_new_done(out_root, new_done_files, ranking_cfg):
    # compute best penalty among these newly completed structures
    if not new_done_files:
        return None
    # load their descriptor csv records
    terms=ranking_cfg.get("score_terms",[]) or []
    missing_policy=(ranking_cfg.get("missing_policy","skip") or "skip").lower()
    miss_pen=float(ranking_cfg.get("missing_penalty",5.0))
    best=None
    for bn in new_done_files:
        sid=os.path.splitext(bn)[0]
        p=os.path.join(out_root, sid, f"{sid}_descriptors.csv")
        if not os.path.exists(p):
            continue
        # build rec for this structure
        import csv
        try:
            rows=list(csv.DictReader(open(p,"r",encoding="utf-8")))
        except Exception:
            continue
        if not rows:
            continue
        rec={"structure": sid}
        for row in rows:
            ads=row.get("adsorbate","").strip()
            if not ads:
                continue
            try:
                rec[ads]=float(row.get("descriptor_value_eV",""))
            except Exception:
                continue
        # score (same as ranking)
        pen=0.0; wsum=0.0; used=0
        for t in terms:
            key=t["key"]; target=float(t.get("target",0.0)); w=float(t.get("weight",1.0))
            v=rec.get(key)
            if v is None:
                if missing_policy=="penalize":
                    pen += w*miss_pen; wsum += w; used += 1
                continue
            pen += w*abs(float(v)-target); wsum += w; used += 1
        if used==0:
            continue
        score=pen/max(wsum,1e-12)
        if best is None or score < best:
            best=score
    if best is None:
        return None
    # reward: negative penalty; add small bonus for number of new completions
    return -best + 0.05*len(new_done_files)

def _update_query_stats(st, query, reward):
    qs=st["query_stats"].setdefault(query, {"n":0,"mean":0.0})
    n=int(qs.get("n",0))+1
    m=float(qs.get("mean",0.0))
    qs["mean"]=m+(reward-m)/n
    qs["n"]=n

def _mark_empty_query(st, q, query):
    # if a query yields no usable CIFs repeatedly, disable
    empty=st.get("query_empty",{})
    c=int(empty.get(query,0))+1
    empty[query]=c; st["query_empty"]=empty
    if c>=2 and query not in st["disabled_queries"]:
        st["disabled_queries"].append(query)
        q2=[x for x in q if x!=query]
        q.clear(); q.extend(q2)
        st["queue"]=list(q)

def _log_progress(path, st, q, note=""):
    with open(path,"a",encoding="utf-8") as f:
        f.write(f"{now_iso()},{st.get('round',0)},{st.get('successes',0)},{st.get('attempts',0)},{len(q)},{len(st.get('disabled_queries',[]))},{len(st.get('downloaded_queries',[]))},{note}\n")

def _expand_queries(cfg, st, q, strategy, max_new, max_queue_size):
    if len(q) >= max_queue_size:
        return
    topq=sorted((st.get("query_stats",{}) or {}).items(), key=lambda kv: -float(kv[1].get("mean",0.0)))[:6]
    top_queries=[k for k,_ in topq]
    goal_hint="CO2 hydrogenation to methanol catalyst screening"
    new=[]
    if strategy in ("deepseek","hybrid"):
        # respect deepseek budget
        ds=cfg.get("controller",{}).get("deepseek",{}) or {}
        # naive budget checks: stored in state
        st.setdefault("deepseek_calls",0)
        st.setdefault("deepseek_tokens",0)
        if st["deepseek_calls"] < int(ds.get("max_calls", 20)):
            arr=propose_queries(goal_hint, top_queries, max_new, cfg)
            if arr:
                st["deepseek_calls"] += 1
                new.extend(arr)
    if strategy in ("heuristic","hybrid") and len(new)<max_new:
        new.extend(_heuristic_expand(top_queries, max_new-len(new)))
    # add to queue
    existing=set(list(q) + st.get("disabled_queries",[]) + st.get("downloaded_queries",[]))
    for item in new:
        if len(q) >= max_queue_size:
            break
        if item not in existing:
            q.append(item); existing.add(item)
    st["queue"]=list(q)

def _heuristic_expand(top_queries, k):
    # expand around common catalyst motifs
    base=["Cu","Zn","Ga","In","Sn","Ni","Co","Fe","Mo","W","Ti","V","Nb","Ta"]
    anions=["O","S","N","P","Se","C"]
    out=[]
    for _ in range(k*2):
        a=random.choice(base); b=random.choice(base)
        if a==b: continue
        out.append(f"{a}-{b}")
        out.append(f"{a}-{b}-O")
    for _ in range(k):
        m=random.choice(base); x=random.choice(anions)
        out.append(f"{m}-{x}")
    # dedup
    seen=set(); out2=[]
    for s in out:
        if s not in seen:
            out2.append(s); seen.add(s)
    return out2[:k]

def _copy_cifs_for_structures(picked_rows, ws, refine_dir):
    # Try to copy from downloads cache by filename mp-id.cif
    for r in picked_rows:
        sid=r.get("structure") or r.get("mp_id") or ""
        if not sid: 
            continue
        bn=f"{sid}.cif"
        # search downloads
        src=None
        for p in glob.glob(os.path.join(ws,"downloads","**",bn), recursive=True):
            src=p; break
        if not src:
            # fallback: stage1 out includes a copy at ht_stage1/<sid>/<sid>.cif
            for p in glob.glob(os.path.join(ws,"ht_stage1",sid, bn)):
                src=p; break
        if src:
            dst=os.path.join(refine_dir, bn)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

def _export_top(final_dir, ranked_rows, ws, topn=20):
    out_cifs=os.path.join(final_dir,"top_candidates_cifs"); ensure_dir(out_cifs)
    out_xyz=os.path.join(final_dir,"top_candidates_xyz"); ensure_dir(out_xyz)
    top=ranked_rows[:topn]
    for r in top:
        sid=r.get("structure") or ""
        if not sid: 
            continue
        bn=f"{sid}.cif"
        # copy cif from downloads/stage1
        src=None
        for p in glob.glob(os.path.join(ws,"downloads","**",bn), recursive=True):
            src=p; break
        if not src:
            p=os.path.join(ws,"ht_stage1",sid,bn)
            if os.path.exists(p): src=p
        if src:
            dst=os.path.join(out_cifs, bn)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        # copy best xyz
        for p in glob.glob(os.path.join(ws,"ht_stage1",sid,f"*best*.xyz"))[:3]:
            dst=os.path.join(out_xyz, os.path.basename(p))
            if not os.path.exists(dst):
                shutil.copy2(p, dst)
