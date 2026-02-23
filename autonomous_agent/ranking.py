import os, json, math, csv
from .util import ensure_dir
from .results import load_structure_records

def _score_structure(rec, score_terms, missing_policy="skip", missing_penalty=5.0):
    pen=0.0; wsum=0.0; used=0
    for t in score_terms:
        key=t["key"]; target=float(t.get("target",0.0)); w=float(t.get("weight",1.0))
        v=rec.get(key)
        if v is None:
            if missing_policy=="penalize":
                pen += w*missing_penalty
                wsum += w
                used += 1
            continue
        pen += w*abs(float(v)-target)
        wsum += w
        used += 1
    if used==0:
        return None
    return pen/max(wsum,1e-12)

def _pareto_front(recs, terms, missing_policy="skip", missing_penalty=5.0):
    # build vectors
    pts=[]
    for rec in recs:
        vec=[]
        ok=True
        for t in terms:
            key=t["key"]; target=float(t.get("target",0.0)); w=float(t.get("weight",1.0))
            v=rec.get(key)
            if v is None:
                if missing_policy=="penalize":
                    vec.append(w*missing_penalty)
                else:
                    ok=False
                    break
            else:
                vec.append(w*abs(float(v)-target))
        if ok:
            pts.append((rec, vec))
    def dominates(a,b):
        return all(x<=y for x,y in zip(a,b)) and any(x<y for x,y in zip(a,b))
    remaining=pts[:]
    fronts=[]
    while remaining:
        front=[]
        for r,v in remaining:
            dom=False
            for r2,v2 in remaining:
                if dominates(v2,v):
                    dom=True; break
            if not dom:
                front.append((r,v))
        fronts.append(front)
        ids=set(id(r) for r,_ in front)
        remaining=[(r,v) for r,v in remaining if id(r) not in ids]
    ranked=[]
    for i,f in enumerate(fronts):
        for r,_ in f:
            rr=dict(r); rr["pareto_rank"]=i
            ranked.append(rr)
    ranked.sort(key=lambda x:(int(x.get("pareto_rank",999)), x.get("structure","")))
    return ranked

def rank_structures(out_root, outdir, ranking_cfg, records=None):
    ensure_dir(outdir)
    if records is None:
        records = load_structure_records(out_root)
    recs=list(records.values())
    terms=ranking_cfg.get("score_terms", []) or []
    method=(ranking_cfg.get("method","score") or "score").lower()
    missing_policy=(ranking_cfg.get("missing_policy","skip") or "skip").lower()
    missing_penalty=float(ranking_cfg.get("missing_penalty", 5.0))
    if method=="pareto":
        ranked=_pareto_front(recs, terms, missing_policy=missing_policy, missing_penalty=missing_penalty)
    else:
        ranked=[]
        for r in recs:
            s=_score_structure(r, terms, missing_policy=missing_policy, missing_penalty=missing_penalty)
            if s is None: 
                continue
            rr=dict(r); rr["score"]=s
            ranked.append(rr)
        ranked.sort(key=lambda x:(float(x.get("score",1e9)), x.get("structure","")))
    # write outputs
    if ranked:
        cols=sorted({k for r in ranked for k in r.keys()})
        with open(os.path.join(outdir,"ranked_structures.csv"),"w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for r in ranked: w.writerow(r)
        with open(os.path.join(outdir,"ranked_structures.json"),"w",encoding="utf-8") as f:
            json.dump(ranked, f, ensure_ascii=False, indent=2)
    return ranked
