import os, pathlib, time
from .util import run_cmd, which, ensure_dir

def slurm_available():
    return which("sbatch") and which("squeue")

def make_sbatch_script(cmd, slurm_cfg, log_dir, name="ht_job"):
    ensure_dir(log_dir)
    lines=[
        "#!/bin/bash",
        f"#SBATCH -J {name}",
        f"#SBATCH -t {slurm_cfg.get('time','04:00:00')}",
        f"#SBATCH --cpus-per-task={slurm_cfg.get('cpus_per_task',4)}",
        f"#SBATCH --mem={slurm_cfg.get('mem','16G')}",
        f"#SBATCH -o {pathlib.Path(log_dir)/'slurm-%j.out'}",
        f"#SBATCH -e {pathlib.Path(log_dir)/'slurm-%j.err'}",
    ]
    part=slurm_cfg.get("partition") or ""
    if part:
        lines.append(f"#SBATCH -p {part}")
    gpus=int(slurm_cfg.get("gpus",0) or 0)
    if gpus:
        lines.append(f"#SBATCH --gres=gpu:{gpus}")
    for extra in (slurm_cfg.get("extra_sbatch",[]) or []):
        lines.append(str(extra))
    lines += ["", "set -e", "echo "[SBATCH] start $(date)"", " ".join(cmd), "echo "[SBATCH] end $(date)""]
    return "\n".join(lines)+"\n"

def submit_job(script_path):
    rc,out,err=run_cmd(["sbatch", script_path])
    if rc!=0:
        raise RuntimeError(f"sbatch failed: {out}\n{err}")
    jobid=None
    for tok in (out or "").split():
        if tok.isdigit():
            jobid=tok; break
    return jobid

def job_done(jobid):
    rc,out,err=run_cmd(["squeue","-j",str(jobid),"-h"])
    if rc!=0:
        return False
    return out.strip()==""  # empty means done
