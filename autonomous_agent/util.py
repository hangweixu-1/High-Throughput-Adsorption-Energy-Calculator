import json, os, subprocess, shutil, pathlib, datetime

def now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")

def ensure_dir(p):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def print_json(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2))

def which(exe):
    return shutil.which(exe)

def run_cmd(cmd, cwd=None, env=None, stdout_path=None, stderr_path=None, timeout=None):
    if stdout_path:
        ensure_dir(pathlib.Path(stdout_path).parent)
    if stderr_path:
        ensure_dir(pathlib.Path(stderr_path).parent)
    out_f = open(stdout_path, "a", encoding="utf-8") if stdout_path else None
    err_f = open(stderr_path, "a", encoding="utf-8") if stderr_path else None
    try:
        p = subprocess.run(
            cmd, cwd=cwd, env=env,
            stdout=out_f or subprocess.PIPE,
            stderr=err_f or subprocess.PIPE,
            text=True, timeout=timeout
        )
        return p.returncode, (p.stdout if out_f is None else ""), (p.stderr if err_f is None else "")
    finally:
        if out_f: out_f.close()
        if err_f: err_f.close()
