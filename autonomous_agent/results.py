import os, csv, glob, json
from collections import defaultdict

def read_global_checkpoint(out_root):
    p=os.path.join(out_root, "global_checkpoint.json")
    if os.path.exists(p):
        try:
            j=json.load(open(p,"r",encoding="utf-8"))
            return set(j.get("completed_files", []) or [])
        except Exception:
            return set()
    return set()

def iter_descriptor_csv(out_root):
    # pattern: out_root/<base>/<base>_descriptors.csv
    return glob.glob(os.path.join(out_root, "*", "*_descriptors.csv"))

def load_structure_records(out_root):
    """Return dict: structure_id -> record with descriptor values keyed by adsorbate."""
    rec={}
    for p in iter_descriptor_csv(out_root):
        try:
            rows=list(csv.DictReader(open(p,"r",encoding="utf-8")))
        except Exception:
            continue
        if not rows:
            continue
        # structure id is in column 'structure' (mp-id)
        sid = rows[0].get("structure") or os.path.basename(p).replace("_descriptors.csv","")
        r={"structure": sid}
        # carry a few meta fields from first row
        for k in ("miller","termination_id","engine"):
            if rows[0].get(k) not in (None,""):
                r[k]=rows[0][k]
        # gather per-adsorbate descriptor_value_eV
        for row in rows:
            ads=row.get("adsorbate","").strip()
            val=row.get("descriptor_value_eV","")
            if not ads:
                continue
            try:
                v=float(val)
            except Exception:
                continue
            r[ads]=v
            # also store raw descriptor name as column if user wants
            dname=row.get("descriptor","").strip()
            if dname:
                r[dname]=v
        rec[sid]=r
    return rec
