import re

def count_sites_cif(path):
    """Lightweight CIF atom_site line counter (fallback when pymatgen unavailable)."""
    try:
        txt=open(path,"r",encoding="utf-8",errors="ignore").read()
    except Exception:
        return None
    m=re.search(r"loop_\s*\n(?:.*atom_site_.*\n)+", txt, flags=re.I)
    if not m:
        return None
    start=m.end()
    rest=txt[start:].splitlines()
    n=0
    for line in rest:
        s=line.strip()
        if not s or s.lower().startswith("loop_") or s.startswith("_"):
            if n>0:
                break
            continue
        n+=1
    return n if n>0 else None
