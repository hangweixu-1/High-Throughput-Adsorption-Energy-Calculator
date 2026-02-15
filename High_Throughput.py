# -*- coding: ascii -*-
"""
High-throughput adsorption energy calculator.

Usage:
  python job.py <cif_folder> <engine: mace|gpaw> [options]

Examples:
  python job.py ./cifs mace
  python job.py ./cifs gpaw --output results_gpaw
  python job.py ./cifs mace --resume              # continue from last run
  python job.py ./cifs mace --miller 1,1,1 --supercell 3,3,1

Output structure:
  ht_results/
    global_checkpoint.json
    gas_cache.json
    calculation.log
    CuPd/
      CuPd.cif                    (copy of input)
      CuPd_slab_relaxed.xyz
      CuPd_CO2_cand0_relaxed.xyz
      CuPd_CO2_BEST_cand2.xyz
      CuPd_descriptors.csv
      checkpoint.json
    NiCu/
      ...
"""
import os
import sys
import glob
import json
import csv
import logging
import warnings
import time
import argparse
import shutil
import numpy as np

# -----------------------------
# Real-time output helpers
# -----------------------------
def plog(msg, level="INFO"):
    timestamp = time.strftime("%H:%M:%S")
    tag = {"INFO": "[OK]", "WARN": "[!!]", "ERROR": "[ERR]", "STEP": "[>>]",
           "DONE": "[DONE]", "RUN": "[...]"}.get(level, "[--]")
    print("[%s] %s %s" % (timestamp, tag, msg), flush=True)
    log_fn = {"WARN": logging.warning, "ERROR": logging.error}.get(level, logging.info)
    log_fn(msg)

def pbar_header(title):
    line = "=" * 60
    print("", flush=True)
    print(line, flush=True)
    print("  %s" % title, flush=True)
    print(line, flush=True)

# -----------------------------
# Parse CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="High-throughput adsorption energy calculation",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("cif_folder", help="Path to folder containing .cif files")
    p.add_argument("engine", choices=["mace", "gpaw"], help="Calculation engine: mace or gpaw")
    p.add_argument("--output", "-o", default="ht_results", help="Output root directory (default: ht_results)")
    p.add_argument("--resume", "-r", action="store_true", help="Resume from last checkpoint")
    p.add_argument("--miller", default="1,1,0", help="Miller index, comma-separated (default: 1,1,0)")
    p.add_argument("--supercell", default="2,2,1", help="Supercell, comma-separated (default: 2,2,1)")
    p.add_argument("--mace-model", default="/public/home/lib/tc/xifuneng/2023-12-03-mace-128-L1_epoch-199.model",
                   help="Path to MACE model file")
    p.add_argument("--mace-device", default="cpu", help="MACE device: cpu or cuda (default: cpu)")
    p.add_argument("--fmax-mace", type=float, default=0.10, help="MACE fmax (default: 0.10)")
    p.add_argument("--fmax-gpaw", type=float, default=0.05, help="GPAW fmax (default: 0.05)")
    p.add_argument("--max-steps", type=int, default=200, help="Max BFGS steps (default: 200)")
    p.add_argument("--max-candidates", type=int, default=12, help="Max adsorption candidates per adsorbate")
    return p.parse_args()

# -----------------------------
# Config object
# -----------------------------
class Config:
    pass

def build_config(args):
    c = Config()
    c.ENGINE = args.engine
    c.OUTDIR = os.path.abspath(args.output)
    c.RESUME = args.resume
    c.CIF_FOLDER = os.path.abspath(args.cif_folder)

    miller = tuple(int(x) for x in args.miller.split(","))
    assert len(miller) == 3
    c.TARGET_MILLER = miller

    sc = tuple(int(x) for x in args.supercell.split(","))
    assert len(sc) == 3
    c.SLAB_SUPERCELL = sc

    c.MACE_MODEL_PATH = args.mace_model
    c.MACE_DEVICE = args.mace_device
    c.MACE_DTYPE = "float64"
    c.MACE_FMAX = args.fmax_mace
    c.MACE_STEPS = args.max_steps

    c.PW_CUTOFF = 470
    c.XC = "PBE"
    c.FMAX = args.fmax_gpaw
    c.KPTS_SLAB = (3, 3, 1)
    c.KPTS_MOL = (1, 1, 1)

    c.MAX_INDEX = 1
    c.MIN_SLAB_SIZE = 6.0
    c.MIN_VACUUM = 12.0
    c.FIX_N_LAYERS = 3
    c.LAYER_TOL = 0.12

    c.USE_TIER_B = True
    c.ADS_CORR = {"*H": 0.0, "*CO": 0.0, "*COOH": 0.0, "*CHO": 0.0, "*CH2O": 0.0, "*CO2": 0.0}
    c.GAS_CORR = {"H2": 0.0, "CO": 0.0, "CO2": 0.0, "H2O": 0.0}
    c.MAX_ADS_CANDIDATES = args.max_candidates
    c.GAS_CACHE_PATH = os.path.join(c.OUTDIR, "gas_cache.json")
    c.SCREEN_ADS = ["CO2", "CO", "H", "COOH", "CHO"]
    return c

# -----------------------------
# Checkpoint helpers
# -----------------------------
def ckpt_path(workdir):
    return os.path.join(workdir, "checkpoint.json")

def load_ckpt(workdir):
    p = ckpt_path(workdir)
    if os.path.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return {}

def save_ckpt(workdir, data):
    p = ckpt_path(workdir)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    shutil.move(tmp, p)

def load_global_ckpt(outdir):
    p = os.path.join(outdir, "global_checkpoint.json")
    if os.path.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return {"completed_files": []}

def save_global_ckpt(outdir, data):
    p = os.path.join(outdir, "global_checkpoint.json")
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    shutil.move(tmp, p)

# -----------------------------
# Suppress warnings
# -----------------------------
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`.*")
logging.getLogger("mace").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("ase").setLevel(logging.ERROR)
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# -----------------------------
# CRITICAL: Detect engine from argv and patch MPI BEFORE any ASE/pymatgen imports.
# If gpaw is installed, importing ASE can trigger gpaw.mpi initialization.
# This must happen first to prevent that.
# -----------------------------
def _early_detect_engine():
    """Quick scan of sys.argv to detect engine before argparse runs."""
    for a in sys.argv[1:]:
        if a in ("mace", "gpaw"):
            return a
    return "mace"  # default fallback

def force_ase_serial():
    """Replace ASE's MPI world with a safe serial dummy."""
    # Purge any already-loaded gpaw modules
    for mod in list(sys.modules.keys()):
        if mod == "_gpaw" or mod.startswith("gpaw"):
            sys.modules.pop(mod, None)
    import ase.parallel as ap
    dummy = ap.DummyMPI()
    # Patch all methods that MACE / ASE internals may call on world
    if not hasattr(dummy, "sum_scalar"):
        dummy.sum_scalar = lambda x, root=-1: x
    if not hasattr(dummy, "sum"):
        dummy.sum = lambda a, root=-1: a
    if not hasattr(dummy, "broadcast"):
        dummy.broadcast = lambda a, root=0: a
    if not hasattr(dummy, "barrier"):
        dummy.barrier = lambda: None
    if not hasattr(dummy, "gather"):
        dummy.gather = lambda a, root=0: [a]
    ap.world = dummy
    ap.rank = 0
    ap.size = 1

_EARLY_ENGINE = _early_detect_engine()
if _EARLY_ENGINE == "mace":
    force_ase_serial()

# -----------------------------
# Imports (now safe -- MPI is already patched if engine=mace)
# -----------------------------
from pymatgen.io.cif import CifParser
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core.surface import generate_all_slabs
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Molecule

from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ase.io import write, read as ase_read

GPAW_AVAILABLE = False

def ensure_gpaw(cfg):
    global GPAW_AVAILABLE
    if GPAW_AVAILABLE:
        return
    if cfg.ENGINE != "gpaw":
        return
    try:
        from gpaw import GPAW, PW  # noqa: F401
        from gpaw.occupations import FermiDirac  # noqa: F401
        from gpaw.poisson import PoissonSolver  # noqa: F401
        GPAW_AVAILABLE = True
    except Exception as e:
        raise ImportError("GPAW import failed: %s" % e)

# -----------------------------
# Utilities
# -----------------------------
def tierB(cfg, E, key, is_ads=False):
    if not cfg.USE_TIER_B:
        return E
    return E + (cfg.ADS_CORR.get(key, 0.0) if is_ads else cfg.GAS_CORR.get(key, 0.0))

def relax_atoms(atoms, calc, fmax=0.05, traj=None, steps=None, label=None):
    atoms.calc = calc
    opt = BFGS(atoms, trajectory=traj, logfile=None)
    t0 = time.time()
    sc = [0]

    def cb():
        sc[0] += 1
        e = atoms.get_potential_energy()
        f = atoms.get_forces()
        fm = np.max(np.linalg.norm(f, axis=1)) if len(f) > 0 else 0.0
        tag = "[%s] " % label if label else ""
        print("    %sStep %3d | E=%12.6f eV | fmax=%.6f | %.1fs" % (tag, sc[0], e, fm, time.time() - t0), flush=True)

    opt.attach(cb)
    if steps is None:
        opt.run(fmax=fmax)
    else:
        opt.run(fmax=fmax, steps=steps)

    E_f = atoms.get_potential_energy()
    st = "converged" if opt.converged() else ("max %d steps" % (steps or 9999))
    tag = "[%s] " % label if label else ""
    print("    %sDone: E=%.6f eV, %d steps, %s, %.1fs" % (tag, E_f, sc[0], st, time.time() - t0), flush=True)
    return E_f

def cluster_layers_by_z(zvals, tol):
    order = np.argsort(zvals)
    zs = np.array(zvals)[order]
    lab = np.zeros(len(zs), dtype=int)
    lid = 0
    for i in range(1, len(zs)):
        if abs(zs[i] - zs[i - 1]) > tol:
            lid += 1
        lab[i] = lid
    centers = [float(np.mean(zs[lab == j])) for j in range(lid + 1)]
    out = np.empty_like(lab)
    out[order] = lab
    return centers, out

def bottom_n_layers(atoms, n, tol):
    z = atoms.get_positions()[:, 2]
    centers, labels = cluster_layers_by_z(z, tol)
    if len(centers) < n:
        raise ValueError("Only %d layers, need %d" % (len(centers), n))
    return [i for i, l in enumerate(labels) if l < n], centers

def map_fixed_ads(ads_ase, centers, n_fix, buf=1.5):
    z = ads_ase.get_positions()[:, 2]
    ca = np.array(centers, dtype=float)
    zt = float(np.max(ca))
    is_a = z > (zt + buf)
    near = np.array([int(np.argmin(np.abs(ca - zi))) for zi in z])
    return [i for i in range(len(ads_ase)) if (near[i] < n_fix and not is_a[i])]

def build_adsorbates():
    a = {}
    a["CO2"] = Molecule(["O", "C", "O"], [[-1.16, 0, 0], [0, 0, 0], [1.16, 0, 0]])
    a["CO"] = Molecule(["C", "O"], [[0, 0, 0], [1.13, 0, 0]])
    a["H"] = Molecule(["H"], [[0, 0, 0]])
    a["COOH"] = Molecule(["H", "O", "C", "O"], [[0, 0, 0], [0.97, 0, 0], [2.20, 0, 0], [3.36, 0, 0]])
    a["CHO"] = Molecule(["H", "C", "O"], [[0, 0, 0], [1.10, 0, 0], [2.33, 0, 0]])
    a["CH2O"] = Molecule(["H", "H", "C", "O"], [[0, .9, 0], [0, -.9, 0], [1.10, 0, 0], [2.33, 0, 0]])
    return a

def build_gas_mols():
    bx = [25.0, 25.0, 25.0]
    g = {}
    g["H2"] = Atoms("H2", positions=[[12, 12, 12], [12.74, 12, 12]], cell=bx, pbc=True)
    g["CO"] = Atoms("CO", positions=[[12, 12, 12], [13.13, 12, 12]], cell=bx, pbc=True)
    g["CO2"] = Atoms("OCO", positions=[[10.84, 12, 12], [12, 12, 12], [13.16, 12, 12]], cell=bx, pbc=True)
    g["H2O"] = Atoms("H2O", positions=[[12, 12, 12], [12.96, 12, 12], [11.76, 12.76, 12]], cell=bx, pbc=True)
    return g

# -----------------------------
# Calculators
# -----------------------------
def mk_mace(cfg):
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=[cfg.MACE_MODEL_PATH], device=cfg.MACE_DEVICE, default_dtype=cfg.MACE_DTYPE)

def mk_gpaw(cfg, txt, kpts, is_slab=True):
    ensure_gpaw(cfg)
    from gpaw import GPAW, PW
    from gpaw.occupations import FermiDirac
    from gpaw.poisson import PoissonSolver
    kw = dict(mode=PW(cfg.PW_CUTOFF), xc=cfg.XC, kpts=kpts, txt=txt, symmetry="off",
              occupations=FermiDirac(0.10),
              convergence={"energy": 1e-5, "density": 1e-4, "eigenstates": 1e-8})
    if is_slab:
        kw["poissonsolver"] = PoissonSolver(dipolelayer="xy")
    return GPAW(**kw)

def do_relax(cfg, atoms, workdir, base, tag, i=None, mace_calc=None, is_slab=True):
    """Unified relax: picks engine, builds calc, relaxes, returns energy."""
    suffix = tag if i is None else "%s_cand%d" % (tag, i)
    label = suffix
    if cfg.ENGINE == "mace":
        if mace_calc is None:
            mace_calc = mk_mace(cfg)
        traj = os.path.join(workdir, "%s_%s.traj" % (base, suffix))
        return relax_atoms(atoms, mace_calc, fmax=cfg.MACE_FMAX, steps=cfg.MACE_STEPS, traj=traj, label=label)
    else:
        txt = os.path.join(workdir, "%s_%s.txt" % (base, suffix))
        traj = os.path.join(workdir, "%s_%s.traj" % (base, suffix))
        calc = mk_gpaw(cfg, txt=txt, kpts=cfg.KPTS_SLAB if is_slab else cfg.KPTS_MOL, is_slab=is_slab)
        return relax_atoms(atoms, calc, fmax=cfg.FMAX, traj=traj, label=label)

# -----------------------------
# Gas refs
# -----------------------------
def gas_key(cfg):
    if cfg.ENGINE == "mace":
        return "mace::%s::dtype=%s" % (cfg.MACE_MODEL_PATH, cfg.MACE_DTYPE)
    return "gpaw::PW%d::XC%s::k%s" % (cfg.PW_CUTOFF, cfg.XC, cfg.KPTS_MOL)

def compute_gas(cfg, workdir, mace_calc=None):
    cache = {}
    if os.path.exists(cfg.GAS_CACHE_PATH):
        with open(cfg.GAS_CACHE_PATH, "r") as f:
            cache = json.load(f)
    key = gas_key(cfg)
    if key in cache:
        plog("Gas refs from cache")
        for n, E in cache[key].items():
            print("    %s: %.6f eV" % (n, E), flush=True)
        return cache[key]

    pbar_header("Computing gas molecule reference energies")
    gas = build_gas_mols()
    gas_E = {}
    names = list(gas.keys())
    for idx, name in enumerate(names, 1):
        plog("[%d/%d] Gas: %s" % (idx, len(names), name), "RUN")
        atoms = gas[name]
        if cfg.ENGINE == "mace":
            if mace_calc is None:
                mace_calc = mk_mace(cfg)
            traj = os.path.join(workdir, "gas_%s.traj" % name)
            E = relax_atoms(atoms, mace_calc, fmax=cfg.MACE_FMAX, traj=traj, steps=cfg.MACE_STEPS,
                            label="Gas-%s" % name)
        else:
            txt = os.path.join(workdir, "gas_%s.txt" % name)
            traj = os.path.join(workdir, "gas_%s.traj" % name)
            calc = mk_gpaw(cfg, txt=txt, kpts=cfg.KPTS_MOL, is_slab=False)
            E = relax_atoms(atoms, calc, fmax=cfg.FMAX, traj=traj, label="Gas-%s" % name)
        gas_E[name] = tierB(cfg, E, name)
        plog("%s: %.6f eV" % (name, gas_E[name]))

    cache[key] = gas_E
    with open(cfg.GAS_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)
    plog("Gas refs cached")
    return gas_E

# -----------------------------
# Slab
# -----------------------------
def relax_slab_term(cfg, slab_pm, adaptor, workdir, base, tid, mace_calc=None):
    slab_pm = slab_pm.copy()
    slab_pm.make_supercell(cfg.SLAB_SUPERCELL)
    sa = adaptor.get_atoms(slab_pm)
    fi, centers = bottom_n_layers(sa, cfg.FIX_N_LAYERS, cfg.LAYER_TOL)
    sa.set_constraint(FixAtoms(indices=fi))
    na = len(sa)
    plog("  Term %d: %d atoms (fixed %d, free %d), %d layers" % (tid, na, len(fi), na - len(fi), len(centers)), "STEP")
    E = do_relax(cfg, sa, workdir, base, "slab_term%d" % tid, mace_calc=mace_calc, is_slab=True)
    return E, sa, fi, centers

def pick_best_term(cfg, slabs, adaptor, workdir, base, mace_calc=None):
    pbar_header("Selecting best termination (%d candidates)" % len(slabs))
    best = None
    for j, sp in enumerate(slabs):
        plog("Term %d/%d ..." % (j + 1, len(slabs)), "RUN")
        try:
            E, sa, fi, ct = relax_slab_term(cfg, sp, adaptor, workdir, base, j, mace_calc)
            plog("Term %d: E = %.6f eV" % (j, E))
            if best is None or E < best[0]:
                best = (E, sa, fi, ct, j)
        except Exception as e:
            plog("Term %d failed: %s" % (j, e), "WARN")
    if best is None:
        raise ValueError("All terminations failed")
    plog("Best: term %d, E = %.6f eV" % (best[4], best[0]), "DONE")
    return best

# -----------------------------
# Descriptors
# -----------------------------
def compute_metric(name, E_ads, E_slab, gE):
    m = {"H":    ("dG_*H",          E_ads - E_slab - 0.5 * gE["H2"]),
         "CO":   ("dG_*CO",         E_ads - E_slab - gE["CO"]),
         "CO2":  ("dG_*CO2",        E_ads - E_slab - gE["CO2"]),
         "COOH": ("dG_*COOH",       E_ads - E_slab - gE["CO2"] - 0.5 * gE["H2"]),
         "CHO":  ("dG_*CHO_proxy",  E_ads - E_slab - gE["CO"] - 0.5 * gE["H2"]),
         "CH2O": ("dG_*CH2O_proxy", E_ads - E_slab - gE["CO"] - 1.0 * gE["H2"])}
    if name not in m:
        raise ValueError("Unknown: %s" % name)
    return m[name]

def ads_key(n):
    return {"H": "*H", "CO2": "*CO2", "CO": "*CO", "COOH": "*COOH", "CHO": "*CHO", "CH2O": "*CH2O"}.get(n, "*%s" % n)

# -----------------------------
# Core per-structure
# -----------------------------
def process_structure(cfg, cif_file, mace_calc=None):
    base = os.path.splitext(os.path.basename(cif_file))[0]
    workdir = os.path.join(cfg.OUTDIR, base)
    os.makedirs(workdir, exist_ok=True)
    t0 = time.time()

    # Checkpoint
    ck = load_ckpt(workdir) if cfg.RESUME else {}
    done_ads = set(ck.get("completed_adsorbates", []))
    results = ck.get("results", [])
    slab_ok = ck.get("slab_done", False)

    pbar_header("Processing: %s" % base)
    plog("CIF: %s" % cif_file)
    plog("Output: %s" % workdir)
    if done_ads:
        plog("Resuming -- done: %s" % ", ".join(sorted(done_ads)))

    # Copy CIF
    dst = os.path.join(workdir, os.path.basename(cif_file))
    if not os.path.exists(dst):
        shutil.copy2(cif_file, dst)

    adaptor = AseAtomsAdaptor()
    ads_mols = build_adsorbates()
    gas_E = compute_gas(cfg, workdir, mace_calc=mace_calc)

    # Parse CIF
    pbar_header("Parsing CIF / generating slabs")
    parser = CifParser(cif_file)
    struct = parser.get_structures()[0]
    plog("Crystal: %s, SG: %s" % (struct.composition.reduced_formula, struct.get_space_group_info()[0]))

    plog("Generating slabs ...", "RUN")
    slabs = generate_all_slabs(struct, max_index=cfg.MAX_INDEX, min_slab_size=cfg.MIN_SLAB_SIZE,
                               min_vacuum_size=cfg.MIN_VACUUM, center_slab=True, max_normal_search=15)
    plog("Total slabs: %d" % len(slabs))

    target = [s for s in slabs if tuple(s.miller_index) == tuple(cfg.TARGET_MILLER)]
    if not target:
        raise ValueError("No slab for miller=%s" % str(cfg.TARGET_MILLER))
    plog("Miller %s: %d terminations" % (cfg.TARGET_MILLER, len(target)))

    # Slab
    slab_xyz = os.path.join(workdir, "%s_slab_relaxed.xyz" % base)
    if slab_ok and "E_slab" in ck and os.path.exists(slab_xyz):
        plog("Slab from checkpoint: E = %.6f eV" % ck["E_slab"])
        E_slab = ck["E_slab"]
        term_id = ck["term_id"]
        centers = ck["centers"]
        slab_ase = ase_read(slab_xyz)
    else:
        Er, slab_ase, fi, centers, term_id = pick_best_term(cfg, target, adaptor, workdir, base, mace_calc)
        E_slab = tierB(cfg, Er, "*", is_ads=True)
        write(slab_xyz, slab_ase)
        plog("Slab E = %.6f eV (term %d)" % (E_slab, term_id))
        ck.update({"slab_done": True, "E_slab": float(E_slab), "term_id": int(term_id),
                   "centers": [float(c) for c in centers]})
        save_ckpt(workdir, ck)

    # Adsorption
    slab_pm = adaptor.get_structure(slab_ase)
    asf = AdsorbateSiteFinder(slab_pm)

    for ai, aname in enumerate(cfg.SCREEN_ADS, 1):
        if aname in done_ads:
            plog("[%d/%d] %s -- SKIP (done)" % (ai, len(cfg.SCREEN_ADS), aname))
            continue

        amol = ads_mols[aname]
        pbar_header("Adsorbate [%d/%d]: %s" % (ai, len(cfg.SCREEN_ADS), aname))

        try:
            plog("Generating sites ...", "RUN")
            structs = asf.generate_adsorption_structures(amol, repeat=[1, 1, 1])
            if not structs:
                plog("No sites for %s" % aname, "WARN")
                done_ads.add(aname)
                ck["completed_adsorbates"] = list(done_ads)
                save_ckpt(workdir, ck)
                continue

            if cfg.MAX_ADS_CANDIDATES and len(structs) > cfg.MAX_ADS_CANDIDATES:
                plog("Truncating %d -> %d" % (len(structs), cfg.MAX_ADS_CANDIDATES), "WARN")
                structs = structs[:cfg.MAX_ADS_CANDIDATES]
            plog("%d candidates" % len(structs))

            cands = []
            for i, st in enumerate(structs):
                aa = adaptor.get_atoms(st)
                fi = map_fixed_ads(aa, centers, cfg.FIX_N_LAYERS)
                aa.set_constraint(FixAtoms(indices=fi))
                plog("  Cand %d/%d ..." % (i + 1, len(structs)), "RUN")
                try:
                    E = do_relax(cfg, aa, workdir, base, aname, i=i, mace_calc=mace_calc, is_slab=True)
                    xyz = os.path.join(workdir, "%s_%s_cand%d_relaxed.xyz" % (base, aname, i))
                    write(xyz, aa)
                    cands.append((E, i, aa, xyz))
                    plog("  Cand %d: E = %.6f eV" % (i, E))
                except Exception as e:
                    plog("  Cand %d failed: %s" % (i, e), "WARN")

            if not cands:
                plog("All failed for %s" % aname, "WARN")
                done_ads.add(aname)
                ck["completed_adsorbates"] = list(done_ads)
                save_ckpt(workdir, ck)
                continue

            cands.sort(key=lambda x: x[0])
            plog("Ranking:")
            for r, (E, idx, _, _) in enumerate(cands):
                mk = " <-- BEST" if r == 0 else ""
                print("    #%d: cand%d, E=%.6f eV%s" % (r + 1, idx, E, mk), flush=True)

            Eb, bi, ba, bx = cands[0]
            Ebc = tierB(cfg, Eb, ads_key(aname), is_ads=True)
            mname, dG = compute_metric(aname, Ebc, E_slab, gas_E)

            best_f = os.path.join(workdir, "%s_%s_BEST_cand%d.xyz" % (base, aname, bi))
            if bx != best_f:
                shutil.copy2(bx, best_f)

            plog("* %s BEST: cand%d, %s = %.4f eV" % (aname, bi, mname, dG), "DONE")

            results.append({
                "structure": base, "miller": str(cfg.TARGET_MILLER),
                "termination_id": int(ck.get("term_id", 0)),
                "adsorbate": aname, "best_site_index": bi,
                "engine": cfg.ENGINE, "E_slab_eV": float(E_slab),
                "E_ads_sys_eV": float(Ebc), "metric": mname,
                "value_eV": float(dG), "best_xyz": os.path.basename(best_f),
            })

            done_ads.add(aname)
            ck["completed_adsorbates"] = list(done_ads)
            ck["results"] = results
            save_ckpt(workdir, ck)
            plog("Checkpoint saved (%s done)" % ", ".join(sorted(done_ads)))

        except Exception as e:
            plog("%s failed: %s" % (aname, e), "ERROR")

    # CSV
    csv_p = os.path.join(workdir, "%s_descriptors.csv" % base)
    flds = ["structure", "miller", "termination_id", "adsorbate", "best_site_index",
            "engine", "E_slab_eV", "E_ads_sys_eV", "metric", "value_eV", "best_xyz"]
    with open(csv_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        for row in results:
            w.writerow(row)

    el = time.time() - t0
    pbar_header("%s COMPLETE" % base)
    plog("Time: %.1f sec (%.1f min)" % (el, el / 60.0))
    plog("CSV: %s" % csv_p)

    if results:
        print("", flush=True)
        print("  +----------+----------------+--------------+", flush=True)
        print("  | Adsorb.  | Descriptor     | Value (eV)   |", flush=True)
        print("  +----------+----------------+--------------+", flush=True)
        for r in results:
            print("  | %-8s | %-14s | %+12.4f |" % (r["adsorbate"], r["metric"], r["value_eV"]), flush=True)
        print("  +----------+----------------+--------------+", flush=True)

    plog("Files in %s:" % workdir)
    for fn in sorted(os.listdir(workdir)):
        sz = os.path.getsize(os.path.join(workdir, fn))
        print("    %-50s %s" % (fn, "%.1fKB" % (sz / 1024.0) if sz > 1024 else "%dB" % sz), flush=True)

    ck["fully_done"] = True
    save_ckpt(workdir, ck)

# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    cfg = build_config(args)
    os.makedirs(cfg.OUTDIR, exist_ok=True)

    logging.basicConfig(filename=os.path.join(cfg.OUTDIR, "calculation.log"),
                        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    cifs = sorted(glob.glob(os.path.join(cfg.CIF_FOLDER, "*.cif")))
    if not cifs:
        print("[ERR] No .cif files in %s" % cfg.CIF_FOLDER, flush=True)
        sys.exit(1)

    gck = load_global_ckpt(cfg.OUTDIR) if cfg.RESUME else {"completed_files": []}
    done_f = set(gck["completed_files"])

    pbar_header("High-throughput adsorption energy calculation")
    plog("Engine     : %s" % cfg.ENGINE)
    plog("CIF folder : %s" % cfg.CIF_FOLDER)
    plog("Output     : %s" % cfg.OUTDIR)
    plog("Miller     : %s" % str(cfg.TARGET_MILLER))
    plog("Supercell  : %s" % str(cfg.SLAB_SUPERCELL))
    plog("Resume     : %s" % cfg.RESUME)
    plog("Total CIFs : %d" % len(cifs))
    for i, f in enumerate(cifs, 1):
        bn = os.path.basename(f)
        tag = " [DONE]" if bn in done_f else ""
        print("  %d. %s%s" % (i, bn, tag), flush=True)

    mc = None
    if cfg.ENGINE == "mace":
        plog("Loading MACE model ...", "RUN")
        mc = mk_mace(cfg)
        plog("MACE loaded", "DONE")
    else:
        ensure_gpaw(cfg)

    for fi, cf in enumerate(cifs, 1):
        bn = os.path.basename(cf)
        if cfg.RESUME and bn in done_f:
            plog("[%d/%d] %s -- SKIP (done)" % (fi, len(cifs), bn))
            continue

        pbar_header("File [%d/%d]: %s" % (fi, len(cifs), bn))
        try:
            process_structure(cfg, cf, mace_calc=mc)
            done_f.add(bn)
            gck["completed_files"] = list(done_f)
            save_global_ckpt(cfg.OUTDIR, gck)
        except Exception as e:
            plog("%s FAILED: %s" % (bn, e), "ERROR")

    pbar_header("ALL TASKS COMPLETE")
    plog("Results: %s" % cfg.OUTDIR)
    print("", flush=True)
    for cf in cifs:
        b = os.path.splitext(os.path.basename(cf))[0]
        cp = os.path.join(cfg.OUTDIR, b, "%s_descriptors.csv" % b)
        if os.path.exists(cp):
            print("  [OK] %-20s -> %s" % (b, cp), flush=True)
        else:
            print("  [--] %-20s    (no results)" % b, flush=True)

if __name__ == "__main__":
    main()
