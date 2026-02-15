# -*- coding: utf-8 -*-
"""
High-throughput adsorption energy calculator (refactored for robustness).

Key fixes implemented (per your request):
1) NO "truncate first N sites": if too many candidates,
   - MACE engine: cheap pre-ranking (single-point) -> select topK -> full relax.
   - GPAW engine: geometric bucketing (nearest surface atom) -> take reps -> (optional) deterministic downselect.
2) Remove hard-coded MACE model path:
   - default from env var MACE_MODEL_PATH, otherwise must pass --mace-model.
3) Fix constraint logic:
   - Do NOT infer adsorbate atoms by z threshold.
   - Build adsorption structures by placing adsorbate with ASE add_adsorbate so indices are stable:
     [0:n_slab) slab, [n_slab:] adsorbate. Only fix slab bottom layers indices.
4) Add *OCHO (formate key intermediate) to adsorbate set (default in SCREEN_ADS).
5) Separate raw energies and descriptor values in output.
6) k-points: compute from k-spacing (default 0.22 Å^-1) instead of fixed (3,3,1).
7) Add simple file locks for gas_cache/global_checkpoint to avoid concurrent corruption.

Usage:
  python job.py <cif_folder> <engine: mace|gpaw> [options]

Examples:
  python job.py ./cifs mace --mace-model /path/to/model --mace-device cuda
  python job.py ./cifs gpaw --output results_gpaw
  python job.py ./cifs mace --resume
  python job.py ./cifs mace --miller 1,1,1 --supercell 3,3,1 --max-candidates 24

Output structure:
  ht_results/
    global_checkpoint.json
    gas_cache.json
    calculation.log
    CuPd/
      CuPd.cif
      CuPd_slab_relaxed.xyz
      CuPd_CO2_best.xyz
      CuPd_CO2_ranked.json
      CuPd_descriptors.csv
      checkpoint.json
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
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

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
# Very small file lock (stdlib)
# -----------------------------
class SimpleFileLock:
    """
    Cross-process lock using atomic create of lockfile.
    Works on shared filesystems better than relying on in-process locks.
    """
    def __init__(self, lock_path: str, timeout: float = 600.0, poll: float = 0.2):
        self.lock_path = lock_path
        self.timeout = timeout
        self.poll = poll
        self._fd = None

    def __enter__(self):
        t0 = time.time()
        while True:
            try:
                # O_EXCL guarantees atomic creation
                self._fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(self._fd, str(os.getpid()).encode("utf-8"))
                return self
            except FileExistsError:
                if time.time() - t0 > self.timeout:
                    raise TimeoutError(f"Timeout acquiring lock: {self.lock_path}")
                time.sleep(self.poll)

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._fd is not None:
                os.close(self._fd)
        finally:
            self._fd = None
            try:
                os.remove(self.lock_path)
            except FileNotFoundError:
                pass

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

    # MACE model path: default from env, otherwise must pass explicitly
    default_model = os.getenv("MACE_MODEL_PATH", "")
    p.add_argument("--mace-model", default=default_model, help="Path to MACE model file (or set env MACE_MODEL_PATH)")
    p.add_argument("--mace-device", default="cpu", help="MACE device: cpu or cuda (default: cpu)")
    p.add_argument("--mace-dtype", default="float64", choices=["float32", "float64"],
                   help="MACE default dtype (default: float64)")
    p.add_argument("--fmax-mace", type=float, default=0.10, help="MACE fmax (default: 0.10)")
    p.add_argument("--fmax-gpaw", type=float, default=0.05, help="GPAW fmax (default: 0.05)")
    p.add_argument("--max-steps", type=int, default=200, help="Max BFGS steps (default: 200)")

    # Candidate control
    p.add_argument("--max-candidates", type=int, default=12, help="Max adsorption candidates per adsorbate")
    p.add_argument("--pre-rank-steps", type=int, default=0,
                   help="MACE only: if >0 do short relax for pre-ranking; else single-point pre-ranking (default: 0)")
    p.add_argument("--pre-rank-fmax", type=float, default=0.30,
                   help="MACE only: fmax for short pre-ranking relax (default: 0.30)")

    # k-point control via kspacing
    p.add_argument("--kspacing", type=float, default=0.22,
                   help="Target k-point spacing in 1/Angstrom for slabs (default: 0.22).")

    # Adsorbates selection
    p.add_argument("--adsorbates", default="CO2,CO,H,COOH,CHO,OCHO",
                   help="Comma-separated adsorbates to screen (default: CO2,CO,H,COOH,CHO,OCHO)")

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
    c.MACE_DTYPE = args.mace_dtype
    c.MACE_FMAX = args.fmax_mace
    c.MACE_STEPS = args.max_steps
    c.PRE_RANK_STEPS = args.pre_rank_steps
    c.PRE_RANK_FMAX = args.pre_rank_fmax

    # GPAW defaults
    c.PW_CUTOFF = 470
    c.XC = "PBE"
    c.FMAX = args.fmax_gpaw

    # k-point spacing control
    c.KSPACING = float(args.kspacing)
    c.KPTS_MOL = (1, 1, 1)

    # Slab generation
    c.MAX_INDEX = 1
    c.MIN_SLAB_SIZE = 6.0
    c.MIN_VACUUM = 12.0
    c.FIX_N_LAYERS = 3
    c.LAYER_TOL = 0.12

    # "Tier-B" corrections (placeholders; keep as configurable)
    c.USE_TIER_B = True
    c.ADS_CORR = {"*H": 0.0, "*CO": 0.0, "*COOH": 0.0, "*CHO": 0.0, "*CH2O": 0.0, "*CO2": 0.0, "*OCHO": 0.0, "*": 0.0}
    c.GAS_CORR = {"H2": 0.0, "CO": 0.0, "CO2": 0.0, "H2O": 0.0}

    c.MAX_ADS_CANDIDATES = int(args.max_candidates)

    # Cache & checkpoint
    c.GAS_CACHE_PATH = os.path.join(c.OUTDIR, "gas_cache.json")
    c.GAS_CACHE_LOCK = c.GAS_CACHE_PATH + ".lock"
    c.GLOBAL_CKPT_PATH = os.path.join(c.OUTDIR, "global_checkpoint.json")
    c.GLOBAL_CKPT_LOCK = c.GLOBAL_CKPT_PATH + ".lock"

    # Adsorbates
    c.SCREEN_ADS = [x.strip() for x in args.adsorbates.split(",") if x.strip()]

    # Heights for ASE add_adsorbate placement (Angstrom)
    c.ADS_HEIGHT = {
        "H": 1.2,
        "CO": 1.8,
        "CO2": 2.0,
        "COOH": 2.0,
        "CHO": 1.8,
        "CH2O": 2.0,
        "OCHO": 2.0,
    }

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

def load_global_ckpt(cfg):
    p = cfg.GLOBAL_CKPT_PATH
    if os.path.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return {"completed_files": []}

def save_global_ckpt(cfg, data):
    p = cfg.GLOBAL_CKPT_PATH
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
# CRITICAL: Detect engine early & patch MPI BEFORE importing ASE/pymatgen
# -----------------------------
def _early_detect_engine():
    for a in sys.argv[1:]:
        if a in ("mace", "gpaw"):
            return a
    return "mace"

def force_ase_serial():
    for mod in list(sys.modules.keys()):
        if mod == "_gpaw" or mod.startswith("gpaw"):
            sys.modules.pop(mod, None)
    import ase.parallel as ap
    dummy = ap.DummyMPI()
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
# Imports (safe now)
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
from ase.build import add_adsorbate

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
    if is_ads:
        return E + cfg.ADS_CORR.get(key, 0.0)
    return E + cfg.GAS_CORR.get(key, 0.0)

def compute_kpts_from_kspacing(atoms: Atoms, kspacing: float, slab_xy_only: bool = True) -> Tuple[int, int, int]:
    """
    Compute k-points from target kspacing in 1/Å.
    Approx: N_i = max(1, ceil(|b_i| / (2π * kspacing))) where b_i is reciprocal vector magnitude in 1/Å.
    For slabs, typically set kz=1.
    """
    cell = np.array(atoms.get_cell())
    if np.linalg.det(cell) == 0:
        return (1, 1, 1)
    recip = 2.0 * math.pi * np.linalg.inv(cell).T  # reciprocal vectors in 1/Å
    bmag = np.linalg.norm(recip, axis=1)
    nk = []
    for i in range(3):
        n = int(max(1, math.ceil(bmag[i] / (2.0 * math.pi * kspacing))))
        nk.append(n)
    if slab_xy_only:
        nk[2] = 1
    return (nk[0], nk[1], nk[2])

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

def bottom_n_layers_indices(atoms, n, tol):
    """
    Return slab indices belonging to bottom n layers (based on z clustering).
    """
    z = atoms.get_positions()[:, 2]
    centers, labels = cluster_layers_by_z(z, tol)
    if len(centers) < n:
        raise ValueError("Only %d layers, need %d" % (len(centers), n))
    fixed = [i for i, l in enumerate(labels) if l < n]
    return fixed, centers

def ads_key(n):
    return {"H": "*H", "CO2": "*CO2", "CO": "*CO", "COOH": "*COOH", "CHO": "*CHO", "CH2O": "*CH2O", "OCHO": "*OCHO"}.get(n, "*%s" % n)

# -----------------------------
# Adsorbates / gas molecules
# -----------------------------
def build_adsorbates() -> Dict[str, Molecule]:
    a: Dict[str, Molecule] = {}
    a["CO2"] = Molecule(["O", "C", "O"], [[-1.16, 0, 0], [0, 0, 0], [1.16, 0, 0]])
    a["CO"] = Molecule(["C", "O"], [[0, 0, 0], [1.13, 0, 0]])
    a["H"] = Molecule(["H"], [[0, 0, 0]])
    a["COOH"] = Molecule(["H", "O", "C", "O"], [[0, 0, 0], [0.97, 0, 0], [2.20, 0, 0], [3.36, 0, 0]])
    a["CHO"] = Molecule(["H", "C", "O"], [[0, 0, 0], [1.10, 0, 0], [2.33, 0, 0]])
    a["CH2O"] = Molecule(["H", "H", "C", "O"], [[0, .9, 0], [0, -.9, 0], [1.10, 0, 0], [2.33, 0, 0]])

    # OCHO (formate-like): minimal geometry seed.
    # NOTE: This is a starting guess; high-throughput will relax anyway.
    # One reasonable seed: H attached to one O, O-C-O backbone.
    a["OCHO"] = Molecule(
        ["H", "O", "C", "O"],
        [
            [0.00, 0.00, 0.00],   # H
            [0.95, 0.00, 0.00],   # O(H)
            [2.20, 0.00, 0.00],   # C
            [3.35, 0.00, 0.00],   # O
        ],
    )
    return a

def build_gas_mols() -> Dict[str, Atoms]:
    bx = [25.0, 25.0, 25.0]
    g: Dict[str, Atoms] = {}
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
    kw = dict(
        mode=PW(cfg.PW_CUTOFF),
        xc=cfg.XC,
        kpts=kpts,
        txt=txt,
        symmetry="off",
        occupations=FermiDirac(0.10),
        convergence={"energy": 1e-5, "density": 1e-4, "eigenstates": 1e-8},
    )
    if is_slab:
        kw["poissonsolver"] = PoissonSolver(dipolelayer="xy")
    return GPAW(**kw)

def do_relax(cfg, atoms, workdir, base, tag, i=None, mace_calc=None, is_slab=True, gpaw_kpts=None, fmax=None, steps=None):
    """
    Unified relax: picks engine, builds calc, relaxes, returns energy.
    """
    suffix = tag if i is None else "%s_cand%d" % (tag, i)
    label = suffix
    if cfg.ENGINE == "mace":
        if mace_calc is None:
            mace_calc = mk_mace(cfg)
        traj = os.path.join(workdir, "%s_%s.traj" % (base, suffix))
        return relax_atoms(
            atoms, mace_calc,
            fmax=(fmax if fmax is not None else cfg.MACE_FMAX),
            steps=(steps if steps is not None else cfg.MACE_STEPS),
            traj=traj, label=label
        )
    else:
        txt = os.path.join(workdir, "%s_%s.txt" % (base, suffix))
        traj = os.path.join(workdir, "%s_%s.traj" % (base, suffix))
        kpts = gpaw_kpts if gpaw_kpts is not None else compute_kpts_from_kspacing(atoms, cfg.KSPACING, slab_xy_only=is_slab)
        calc = mk_gpaw(cfg, txt=txt, kpts=kpts, is_slab=is_slab)
        return relax_atoms(
            atoms, calc,
            fmax=(fmax if fmax is not None else cfg.FMAX),
            traj=traj, label=label
        )

def mace_single_point_energy(atoms: Atoms, mace_calc) -> float:
    aa = atoms.copy()
    aa.calc = mace_calc
    return float(aa.get_potential_energy())

def mace_short_relax_energy(cfg, atoms: Atoms, mace_calc, steps: int, fmax: float) -> float:
    aa = atoms.copy()
    # Use the same relax routine but with limited steps and looser fmax
    return float(relax_atoms(aa, mace_calc, fmax=fmax, steps=steps, traj=None, label="pre-rank"))

# -----------------------------
# Gas refs (with lock)
# -----------------------------
def gas_key(cfg):
    if cfg.ENGINE == "mace":
        return "mace::%s::dtype=%s" % (cfg.MACE_MODEL_PATH, cfg.MACE_DTYPE)
    return "gpaw::PW%d::XC%s::kspacing=%.3f" % (cfg.PW_CUTOFF, cfg.XC, cfg.KSPACING)

def compute_gas(cfg, workdir, mace_calc=None) -> Dict[str, float]:
    cache: Dict[str, Any] = {}
    with SimpleFileLock(cfg.GAS_CACHE_LOCK, timeout=600):
        if os.path.exists(cfg.GAS_CACHE_PATH):
            with open(cfg.GAS_CACHE_PATH, "r") as f:
                cache = json.load(f)

        key = gas_key(cfg)
        if key in cache:
            plog("Gas refs from cache")
            for n, E in cache[key].items():
                print("    %s: %.6f eV" % (n, E), flush=True)
            return cache[key]

        # Compute and write (still under lock)
        pbar_header("Computing gas molecule reference energies")
        gas = build_gas_mols()
        gas_E: Dict[str, float] = {}
        names = list(gas.keys())
        for idx, name in enumerate(names, 1):
            plog("[%d/%d] Gas: %s" % (idx, len(names), name), "RUN")
            atoms = gas[name]
            if cfg.ENGINE == "mace":
                if mace_calc is None:
                    mace_calc = mk_mace(cfg)
                traj = os.path.join(workdir, "gas_%s.traj" % name)
                E = relax_atoms(atoms, mace_calc, fmax=cfg.MACE_FMAX, traj=traj, steps=cfg.MACE_STEPS, label="Gas-%s" % name)
            else:
                txt = os.path.join(workdir, "gas_%s.txt" % name)
                traj = os.path.join(workdir, "gas_%s.traj" % name)
                calc = mk_gpaw(cfg, txt=txt, kpts=cfg.KPTS_MOL, is_slab=False)
                E = relax_atoms(atoms, calc, fmax=cfg.FMAX, traj=traj, label="Gas-%s" % name)
            gas_E[name] = tierB(cfg, float(E), name, is_ads=False)
            plog("%s: %.6f eV" % (name, gas_E[name]))

        cache[key] = gas_E
        tmp = cfg.GAS_CACHE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cache, f, indent=2)
        shutil.move(tmp, cfg.GAS_CACHE_PATH)
        plog("Gas refs cached")
        return gas_E

# -----------------------------
# Slab selection
# -----------------------------
def relax_slab_term(cfg, slab_pm, adaptor, workdir, base, tid, mace_calc=None):
    slab_pm = slab_pm.copy()
    slab_pm.make_supercell(cfg.SLAB_SUPERCELL)
    sa = adaptor.get_atoms(slab_pm)

    # Fix bottom N layers by z clustering
    fixed_idx, centers = bottom_n_layers_indices(sa, cfg.FIX_N_LAYERS, cfg.LAYER_TOL)
    sa.set_constraint(FixAtoms(indices=fixed_idx))

    na = len(sa)
    plog("  Term %d: %d atoms (fixed %d, free %d), %d layers" % (tid, na, len(fixed_idx), na - len(fixed_idx), len(centers)), "STEP")

    # kpts for gpaw computed from kspacing
    gpaw_kpts = compute_kpts_from_kspacing(sa, cfg.KSPACING, slab_xy_only=True)

    E = do_relax(cfg, sa, workdir, base, "slab_term%d" % tid, mace_calc=mace_calc, is_slab=True, gpaw_kpts=gpaw_kpts)
    return float(E), sa, fixed_idx, centers

def pick_best_term(cfg, slabs, adaptor, workdir, base, mace_calc=None):
    pbar_header("Selecting best termination (%d candidates)" % len(slabs))
    best = None
    for j, sp in enumerate(slabs):
        plog("Term %d/%d ..." % (j + 1, len(slabs)), "RUN")
        try:
            E, sa, fixed_idx, centers = relax_slab_term(cfg, sp, adaptor, workdir, base, j, mace_calc)
            plog("Term %d: E = %.6f eV" % (j, E))
            if best is None or E < best[0]:
                best = (E, sa, fixed_idx, centers, j)
        except Exception as e:
            plog("Term %d failed: %s" % (j, e), "WARN")
    if best is None:
        raise ValueError("All terminations failed")
    plog("Best: term %d, E = %.6f eV" % (best[4], best[0]), "DONE")
    return best

# -----------------------------
# Descriptor calculations
# -----------------------------
def compute_descriptor(name: str, E_ads_sys: float, E_slab: float, gE: Dict[str, float]) -> Tuple[str, float]:
    """
    Return (descriptor_name, value_eV).

    NOTE: These are *descriptor proxies* (DFT total energies + optional constant corrections).
    They are not full electrochemical free energies ΔG(U). We keep them as consistent ranking metrics.
    """
    # Descriptor mapping:
    # - H:  * + 1/2 H2 -> *H
    # - CO: * + CO -> *CO
    # - CO2:* + CO2 -> *CO2
    # - COOH proxy: * + CO2 + 1/2 H2 -> *COOH  (CHE-ish)
    # - OCHO proxy: * + CO2 + 1/2 H2 -> *OCHO  (formate-like key descriptor)
    # - CHO proxy:  * + CO + 1/2 H2 -> *CHO
    m = {
        "H":    ("dE_*H",          E_ads_sys - E_slab - 0.5 * gE["H2"]),
        "CO":   ("dE_*CO",         E_ads_sys - E_slab - gE["CO"]),
        "CO2":  ("dE_*CO2",        E_ads_sys - E_slab - gE["CO2"]),
        "COOH": ("dE_*COOH_proxy", E_ads_sys - E_slab - gE["CO2"] - 0.5 * gE["H2"]),
        "OCHO": ("dE_*OCHO_proxy", E_ads_sys - E_slab - gE["CO2"] - 0.5 * gE["H2"]),
        "CHO":  ("dE_*CHO_proxy",  E_ads_sys - E_slab - gE["CO"] - 0.5 * gE["H2"]),
        "CH2O": ("dE_*CH2O_proxy", E_ads_sys - E_slab - gE["CO"] - 1.0 * gE["H2"]),
    }
    if name not in m:
        raise ValueError("Unknown adsorbate for descriptor: %s" % name)
    return m[name]

# -----------------------------
# Candidate generation without generate_adsorption_structures()
# -----------------------------
def pmg_mol_to_ase(mol: Molecule) -> Atoms:
    species = [str(s) for s in mol.species]
    positions = [list(x) for x in mol.cart_coords]
    return Atoms(symbols=species, positions=positions)

def find_ads_sites(asf: AdsorbateSiteFinder) -> List[List[float]]:
    """
    Get adsorption sites from asf.find_adsorption_sites().
    Use the 'all' sites if available; fall back to other keys.
    """
    d = asf.find_adsorption_sites()
    if not d:
        return []
    # prefer "all"
    if "all" in d and d["all"]:
        return [list(x) for x in d["all"]]
    # otherwise merge any coordinate lists
    coords = []
    for k, v in d.items():
        if isinstance(v, (list, tuple)) and v:
            # might be list of coords
            if isinstance(v[0], (list, tuple, np.ndarray)) and len(v[0]) == 3:
                coords.extend([list(x) for x in v])
    return coords

def build_adsorption_candidates_ase(
    cfg: Config,
    slab_ase: Atoms,
    fixed_slab_indices: List[int],
    adsorbate_ase: Atoms,
    site_coords_cart: List[List[float]],
    height: float,
) -> List[Tuple[int, Atoms, List[float]]]:
    """
    Create candidate adsorbed structures using ASE add_adsorbate.
    Returns list of tuples: (site_id, atoms_with_adsorbate, site_coord)
    """
    cands = []
    for i, xyz in enumerate(site_coords_cart):
        aa = slab_ase.copy()
        n_slab = len(aa)
        # ensure constraints are set ONLY on slab indices
        aa.set_constraint(FixAtoms(indices=fixed_slab_indices))

        # add adsorbate appended at end
        ads = adsorbate_ase.copy()
        # add_adsorbate uses position in xy; provide x,y from site coord
        add_adsorbate(aa, ads, height=height, position=(xyz[0], xyz[1]))

        # re-apply constraint (ASE sometimes keeps it; do it anyway)
        aa.set_constraint(FixAtoms(indices=fixed_slab_indices))

        # sanity: ensure slab atoms are first, ads are last
        if len(aa) <= n_slab:
            # should never happen
            continue
        cands.append((i, aa, xyz))
    return cands

def gpaw_geometric_bucket_downselect(
    slab_ase: Atoms,
    candidates: List[Tuple[int, Atoms, List[float]]],
    max_keep: int
) -> List[Tuple[int, Atoms, List[float]]]:
    """
    GPAW mode downselect without energy pre-ranking:
    bucket by nearest surface atom (in xy) + element, pick one per bucket.
    This avoids "take first N" and is deterministic.
    """
    if len(candidates) <= max_keep:
        return candidates

    slab_pos = slab_ase.get_positions()
    slab_sym = slab_ase.get_chemical_symbols()

    def nearest_surface_atom_key(xy):
        dx = slab_pos[:, 0] - xy[0]
        dy = slab_pos[:, 1] - xy[1]
        d2 = dx * dx + dy * dy
        j = int(np.argmin(d2))
        return (slab_sym[j], j)

    buckets: Dict[Tuple[str, int], List[Tuple[int, Atoms, List[float]]]] = {}
    for cid, aa, xyz in candidates:
        key = nearest_surface_atom_key((xyz[0], xyz[1]))
        buckets.setdefault(key, []).append((cid, aa, xyz))

    # deterministic: sort keys, take first entry in each bucket
    selected = []
    for k in sorted(buckets.keys(), key=lambda x: (x[0], x[1])):
        selected.append(buckets[k][0])

    if len(selected) <= max_keep:
        return selected

    # still too many -> deterministic stride sampling
    stride = max(1, len(selected) // max_keep)
    sel2 = selected[::stride][:max_keep]
    return sel2

def mace_pre_rank_downselect(
    cfg: Config,
    candidates: List[Tuple[int, Atoms, List[float]]],
    mace_calc,
    max_keep: int,
    pre_rank_steps: int
) -> Tuple[List[Tuple[int, Atoms, List[float]]], List[Dict[str, Any]]]:
    """
    MACE mode: cheap pre-ranking (single-point or short relax) -> select topK -> return selected candidates
    plus a ranking log list for saving.
    """
    if len(candidates) <= max_keep:
        return candidates, [{"cand": cid, "E_pre": None} for cid, _, _ in candidates]

    scored = []
    ranklog = []
    for cid, aa, xyz in candidates:
        try:
            if pre_rank_steps and pre_rank_steps > 0:
                # short relax with loose fmax
                Epre = mace_short_relax_energy(cfg, aa, mace_calc, steps=pre_rank_steps, fmax=cfg.PRE_RANK_FMAX)
            else:
                Epre = mace_single_point_energy(aa, mace_calc)
            scored.append((float(Epre), cid, aa, xyz))
            ranklog.append({"cand": cid, "E_pre": float(Epre), "site": [float(x) for x in xyz]})
        except Exception as e:
            ranklog.append({"cand": cid, "E_pre": None, "error": str(e), "site": [float(x) for x in xyz]})

    scored.sort(key=lambda x: x[0])
    selected = [(cid, aa, xyz) for _, cid, aa, xyz in scored[:max_keep]]
    return selected, ranklog

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

    # Check: required ads are defined
    for an in cfg.SCREEN_ADS:
        if an not in ads_mols:
            raise ValueError(f"Adsorbate '{an}' not defined in build_adsorbates()")

    # Gas refs
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

    # Slab: load from checkpoint or compute
    slab_xyz = os.path.join(workdir, "%s_slab_relaxed.xyz" % base)
    if slab_ok and "E_slab_raw" in ck and "E_slab_corr" in ck and os.path.exists(slab_xyz) and "fixed_slab_indices" in ck:
        plog("Slab from checkpoint: E_raw = %.6f eV, E_corr = %.6f eV" % (ck["E_slab_raw"], ck["E_slab_corr"]))
        E_slab_raw = float(ck["E_slab_raw"])
        E_slab_corr = float(ck["E_slab_corr"])
        term_id = int(ck["term_id"])
        fixed_slab_indices = list(ck["fixed_slab_indices"])
        slab_ase = ase_read(slab_xyz)
    else:
        Er_raw, slab_ase, fixed_slab_indices, centers, term_id = pick_best_term(cfg, target, adaptor, workdir, base, mace_calc)
        E_slab_raw = float(Er_raw)
        # apply optional correction for slab baseline "*"
        E_slab_corr = float(tierB(cfg, E_slab_raw, "*", is_ads=True))
        write(slab_xyz, slab_ase)
        plog("Slab E_raw = %.6f eV, E_corr = %.6f eV (term %d)" % (E_slab_raw, E_slab_corr, term_id))
        ck.update({
            "slab_done": True,
            "E_slab_raw": E_slab_raw,
            "E_slab_corr": E_slab_corr,
            "term_id": int(term_id),
            "fixed_slab_indices": [int(i) for i in fixed_slab_indices],
            "centers": [float(c) for c in centers],
        })
        save_ckpt(workdir, ck)

    # Build adsorption sites on the relaxed slab (pymatgen structure)
    slab_pm = adaptor.get_structure(slab_ase)
    asf = AdsorbateSiteFinder(slab_pm)
    site_coords = find_ads_sites(asf)
    if not site_coords:
        plog("No adsorption sites found by AdsorbateSiteFinder. Skipping structure.", "WARN")
        ck["fully_done"] = True
        save_ckpt(workdir, ck)
        return

    plog("Total raw adsorption sites (from find_adsorption_sites): %d" % len(site_coords))

    # Screen adsorbates
    for ai, aname in enumerate(cfg.SCREEN_ADS, 1):
        if aname in done_ads:
            plog("[%d/%d] %s -- SKIP (done)" % (ai, len(cfg.SCREEN_ADS), aname))
            continue

        pbar_header("Adsorbate [%d/%d]: %s" % (ai, len(cfg.SCREEN_ADS), aname))
        try:
            height = cfg.ADS_HEIGHT.get(aname, 2.0)
            ads_ase = pmg_mol_to_ase(ads_mols[aname])

            plog("Building adsorption candidates (ASE add_adsorbate) ...", "RUN")
            candidates = build_adsorption_candidates_ase(
                cfg=cfg,
                slab_ase=slab_ase,
                fixed_slab_indices=fixed_slab_indices,
                adsorbate_ase=ads_ase,
                site_coords_cart=site_coords,
                height=height
            )
            if not candidates:
                plog("No candidates constructed for %s" % aname, "WARN")
                done_ads.add(aname)
                ck["completed_adsorbates"] = list(done_ads)
                save_ckpt(workdir, ck)
                continue

            plog("Constructed %d candidates before downselect" % len(candidates))

            # Downselect (NO truncating by order)
            ranklog_path = os.path.join(workdir, "%s_%s_ranked.json" % (base, aname))
            if cfg.MAX_ADS_CANDIDATES and len(candidates) > cfg.MAX_ADS_CANDIDATES:
                if cfg.ENGINE == "mace":
                    if mace_calc is None:
                        mace_calc = mk_mace(cfg)
                    plog("MACE pre-ranking %d -> %d (steps=%d)" % (len(candidates), cfg.MAX_ADS_CANDIDATES, cfg.PRE_RANK_STEPS), "STEP")
                    selected, ranklog = mace_pre_rank_downselect(
                        cfg, candidates, mace_calc,
                        max_keep=cfg.MAX_ADS_CANDIDATES,
                        pre_rank_steps=cfg.PRE_RANK_STEPS
                    )
                    with open(ranklog_path, "w") as f:
                        json.dump(ranklog, f, indent=2)
                    candidates = selected
                else:
                    plog("GPAW geometric bucketing %d -> %d" % (len(candidates), cfg.MAX_ADS_CANDIDATES), "STEP")
                    candidates = gpaw_geometric_bucket_downselect(slab_ase, candidates, cfg.MAX_ADS_CANDIDATES)
                    # save a small log for reproducibility
                    with open(ranklog_path, "w") as f:
                        json.dump(
                            [{"cand": cid, "site": [float(x) for x in xyz]} for cid, _, xyz in candidates],
                            f, indent=2
                        )

            plog("%d candidates after downselect" % len(candidates), "DONE")

            # Relax all selected candidates and pick best by final energy
            cands_relaxed = []
            for ii, (cid, aa, xyz) in enumerate(candidates, 1):
                plog("  Cand %d/%d (site_id=%d) ..." % (ii, len(candidates), cid), "RUN")
                try:
                    gpaw_kpts = compute_kpts_from_kspacing(aa, cfg.KSPACING, slab_xy_only=True)
                    E_raw = do_relax(cfg, aa, workdir, base, aname, i=cid, mace_calc=mace_calc, is_slab=True, gpaw_kpts=gpaw_kpts)
                    E_raw = float(E_raw)

                    # Apply correction to adsorbed system energy (optional)
                    E_corr = float(tierB(cfg, E_raw, ads_key(aname), is_ads=True))

                    xyz_out = os.path.join(workdir, "%s_%s_site%d_relaxed.xyz" % (base, aname, cid))
                    write(xyz_out, aa)

                    cands_relaxed.append({
                        "site_id": int(cid),
                        "E_ads_sys_raw_eV": E_raw,
                        "E_ads_sys_corr_eV": E_corr,
                        "xyz": os.path.basename(xyz_out),
                        "site_cart": [float(x) for x in xyz],
                    })
                    plog("  site%d: E_raw=%.6f eV, E_corr=%.6f eV" % (cid, E_raw, E_corr))
                except Exception as e:
                    plog("  site%d failed: %s" % (cid, e), "WARN")

            if not cands_relaxed:
                plog("All candidates failed for %s" % aname, "WARN")
                done_ads.add(aname)
                ck["completed_adsorbates"] = list(done_ads)
                save_ckpt(workdir, ck)
                continue

            # Pick best by corrected energy (more consistent with tierB); fall back to raw if needed
            cands_relaxed.sort(key=lambda d: d.get("E_ads_sys_corr_eV", d["E_ads_sys_raw_eV"]))
            best = cands_relaxed[0]
            best_site = int(best["site_id"])
            E_ads_sys_raw = float(best["E_ads_sys_raw_eV"])
            E_ads_sys_corr = float(best["E_ads_sys_corr_eV"])

            # Descriptor value based on corrected energies (consistent)
            desc_name, desc_val = compute_descriptor(aname, E_ads_sys_corr, E_slab_corr, gas_E)

            best_f = os.path.join(workdir, "%s_%s_best.xyz" % (base, aname))
            # copy relaxed best xyz
            src_best = os.path.join(workdir, best["xyz"])
            if os.path.exists(src_best):
                shutil.copy2(src_best, best_f)

            plog("* %s BEST: site%d, %s = %.4f eV" % (aname, best_site, desc_name, desc_val), "DONE")

            # Save ranked relaxed candidates for traceability
            relax_rank_path = os.path.join(workdir, "%s_%s_relaxed_rank.json" % (base, aname))
            with open(relax_rank_path, "w") as f:
                json.dump(cands_relaxed, f, indent=2)

            # Store result record (keep both raw and corrected energies)
            results.append({
                "structure": base,
                "miller": str(cfg.TARGET_MILLER),
                "termination_id": int(term_id),
                "adsorbate": aname,
                "best_site_id": int(best_site),
                "engine": cfg.ENGINE,

                "E_slab_raw_eV": float(E_slab_raw),
                "E_slab_corr_eV": float(E_slab_corr),

                "E_ads_sys_raw_eV": float(E_ads_sys_raw),
                "E_ads_sys_corr_eV": float(E_ads_sys_corr),

                "descriptor": desc_name,
                "descriptor_value_eV": float(desc_val),

                "best_xyz": os.path.basename(best_f),
                "ranklog": os.path.basename(ranklog_path),
                "relaxed_rank": os.path.basename(relax_rank_path),
            })

            done_ads.add(aname)
            ck["completed_adsorbates"] = list(done_ads)
            ck["results"] = results
            save_ckpt(workdir, ck)
            plog("Checkpoint saved (%s done)" % ", ".join(sorted(done_ads)))

        except Exception as e:
            plog("%s failed: %s" % (aname, e), "ERROR")

    # CSV output (now with raw+corrected energies)
    csv_p = os.path.join(workdir, "%s_descriptors.csv" % base)
    flds = [
        "structure", "miller", "termination_id", "adsorbate", "best_site_id", "engine",
        "E_slab_raw_eV", "E_slab_corr_eV",
        "E_ads_sys_raw_eV", "E_ads_sys_corr_eV",
        "descriptor", "descriptor_value_eV",
        "best_xyz", "ranklog", "relaxed_rank"
    ]
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
        print("  +----------+-------------------+--------------+", flush=True)
        print("  | Adsorb.  | Descriptor        | Value (eV)   |", flush=True)
        print("  +----------+-------------------+--------------+", flush=True)
        for r in results:
            print("  | %-8s | %-17s | %+12.4f |" % (r["adsorbate"], r["descriptor"], r["descriptor_value_eV"]), flush=True)
        print("  +----------+-------------------+--------------+", flush=True)

    plog("Files in %s:" % workdir)
    for fn in sorted(os.listdir(workdir)):
        fp = os.path.join(workdir, fn)
        if os.path.isdir(fp):
            continue
        sz = os.path.getsize(fp)
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

    logging.basicConfig(
        filename=os.path.join(cfg.OUTDIR, "calculation.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

    if cfg.ENGINE == "mace":
        if not cfg.MACE_MODEL_PATH:
            raise SystemExit("Missing MACE model. Set --mace-model or env MACE_MODEL_PATH.")
        if not os.path.exists(cfg.MACE_MODEL_PATH):
            raise SystemExit("MACE model not found: %s" % cfg.MACE_MODEL_PATH)

    cifs = sorted(glob.glob(os.path.join(cfg.CIF_FOLDER, "*.cif")))
    if not cifs:
        print("[ERR] No .cif files in %s" % cfg.CIF_FOLDER, flush=True)
        sys.exit(1)

    # Global checkpoint with lock
    if cfg.RESUME:
        with SimpleFileLock(cfg.GLOBAL_CKPT_LOCK, timeout=600):
            gck = load_global_ckpt(cfg)
    else:
        gck = {"completed_files": []}

    done_f = set(gck.get("completed_files", []))

    pbar_header("High-throughput adsorption energy calculation")
    plog("Engine       : %s" % cfg.ENGINE)
    plog("CIF folder   : %s" % cfg.CIF_FOLDER)
    plog("Output       : %s" % cfg.OUTDIR)
    plog("Miller       : %s" % str(cfg.TARGET_MILLER))
    plog("Supercell    : %s" % str(cfg.SLAB_SUPERCELL))
    plog("kspacing     : %.3f 1/Å" % cfg.KSPACING)
    plog("MaxCand/ads  : %d" % cfg.MAX_ADS_CANDIDATES)
    plog("Resume       : %s" % cfg.RESUME)
    plog("Adsorbates   : %s" % ", ".join(cfg.SCREEN_ADS))
    plog("Total CIFs   : %d" % len(cifs))
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
            gck["completed_files"] = sorted(list(done_f))
            # Save global ckpt with lock to avoid corruption if multiple jobs share outdir
            with SimpleFileLock(cfg.GLOBAL_CKPT_LOCK, timeout=600):
                save_global_ckpt(cfg, gck)
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
