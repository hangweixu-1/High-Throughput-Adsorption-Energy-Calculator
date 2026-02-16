# High-Throughput Adsorption Energy Calculator

Automated workflow for screening alloy catalysts: download crystal structures from Materials Project, generate surface slabs, and compute adsorption(-descriptor) energies using **MACE (machine-learning potential)** or **GPAW (DFT)**.

This repository is designed for **high-throughput, reproducible** adsorption screening on surfaces, with **robust candidate selection** (no "truncate first N sites") and **stable constraint handling** (no heuristic adsorbate detection by `z`).

---

## Overview

The project consists of two scripts:

1. **`download_cif.py`** — Query the Materials Project database and batch-download CIF files with flexible composition / stability / band-gap filters.
2. **`High_throughput.py`** — Read CIF files, build surface slabs, enumerate adsorption sites, place adsorbates (CO₂, CO, H, COOH, CHO, **OCHO** …), relax with MACE or GPAW, and output adsorption descriptors.

---

## Key Features (What this workflow guarantees)

### Candidate selection is NOT order-dependent

* **No "take the first N sites" truncation**.
* If there are too many sites:

  * **MACE engine** performs **cheap pre-ranking** (single-point or short relax) and keeps the top-K before full relaxation.
  * **GPAW engine** performs **geometric bucketing/downselect** (deterministic) before full relaxation.

### Robust constraints (no adsorbate misclassification)

* Adsorbates are placed using **ASE `add_adsorbate`** so that:

  * slab atoms are always `0 : n_slab`
  * adsorbate atoms are always appended at the end
* **Fixed layers** are applied only to the **slab indices**, avoiding bugs when:

  * surfaces rumple/reconstruct
  * adsorbate starts low or migrates into the surface
  * high-index / stepped surfaces exist

### Reproducible and restart-safe

* Gas reference cache and global checkpoints are protected by **simple file locks** to reduce corruption when multiple jobs share an output directory.
* Supports `--resume` at both **global** and **per-structure** levels.

### CO₂RR-ready adsorbates

* Default adsorbates include **OCHO** (formate/formic acid key intermediate).

---

## Prerequisites

* Python 3.10+
* A Materials Project API key (free): [https://next-gen.materialsproject.org/api](https://next-gen.materialsproject.org/api)

Set the key as an environment variable so both scripts can use it:

```bash
export MP_API_KEY="your_key_here"
```

---

## 1. Downloading CIF Files (`download_cif.py`)

### Basic Usage

```bash
python download_cif.py --api-key YOUR_KEY --elements Cu
```

This downloads **all materials containing Cu** into the `./cif` folder. For catalyst screening you usually want tighter filters — see the examples below.

### Command-Line Options

| Flag                      | Default          | Description                                               |
| ------------------------- | ---------------- | --------------------------------------------------------- |
| `--api-key`               | env `MP_API_KEY` | Materials Project API key                                 |
| `--elements`              | `Cu`             | Elements that **must** be present (comma-separated)       |
| `--chemsys`               | —                | Chemical system, e.g. `Cu-Zn-O`. Overrides `--elements`  |
| `--exclude-elements`      | —                | Elements that must **not** be present (comma-separated)   |
| `--is-stable`             | off              | Only thermodynamically stable phases (on the convex hull) |
| `--energy-above-hull-max` | —                | Max energy above hull in eV/atom (e.g. `0.05`)           |
| `--band-gap-min`          | —                | Minimum band gap (eV)                                    |
| `--band-gap-max`          | —                | Maximum band gap (eV)                                    |
| `--num-elements-max`      | —                | Maximum number of element types                           |
| `--outdir`                | `cif`            | Output folder                                             |
| `--limit`                 | 0 (no limit)     | Max number of structures to download                      |
| `--conventional`          | off              | Export as conventional standard cell                      |
| `--skip-existing`         | off              | Skip files already on disk                                |

### Copper-Based Catalyst Examples

**All stable Cu-containing structures (broad screen):**

```bash
python download_cif.py --elements Cu --is-stable --outdir cif_cu_stable
```

**Copper oxides — only synthesizable metastable phases:**

```bash
python download_cif.py --elements Cu,O --energy-above-hull-max 0.05 \
    --exclude-elements Hg,Cd,Pb,Tl --outdir cif_cuo
```

**Cu-Zn-O ternary system (methanol catalyst family):**

```bash
python download_cif.py --chemsys Cu-Zn-O --energy-above-hull-max 0.1 \
    --outdir cif_cuzno
```

**Quick test — just 5 structures:**

```bash
python download_cif.py --elements Cu,O --is-stable --limit 5 \
    --conventional --outdir cif_test
```

---

## 2. Computing Adsorption Energies (`High_throughput.py`)

### Quick Start (MACE)

You must provide the MACE model path either by environment variable or CLI.

**Option A: environment variable (recommended)**

```bash
export MACE_MODEL_PATH=/path/to/your.model
python High_throughput.py ./cif_cu_binary mace
```

**Option B: CLI**

```bash
python High_throughput.py ./cif_cu_binary mace --mace-model /path/to/your.model
```

**GPU**

```bash
python High_throughput.py ./cif_cu_binary mace \
  --mace-model /path/to/your.model \
  --mace-device cuda
```

### Quick Start (GPAW)

```bash
python High_throughput.py ./cif_cu_binary gpaw
```

### Resume after interruption

```bash
python High_throughput.py ./cif_cu_binary mace --resume
```

### Custom Miller index and supercell

```bash
python High_throughput.py ./cif_cu_binary mace \
  --mace-model /path/to/your.model \
  --miller 1,1,1 \
  --supercell 3,3,1
```

---

## Command-Line Options (High_throughput.py)

| Flag               | Default                   | Description                                                                   |
| ------------------ | ------------------------- | ----------------------------------------------------------------------------- |
| `cif_folder`       | (required)                | Folder containing `.cif` files                                                |
| `engine`           | (required)                | `mace` or `gpaw`                                                              |
| `-o, --output`     | `ht_results`              | Output root directory                                                         |
| `-r, --resume`     | off                       | Resume from last checkpoint                                                   |
| `--miller`         | `1,1,0`                   | Miller index (comma-separated)                                                |
| `--supercell`      | `2,2,1`                   | Supercell size (comma-separated)                                              |
| `--adsorbates`     | `CO2,CO,H,COOH,CHO,OCHO` | Adsorbates to screen (comma-separated)                                        |
| `--max-candidates` | `12`                      | Max candidates kept per adsorbate after downselect                            |
| `--kspacing`       | `0.22`                    | Target k-point spacing (1/Å) for slabs (GPAW uses this to auto-generate kpts) |
| `--mace-model`     | env `MACE_MODEL_PATH`     | Path to MACE `.model` file                                                    |
| `--mace-device`    | `cpu`                     | `cpu` or `cuda`                                                               |
| `--mace-dtype`     | `float64`                 | `float32` or `float64`                                                        |
| `--fmax-mace`      | `0.10`                    | MACE relax force threshold (eV/Å)                                             |
| `--fmax-gpaw`      | `0.05`                    | GPAW relax force threshold (eV/Å)                                             |
| `--max-steps`      | `200`                     | Max BFGS optimization steps                                                   |
| `--pre-rank-steps` | `0`                       | **MACE only**: pre-ranking by single-point (`0`) or short relax (`>0`)        |
| `--pre-rank-fmax`  | `0.30`                    | **MACE only**: loose fmax for short pre-rank relax                            |

---

## Workflow

For each CIF file the script:

1. Parses the crystal structure
2. Generates surface slabs for the target Miller index and enumerates terminations
3. Relaxes all terminations and selects the lowest-energy slab
4. Finds adsorption sites using `AdsorbateSiteFinder.find_adsorption_sites()`
5. Builds adsorption candidates by placing adsorbates using **ASE `add_adsorbate`** (adsorbate atoms appended at the end)
6. If too many candidates:

   * **MACE**: cheap pre-ranking → keep top-K
   * **GPAW**: geometric bucketing/downselect → keep K representatives
7. Relaxes each selected candidate, ranks by final energy, and writes best structure
8. Computes adsorption-descriptor values relative to gas-phase references
9. Writes per-structure CSV + ranked JSON logs + XYZ files

Gas-phase references (H₂, CO, CO₂, H₂O) are computed once and cached in `gas_cache.json`.

---

## Adsorbates and Descriptors

> Note: These are **descriptor proxies** based on total energies + optional constant corrections.
> They are consistent for ranking but not full electrochemical free energies ΔG(U).

| Adsorbate | Descriptor       | Formula                                         |
| --------- | ---------------- | ----------------------------------------------- |
| H         | dE\*H            | E(ads\_sys) − E(slab) − 0.5 × E(H₂)           |
| CO        | dE\*CO           | E(ads\_sys) − E(slab) − E(CO)                  |
| CO₂       | dE\*CO₂          | E(ads\_sys) − E(slab) − E(CO₂)                 |
| COOH      | dE\*COOH (proxy) | E(ads\_sys) − E(slab) − E(CO₂) − 0.5 × E(H₂) |
| CHO       | dE\*CHO (proxy)  | E(ads\_sys) − E(slab) − E(CO) − 0.5 × E(H₂)  |
| OCHO      | dE\*OCHO (proxy) | E(ads\_sys) − E(slab) − E(CO₂) − 0.5 × E(H₂) |

---

## Output Structure

```
ht_results/
  global_checkpoint.json                # tracks completed CIF files (locked writes)
  gas_cache.json                        # cached gas-phase reference energies (locked writes)
  calculation.log                       # full log
  Cu3Pd/
    Cu3Pd.cif                           # copy of input
    checkpoint.json                     # per-structure progress
    Cu3Pd_slab_relaxed.xyz              # relaxed clean slab
    Cu3Pd_CO2_ranked.json               # pre-rank / downselect log
    Cu3Pd_CO2_relaxed_rank.json         # full relax ranking for selected candidates
    Cu3Pd_CO2_site12_relaxed.xyz        # relaxed candidate structures (site-based naming)
    Cu3Pd_CO2_best.xyz                  # best candidate
    Cu3Pd_descriptors.csv               # summary table (raw + corrected energies + descriptor)
  CuZn/
    ...
```

### What to inspect first

* `*_descriptors.csv` — summary of best site per adsorbate with:

  * `E_slab_raw/corr`
  * `E_ads_sys_raw/corr`
  * descriptor value
* `*_relaxed_rank.json` — check near-degenerate sites / confirm ranking stability
* `*_ranked.json` — confirm pre-ranking/downselect behavior (especially for MACE)

---

## Checkpoint / Resume

Progress is saved at two levels:

* **Global** (`global_checkpoint.json`) — which CIF files are fully done; skipped on resume.
* **Per-structure** (`checkpoint.json`) — slab relaxation status + completed adsorbates.

```bash
# Resume
python High_throughput.py ./cif_cu_binary mace --resume

# Start fresh (ignores all checkpoints)
python High_throughput.py ./cif_cu_binary mace
```

---

## MACE Configuration

### Provide model path (required)

**Env:**

```bash
export MACE_MODEL_PATH=/path/to/your.model
python High_throughput.py ./cifs mace
```

**CLI:**

```bash
python High_throughput.py ./cifs mace --mace-model /path/to/your.model
```

### Cheap pre-ranking (recommended for large site counts)

Single-point pre-ranking (fast):

```bash
python High_throughput.py ./cifs mace \
  --mace-model /path/to/your.model \
  --max-candidates 12 \
  --pre-rank-steps 0
```

Short-relax pre-ranking (more robust, slower):

```bash
python High_throughput.py ./cifs mace \
  --mace-model /path/to/your.model \
  --max-candidates 12 \
  --pre-rank-steps 10 \
  --pre-rank-fmax 0.30
```

---

## GPAW Configuration

Default settings:

* Plane-wave cutoff: 470 eV
* Exchange-correlation: PBE
* Molecule k-points: 1 × 1 × 1
* Slab k-points: **auto-generated from `--kspacing`** (default 0.22 1/Å)
* Fermi–Dirac smearing: 0.10 eV
* Dipole correction: xy

Example:

```bash
python High_throughput.py ./cifs gpaw --kspacing 0.22
```

Higher accuracy (slower):

```bash
python High_throughput.py ./cifs gpaw --kspacing 0.18
```

---

## End-to-End Example

```bash
# 1) Set API key
export MP_API_KEY="your_key"

# 2) Download stable Cu-Pd alloys
python download_cif.py --chemsys Cu-Pd --is-stable --conventional --outdir cifs_cupd

# 3) Screen with MACE (fast, includes OCHO by default)
export MACE_MODEL_PATH=/path/to/your.model
python High_throughput.py cifs_cupd mace -o results_cupd

# 4) Inspect results
cat results_cupd/Cu3Pd/Cu3Pd_descriptors.csv

# 5) Resume if interrupted
python High_throughput.py cifs_cupd mace -o results_cupd --resume
```

---

## Installation

```bash
conda create -n adsorption python=3.10
conda activate adsorption
pip install -r requirements.txt
```

For GPAW:

```bash
conda install -c conda-forge gpaw
# or
pip install gpaw && gpaw install-data
```

For MACE with GPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install mace-torch
```

---

## License

For internal / research use.
