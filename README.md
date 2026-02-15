# High-Throughput Adsorption Energy Calculator

Automated workflow for screening copper-based (and other) alloy catalysts: download crystal structures from Materials Project, generate surface slabs, and compute adsorption energies using MACE (machine learning potential) or GPAW (DFT).

## Overview

The project consists of two scripts:

1. **`download_cif.py`** — Query the Materials Project database and batch-download CIF files with flexible composition / stability / band-gap filters.
2. **`High_throughput.py`** — Read CIF files, build surface slabs, place adsorbates (CO₂, CO, H, COOH, CHO …), relax with MACE or GPAW, and output adsorption-energy descriptors.

## Prerequisites

- Python 3.10+
- A Materials Project API key (free): <https://next-gen.materialsproject.org/api>

Set the key as an environment variable so both scripts can use it:

```bash
export MP_API_KEY="your_key_here"
```

## 1. Downloading CIF Files (`download_cif.py`)

### Basic Usage

```bash
python download_cif.py --api-key YOUR_KEY --elements Cu
```

This downloads **all materials containing Cu** into the `./cif` folder. For catalyst screening you usually want tighter filters — see the examples below.

### Command-Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `--api-key` | env `MP_API_KEY` | Materials Project API key |
| `--elements` | `Cu` | Elements that **must** be present (comma-separated) |
| `--chemsys` | — | Chemical system, e.g. `Cu-Zn-O`. Overrides `--elements` |
| `--exclude-elements` | — | Elements that must **not** be present (comma-separated) |
| `--is-stable` | off | Only thermodynamically stable phases (on the convex hull) |
| `--energy-above-hull-max` | — | Max energy above hull in eV/atom (e.g. `0.05`) |
| `--band-gap-min` | — | Minimum band gap (eV) |
| `--band-gap-max` | — | Maximum band gap (eV) |
| `--num-elements-max` | — | Maximum number of element types |
| `--outdir` | `cif` | Output folder |
| `--limit` | 0 (no limit) | Max number of structures to download |
| `--conventional` | off | Export as conventional standard cell |
| `--skip-existing` | off | Skip files already on disk |

### Copper-Based Catalyst Examples

**All stable Cu-containing structures (broad screen):**

```bash
python download_cif.py --elements Cu --is-stable --outdir cif_cu_stable
```

**Copper oxides — only synthesisable metastable phases:**

```bash
python download_cif.py --elements Cu,O --energy-above-hull-max 0.05 \
    --exclude-elements Hg,Cd,Pb,Tl --outdir cif_cuo
```

**Cu-Zn-O ternary system (methanol synthesis catalyst family):**

```bash
python download_cif.py --chemsys Cu-Zn-O --energy-above-hull-max 0.1 \
    --outdir cif_cuzno
```

**Cu-N compounds (electrocatalysis, nitrogen reduction):**

```bash
python download_cif.py --elements Cu,N --is-stable \
    --exclude-elements Hg,Cd,Pb --outdir cif_cun
```

**Cu-based bimetallics for CO₂ reduction (binary only):**

```bash
python download_cif.py --elements Cu --num-elements-max 2 \
    --energy-above-hull-max 0.05 --exclude-elements Hg,Cd,Pb,Tl,As \
    --outdir cif_cu_binary
```

**Cu-Pd alloys (complete chemical system):**

```bash
python download_cif.py --chemsys Cu-Pd --is-stable --outdir cif_cupd
```

**Semiconducting Cu compounds (photo-catalysis, band gap 1–3 eV):**

```bash
python download_cif.py --elements Cu --band-gap-min 1.0 --band-gap-max 3.0 \
    --energy-above-hull-max 0.05 --outdir cif_cu_semi
```

**Quick test — just 5 structures:**

```bash
python download_cif.py --elements Cu,O --is-stable --limit 5 \
    --conventional --outdir cif_test
```

## 2. Computing Adsorption Energies (`job.py`)

### Quick Start

```bash
# MACE (fast, ML potential)
python job.py ./cif_cu_binary mace

# GPAW (DFT, more accurate, much slower)
python job.py ./cif_cu_binary gpaw

# Resume after interruption
python job.py ./cif_cu_binary mace --resume

# Custom Miller index and supercell
python job.py ./cif_cu_binary mace --miller 1,1,1 --supercell 3,3,1
```

### Command-Line Options

| Flag | Default | Description |
|------|---------|-------------|
| `cif_folder` | (required) | Folder containing `.cif` files |
| `engine` | (required) | `mace` or `gpaw` |
| `-o, --output` | `ht_results` | Output root directory |
| `-r, --resume` | off | Resume from last checkpoint |
| `--miller` | `1,1,0` | Miller index (comma-separated) |
| `--supercell` | `2,2,1` | Supercell size (comma-separated) |
| `--mace-model` | (built-in path) | Path to MACE `.model` file |
| `--mace-device` | `cpu` | `cpu` or `cuda` |
| `--fmax-mace` | `0.10` | Force convergence for MACE (eV/Å) |
| `--fmax-gpaw` | `0.05` | Force convergence for GPAW (eV/Å) |
| `--max-steps` | `200` | Max BFGS optimisation steps |
| `--max-candidates` | `12` | Max adsorption site candidates per adsorbate |

### Workflow

For each CIF file the script:

1. Parses the crystal structure
2. Generates surface slabs for the target Miller index
3. Relaxes all terminations and selects the lowest-energy one
4. Places adsorbates (CO₂, CO, H, COOH, CHO) via pymatgen's `AdsorbateSiteFinder`
5. Relaxes every adsorption candidate and ranks by energy
6. Computes adsorption-energy descriptors relative to gas-phase references
7. Writes per-structure CSV and XYZ files

### Adsorbates and Descriptors

| Adsorbate | Descriptor | Formula |
|-----------|------------|---------|
| H | ΔG\*H | E(ads) − E(slab) − 0.5 × E(H₂) |
| CO | ΔG\*CO | E(ads) − E(slab) − E(CO) |
| CO₂ | ΔG\*CO₂ | E(ads) − E(slab) − E(CO₂) |
| COOH | ΔG\*COOH | E(ads) − E(slab) − E(CO₂) − 0.5 × E(H₂) |
| CHO | ΔG\*CHO (proxy) | E(ads) − E(slab) − E(CO) − 0.5 × E(H₂) |

Gas-phase references (H₂, CO, CO₂, H₂O) are computed once and cached in `gas_cache.json`.

### Output Structure

```
ht_results/
  global_checkpoint.json          # tracks completed CIF files
  gas_cache.json                  # cached gas-phase reference energies
  calculation.log                 # full log
  Cu3Pd/
    Cu3Pd.cif                     # copy of input
    checkpoint.json               # per-structure progress
    Cu3Pd_slab_relaxed.xyz        # relaxed clean slab
    Cu3Pd_CO2_cand0_relaxed.xyz   # all relaxed adsorption candidates
    Cu3Pd_CO2_BEST_cand0.xyz      # best candidate (clearly labelled)
    Cu3Pd_CO_BEST_cand2.xyz
    Cu3Pd_H_BEST_cand1.xyz
    Cu3Pd_COOH_BEST_cand0.xyz
    Cu3Pd_CHO_BEST_cand3.xyz
    Cu3Pd_descriptors.csv         # summary table
  CuZn/
    ...
```

### Checkpoint / Resume

Progress is saved at two levels:

- **Global** (`global_checkpoint.json`) — which CIF files are fully done; skipped on resume.
- **Per-structure** (`checkpoint.json`) — slab relaxation status and completed adsorbates; only remaining work is redone.

```bash
# Resume
python job.py ./cif_cu_binary mace --resume

# Start fresh (ignores all checkpoints)
python job.py ./cif_cu_binary mace
```

### MACE Configuration

The default model path is hard-coded for a specific cluster. Override it:

```bash
python job.py ./cifs mace --mace-model /path/to/your/model.model
```

GPU acceleration:

```bash
python job.py ./cifs mace --mace-device cuda
```

### GPAW Configuration

Default settings when using the GPAW engine:

- Plane-wave cutoff: 470 eV
- Exchange-correlation: PBE
- k-points (slab): 3 × 3 × 1; (molecule): 1 × 1 × 1
- Fermi–Dirac smearing: 0.10 eV
- Dipole correction: xy

## End-to-End Example

```bash
# 1. Set API key
export MP_API_KEY="your_key"

# 2. Download stable Cu-Pd alloys
python download_cif.py --chemsys Cu-Pd --is-stable --conventional --outdir cifs_cupd

# 3. Screen with MACE (fast)
python job.py cifs_cupd mace -o results_cupd

# 4. Inspect results
cat results_cupd/Cu3Pd/Cu3Pd_descriptors.csv

# 5. If interrupted, resume
python job.py cifs_cupd mace -o results_cupd --resume
```

## Installation

```bash
conda create -n adsorption python=3.10
conda activate adsorption
pip install -r requirements.txt
```

For GPAW (requires C extensions):

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

## License

For internal / research use.
