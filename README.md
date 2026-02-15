
---

```markdown
# High-Throughput Adsorption Energy Calculator

Automated workflow for screening alloy catalysts: download crystal structures from Materials Project, generate surface slabs, and compute adsorption (and descriptor) energies using **MACE (machine-learning potential)** or **GPAW (DFT)**.

This repository is designed for **high-throughput, reproducible** adsorption screening on surfaces, with **robust candidate selection** (no "truncate first N sites") and **stable constraint handling** (no heuristic adsorbate detection by `z`).

---

## Overview

The project consists of two core scripts:

1. **`download_cif.py`** — Query the Materials Project database and batch-download CIF files with flexible composition, stability, and band-gap filters.
2. **`High_throughput.py`** — Read CIF files, build surface slabs, enumerate adsorption sites, place adsorbates (CO₂, CO, H, COOH, CHO, **OCHO**...), relax with MACE or GPAW, and output adsorption descriptors.

---

## Key Features

### 1. Robust Candidate Selection
* **Order-Independent:** No "take the first N sites" truncation.
* **Smart Downselection:**
    * **MACE engine:** Performs **cheap pre-ranking** (single-point or short relax) and keeps the top-K candidates before full relaxation.
    * **GPAW engine:** Performs **geometric bucketing** (deterministic spatial clustering) to select representatives before relaxation.

### 2. Stable Constraints
* **Explicit Indexing:** Adsorbates are placed using `ASE add_adsorbate` such that:
    * Slab atoms are always indices `0 : n_slab`.
    * Adsorbate atoms are always appended at the end.
* **Fixed Layers:** Constraints are applied strictly to bottom slab indices, avoiding bugs when surfaces reconstruct or adsorbates migrate into the surface.

### 3. Reproducible & Restart-Safe
* **File Locks:** Gas reference cache and global checkpoints are protected by file locks (`fcntl`/atomic files) to prevent corruption when multiple jobs share an output directory.
* **Resume Capability:** Supports `--resume` at both **global** (skip finished files) and **per-structure** (resume specific adsorbate) levels.

### 4. CO₂RR-Ready
* Default adsorbates include **OCHO** (formate), a key intermediate in CO₂ reduction, utilizing a robust initial geometry guess.

---

## Prerequisites

* **Python 3.10+**
* **Materials Project API Key**: Get one for free at [Materials Project API](https://next-gen.materialsproject.org/api).

Set the key as an environment variable:

```bash
export MP_API_KEY="your_key_here"

```

---

## 1. Downloading CIF Files (`download_cif.py`)

### Basic Usage

```bash
python download_cif.py --api-key YOUR_KEY --elements Cu

```

*Downloads all materials containing Cu into the `./cif` folder.*

### Command-Line Options

| Flag | Default | Description |
| --- | --- | --- |
| `--api-key` | env `MP_API_KEY` | Materials Project API key. |
| `--elements` | `Cu` | Elements that **must** be present (comma-separated). |
| `--chemsys` | — | Chemical system, e.g. `Cu-Zn-O`. Overrides `--elements`. |
| `--exclude-elements` | — | Elements that must **not** be present (e.g. `Hg,Cd`). |
| `--is-stable` | off | Only stable phases (on the convex hull). |
| `--energy-above-hull-max` | — | Max energy above hull in eV/atom (e.g. `0.05`). |
| `--band-gap-min` | — | Minimum band gap (eV). |
| `--band-gap-max` | — | Maximum band gap (eV). |
| `--num-elements-max` | — | Maximum number of element types (e.g. `3` for ternaries). |
| `--outdir` | `cif` | Output folder. |
| `--limit` | 0 | Max number of structures to download (0 = no limit). |
| `--conventional` | off | Export as conventional standard cell. |
| `--skip-existing` | off | Skip downloading if file already exists. |

### Examples

**Copper oxides — only synthesizable metastable phases:**

```bash
python download_cif.py --elements Cu,O --energy-above-hull-max 0.05 \
    --exclude-elements Hg,Cd,Pb,Tl --outdir cif_cuo

```

**Cu-Zn-O ternary system (Methanol synthesis family):**

```bash
python download_cif.py --chemsys Cu-Zn-O --energy-above-hull-max 0.1 \
    --outdir cif_cuzno

```

---

## 2. Computing Adsorption Energies (`High_throughput.py`)

### Quick Start (MACE)

**Option A: Environment variable (Recommended)**

```bash
export MACE_MODEL_PATH=/path/to/your.model
python High_throughput.py ./cif_cu_binary mace

```

**Option B: CLI Argument**

```bash
python High_throughput.py ./cif_cu_binary mace --mace-model /path/to/your.model

```

**Using GPU**

```bash
python High_throughput.py ./cif_cu_binary mace \
  --mace-model /path/to/your.model \
  --mace-device cuda

```

### Quick Start (GPAW)

```bash
python High_throughput.py ./cif_cu_binary gpaw --kspacing 0.22

```

### Resume after interruption

```bash
python High_throughput.py ./cif_cu_binary mace --resume

```

### Command-Line Options

| Flag | Default | Description |
| --- | --- | --- |
| `cif_folder` | *(required)* | Folder containing `.cif` files. |
| `engine` | *(required)* | `mace` or `gpaw`. |
| `-o`, `--output` | `ht_results` | Output root directory. |
| `-r`, `--resume` | off | Resume from last checkpoint. |
| `--miller` | `1,1,0` | Miller index (comma-separated). |
| `--supercell` | `2,2,1` | Supercell size (comma-separated). |
| `--adsorbates` | `CO2,CO,H...` | Adsorbates to screen (see defaults). |
| `--max-candidates` | `12` | Max candidates kept per adsorbate after downselect. |
| `--kspacing` | `0.22` | Target k-point spacing (1/Å). GPAW auto-generates k-points. |
| `--mace-model` | env `MACE...` | Path to MACE `.model` file. |
| `--mace-device` | `cpu` | `cpu` or `cuda`. |
| `--mace-dtype` | `float64` | `float32` or `float64`. |
| `--fmax-mace` | `0.10` | MACE relax force threshold (eV/Å). |
| `--fmax-gpaw` | `0.05` | GPAW relax force threshold (eV/Å). |
| `--max-steps` | `200` | Max BFGS optimization steps. |
| `--pre-rank-steps` | `0` | **MACE**: `0`=single-point, `>0`=short relax pre-ranking. |
| `--pre-rank-fmax` | `0.30` | **MACE**: loose fmax for short pre-rank relax. |

---

## Workflow Steps

For each CIF file, the script performs the following:

1. **Parse** the crystal structure.
2. **Generate Slabs** for the target Miller index and enumerate unique terminations.
3. **Relax Slabs** (all terminations) and select the lowest energy surface.
4. **Find Sites** using `AdsorbateSiteFinder`.
5. **Build Candidates** by placing adsorbates using ASE.
6. **Downselect**:
* *MACE:* Pre-rank (single-point or short relax) -> Keep top K.
* *GPAW:* Geometric bucketing -> Keep K representatives.


7. **Full Relaxation** of selected candidates.
8. **Compute Descriptors** relative to gas-phase references.
9. **Output** CSV summary, JSON logs, and XYZ structures.

---

## Adsorbates and Descriptors

> **Note:** These are descriptor proxies based on total energies + optional constant corrections. They are consistent for ranking but are not full electrochemical free energies .

| Adsorbate | Descriptor Key | Formula (Approx) |
| --- | --- | --- |
| H | `dE_*H` |  |
| CO | `dE_*CO` |  |
| CO₂ | `dE_*CO2` |  |
| COOH | `dE_*COOH` |  |
| CHO | `dE_*CHO` |  |
| OCHO | `dE_*OCHO` |  |

---

## Output Structure

```text
ht_results/
  global_checkpoint.json        # Tracks completed CIF files (locked)
  gas_cache.json                # Cached gas-phase reference energies
  calculation.log               # Main log file
  
  Cu3Pd/                        # Per-structure folder
    Cu3Pd.cif                   # Copy of input
    checkpoint.json             # Per-structure progress
    Cu3Pd_slab_relaxed.xyz      # Relaxed clean slab
    Cu3Pd_CO2_ranked.json       # Pre-rank / downselect log
    Cu3Pd_CO2_relaxed_rank.json # Full relax ranking
    Cu3Pd_CO2_best.xyz          # Best candidate geometry
    Cu3Pd_descriptors.csv       # Final summary table

```

**What to inspect first:**

* **`*_descriptors.csv`**: Summary of the best site for each adsorbate.
* **`*_best.xyz`**: The geometry of the lowest energy configuration.

---

## Installation

### 1. Create Environment

```bash
conda create -n adsorption python=3.10
conda activate adsorption

```

### 2. Install PyTorch

*Select the command matching your CUDA version (see [pytorch.org](https://pytorch.org)):*

```bash
# Example for CUDA 12.1
pip install torch --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

```

### 3. Install MACE and Dependencies

```bash
pip install mace-torch
pip install -r requirements.txt

```

### 4. (Optional) Install GPAW

If you plan to use the DFT engine:

```bash
conda install -c conda-forge gpaw
# OR via pip:
# pip install gpaw && gpaw install-data

```

---

## License

For internal / research use.

```

```
