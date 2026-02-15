# High-Throughput Adsorption Energy Calculator

Automated workflow for computing adsorption energies on alloy surfaces using MACE (machine learning potential) or GPAW (DFT). Supports batch processing of CIF files with checkpoint/resume capability.

## Features

- **Two engines**: MACE (fast ML potential) or GPAW (DFT)
- **Command-line interface**: specify input folder, engine, and options
- **Checkpoint/resume**: safely interrupt and continue long calculations
- **Organized output**: one folder per CIF with all relaxed structures, trajectories, and summary CSV
- **Real-time progress**: every step prints what is being computed, with timing

## Quick Start

```bash
# Basic usage
python job.py ./cifs mace

# With GPAW
python job.py ./cifs gpaw

# Resume interrupted run
python job.py ./cifs mace --resume

# Custom output directory
python job.py ./cifs mace -o my_results

# Custom Miller index and supercell
python job.py ./cifs mace --miller 1,1,1 --supercell 3,3,1
```

## Command-Line Options

```
positional arguments:
  cif_folder             Path to folder containing .cif files
  engine                 Calculation engine: mace or gpaw

optional arguments:
  -o, --output DIR       Output root directory (default: ht_results)
  -r, --resume           Resume from last checkpoint
  --miller M             Miller index, comma-separated (default: 1,1,0)
  --supercell S          Supercell size, comma-separated (default: 2,2,1)
  --mace-model PATH      Path to MACE model file
  --mace-device DEV      cpu or cuda (default: cpu)
  --fmax-mace F          Force convergence for MACE in eV/A (default: 0.10)
  --fmax-gpaw F          Force convergence for GPAW in eV/A (default: 0.05)
  --max-steps N          Max BFGS optimization steps (default: 200)
  --max-candidates N     Max adsorption site candidates per adsorbate (default: 12)
```

## Output Structure

```
ht_results/
  global_checkpoint.json      # tracks which CIF files are fully completed
  gas_cache.json              # cached gas molecule reference energies
  calculation.log             # full log
  CuPd/
    CuPd.cif                  # copy of input file
    checkpoint.json            # per-structure progress (slab + each adsorbate)
    CuPd_slab_relaxed.xyz      # relaxed clean slab
    CuPd_slab_term0.traj       # slab optimization trajectory
    CuPd_CO2_cand0_relaxed.xyz # all relaxed adsorption candidates
    CuPd_CO2_cand1_relaxed.xyz
    CuPd_CO2_cand2_relaxed.xyz
    CuPd_CO2_BEST_cand0.xyz    # best candidate clearly labeled
    CuPd_CO_BEST_cand2.xyz
    CuPd_H_BEST_cand1.xyz
    CuPd_COOH_BEST_cand0.xyz
    CuPd_CHO_BEST_cand3.xyz
    CuPd_descriptors.csv       # summary table with all adsorption energies
  NiCu/
    ...
```

## Checkpoint / Resume

The script saves progress at two levels:

1. **Global checkpoint** (`global_checkpoint.json`): records which CIF files have been fully processed. On resume, completed files are skipped entirely.

2. **Per-structure checkpoint** (`<name>/checkpoint.json`): records whether the slab has been relaxed and which adsorbates have been computed. On resume, the slab relaxation is skipped if already done, and only remaining adsorbates are calculated.

To resume after an interruption:

```bash
python job.py ./cifs mace --resume
```

To start fresh (ignore previous checkpoints):

```bash
python job.py ./cifs mace
```

## Adsorbates and Descriptors

The following adsorbates are screened by default:

| Adsorbate | Descriptor          | Reference                        |
|-----------|---------------------|----------------------------------|
| H         | dG_*H               | E_ads - E_slab - 0.5*E(H2)      |
| CO        | dG_*CO              | E_ads - E_slab - E(CO)           |
| CO2       | dG_*CO2             | E_ads - E_slab - E(CO2)          |
| COOH      | dG_*COOH            | E_ads - E_slab - E(CO2) - 0.5*E(H2) |
| CHO       | dG_*CHO_proxy       | E_ads - E_slab - E(CO) - 0.5*E(H2)  |

Gas-phase reference molecules (H2, CO, CO2, H2O) are computed once and cached in `gas_cache.json`.

## Workflow

For each CIF file the script performs:

1. **Parse crystal structure** from CIF
2. **Generate surface slabs** for the target Miller index
3. **Select best termination** by relaxing all candidates and picking the lowest energy
4. **Generate adsorption sites** using pymatgen's AdsorbateSiteFinder
5. **Relax all adsorption candidates** and rank by energy
6. **Compute adsorption energy descriptors** relative to gas-phase references
7. **Save results** as CSV and XYZ files

## MACE Model

The default MACE model path is set to:

```
/public/home/lib/tc/xifuneng/2023-12-03-mace-128-L1_epoch-199.model
```

Change it with `--mace-model`:

```bash
python job.py ./cifs mace --mace-model /path/to/your/model.model
```

For GPU acceleration:

```bash
python job.py ./cifs mace --mace-device cuda
```

## GPAW Settings

When using GPAW engine, the following defaults are used:

- Plane-wave cutoff: 470 eV
- Exchange-correlation: PBE
- k-points (slab): 3x3x1
- k-points (molecule): 1x1x1
- Fermi-Dirac smearing: 0.10 eV
- Dipole correction: xy layer

## Installation

See `requirements.txt` for Python dependencies. A conda environment is recommended:

```bash
conda create -n adsorption python=3.10
conda activate adsorption
pip install -r requirements.txt
```

For GPAW, follow the official GPAW installation guide as it requires C extensions:

```bash
# Option 1: conda
conda install -c conda-forge gpaw

# Option 2: pip (needs libxc, BLAS, etc.)
pip install gpaw
gpaw install-data
```

For MACE:

```bash
pip install mace-torch
```

## Example

```bash
# Prepare input
mkdir cifs
cp CuPd.cif NiCu.cif FePd.cif cifs/

# Run with MACE
python job.py cifs mace -o results

# Check results
cat results/CuPd/CuPd_descriptors.csv

# Resume if interrupted
python job.py cifs mace -o results --resume
```

## License

For internal/research use.
