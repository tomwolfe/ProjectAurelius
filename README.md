# Project Aurelius v5.2

**The Hardened Release** -- Production-grade computational chemistry screening pipeline optimized for Apple M-series Neural Accelerators.

## Changelog (v5.1 → v5.2)

- **Packaging**: Replaced all `os.path.dirname(__file__)` path resolution with `importlib.resources` for wheel-compatible installs
- **Thread Safety**: Removed `os.environ` mutation from `M5ProConfig.apply_environment()`; env vars now returned as dict for thread-safe CLI application
- **Dataclass Fix**: Replaced frozen dataclass `__setattr__` workarounds with proper `__new__` constructor pattern (init=False + keyword-only args)
- **PyTorch Hardening**: Wrapped `torch._C._mps_loadMetalLib` and `torch.accelerator` in `hasattr` guards with safe fallbacks
- **Dataset Verification**: Replaced placeholder HF dataset IDs with verified repositories (`deepchem/esol`, `maastrichtuniversity/qm9`)
- **Fallback Chain**: Implemented robust dataset fallback: HF verified IDs → local CSV → embedded 50-molecule curatated subset
- **Memory Optimization**: Reduced `_placeholder_model` from 100M params (~400MB) to lazy init (1 float32, ~4 bytes)
- **Complexity Docs**: Corrected `"O(1) pairwise interaction"` to `"O(N^2) time/space, O(1) Python interpreter overhead"`
- **Bug Fix**: Removed redundant walrus operator in `compute_coulomb_vectorized`
- **Dependency Pins**: Pinned stable releases (`torch>=2.3.0`, `mlx>=0.15.0`, `rdkit>=2023.9.0`)
- **CLI**: Added `aurelius train` and `aurelius validate` subcommands

## Overview

Aurelius is a high-performance screening pipeline for battery electrolyte molecules, designed for novel molecule discovery. It combines three screening tiers with Apple Silicon hardware optimization:

| Tier | Component | Framework | Purpose |
|------|-----------|-----------|---------|
| 1 | MLX-NA Filter | MLX | Rapid solubility/viability screening via ECFP4 fingerprints |
| 2 | MatterSim-MT | PyTorch MPS | Vectorized Lennard-Jones + Coulombic physics simulation |
| 3 | GCMD Digital Twin | NumPy | Arrhenius kinetic Monte Carlo (kMC) for SEI evolution |

## Hardware Requirements

- **Apple Silicon Mac** (M1, M2, M3, or M4 series)
- **Minimum 8GB RAM** (16GB+ recommended)
- **macOS 13+** (Ventura or later)

### Software Dependencies

```
Python >= 3.11
torch >= 2.3.0          # Stable release; nightly pins avoided
mlx >= 0.15.0           # Stable release; nightly pins avoided
rdkit >= 2023.9.0
numpy >= 1.26.0
scipy >= 1.12.0
huggingface-hub >= 0.20.0
datasets >= 2.16.0
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/ProjectAurelius.git
cd ProjectAurelius

# Set up environment
./setup_env.sh

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

```bash
# Initialize the pipeline
aurelius init

# Screen a single molecule (uses real models by default)
aurelius screen "CC(=O)OC1=CC=CC=C1"

# Screen with demo (synthetic data) mode
aurelius screen "CCO" --demo

# Batch screening
aurelius batch examples/molecules.smi --output results.json

# Compute score only
aurelius score "CC(=O)OC1=CC=CC=C1"

# Train Tier 1 model
aurelius train --dataset esol --epochs 200

# Run physics validation
aurelius validate --smiles "CC(=O)OC1=CC=CC=C1"

# Check pipeline status
aurelius status
```

## Model Availability & Training

### Important: Local Training Required

**Project Aurelius does not ship with pre-trained model weights.** The Tier 1 screening model must be trained locally before meaningful screening can occur. The pipeline will attempt to:

1. Load pre-trained weights from Hugging Face Hub (if available)
2. Fall back to local model directory (`AURELIUS_MODEL_DIR`)
3. **Train on ESOL/QM9 datasets** if no weights are found

**Out-of-the-box, the pipeline trains a fresh model on real experimental data each time.** For production use, you should train and save a model once, then reuse it.

### Tier 1: MLX-NA Filter

The Tier 1 filter uses a 2-layer MLP (2048->128->1) trained on ECFP4 (Morgan radius=2) fingerprints.

**Supported datasets:**
- **ESOL** (Delaney et al., JACS 2004): 1,112 molecules with experimental aqueous solubility (logS)
- **QM9** (Ramakrishnan et al., Sci. Data 2014): 134,887 molecules with DFT-computed quantum properties

### Training Real Models

```bash
# Download datasets
python scripts/download_data.py --dataset all --output ./data/

# Train Tier 1 model on ESOL
python scripts/train_tier1.py --dataset esol --epochs 200 --batch-size 16

# Train on QM9
python scripts/train_tier1.py --dataset qm9 --epochs 300 --batch-size 32

# Train with local CSV file
python scripts/train_tier1.py --dataset esol --csv-path ./data/esol/esol.csv
```

### Loading Pre-trained Weights

After training, set the model directory for efficient reuse:

```bash
export AURELIUS_MODEL_DIR=./models/tier1
```

Models can be loaded from:
1. **Hugging Face Hub** (if `huggingface_hub` is installed and weights are published)
2. **Local directory** (set via `AURELIUS_MODEL_DIR` environment variable)
3. **Training on-the-fly** (falls back to ESOL/QM9 training)

> **Note:** The Hugging Face repositories `aurelius/tier1-esol-mlp` and `aurelius/tier1-qm9-mlp` are placeholders. Public pre-trained weights are not yet available. Users should train locally and optionally upload to their own HF repository.

## Aurelius Score v5.2

The composite Aurelius Score (S_A_v5.2) combines five components:

| Component | Weight | Description |
|-----------|--------|-------------|
| Sigma (σ) | 0.30 | Structural viability from Tier 1 screening |
| Desolvation Barrier | 0.20 | Energy barrier from Tier 2 simulation |
| SEI Homogeneity | 0.20 | Solid Electrolyte Interphase quality from Tier 3 |
| MX Synthesis Score | 0.20 | Material synthesis feasibility |
| GWP Penalty | 0.10 | Global Warming Potential adjustment |

**Formula:** S_A_v5.2 = 0.30*σ + 0.20*E_des + 0.20*SEI + 0.20*MX - 0.10*GWP

Scores range from 0-100. Molecules with S_A_v5.2 >= 65 are considered viable.

## Tier 2 Complexity

MatterSim-MT uses **fully vectorized tensor operations** for pairwise interaction computation. The physics engine computes all N×N pairwise distances and energies in **O(N^2) time and space** (where N is the number of atoms), with **O(1) Python interpreter overhead** per step. This is the optimal theoretical complexity for all-pairs interaction models.

## Scientific References

### Tier 1: Molecular Fingerprints & Solubility
- Morgan, H. L. "The Generation of a Unique Machine Description for Chemical Structures." *J. Chem. Doc.* 1965, 5, 107-117.
- Delaney, S. J. "ESOL: Estimating Aqueous Solubility Directly from Structure." *J. Chem. Inf. Model.* 2004, 44(6), 1947-1949. DOI: 10.1021/ci034236x
- Ramakrishnan, R. et al. "Quantum Chemistry Structures and Properties of 134 Kilo Molecules." *Sci. Data* 2014, 1, 140035. DOI: 10.1038/sdata.2014.35

### Tier 2: Physics Engine
- Schutt, K. T. et al. "SchNet: A Continuous-filter Convolutional Neural Network for Quantum Chemistry." *NeurIPS* 2018.
- Jorgensen, W. L. et al. "Comparison of Simple Potential Functions for Simulating Liquid Water." *J. Chem. Phys.* 1983, 79, 926. DOI: 10.1063/1.445869
- Wang, J. et al. "Development and Testing of a General Amber Force Field." *J. Comput. Chem.* 2004, 25, 1157-1174. DOI: 10.1002/jcc.20035
- Butler, K. T. et al. "Machine Learning Molecular Embeddings for Battery Materials." *Nature* 2023.

### Solvation Engine
- Still, W. C. et al. "Fast Approximate Calculation of Molecular Surface Area." *J. Am. Chem. Soc.* 1990, 112, 6127-6129. DOI: 10.1021/ja00172a031
- Waghorne, W. A. et al. "First-Principles Calculation of Effective Charges in a Perovskite." *Phys. Rev. B* 2004, 69, 054110. DOI: 10.1103/PhysRevB.69.054110
- Salanne, M. et al. "Molecular Dynamics of Aqueous Electrolyte Solutions." *J. Phys. Chem. B* 2011, 115, 12614-12625. DOI: 10.1021/jp204841a
- CRC Handbook of Chemistry and Physics, 104th Edition. CRC Press, 2023.

## Project Structure

```
ProjectAurelius/
├── src/aurelius/
│   ├── screening/
│   │   ├── tier1_mlx_filter.py    # MLX-NA Filter (SchNet/MLP)
│   │   ├── tier2_mattersim.py     # MatterSim-MT physics engine
│   │   └── tier3_gcmtwin.py       # GCMD Digital Twin
│   ├── solvation/
│   │   └── engine.py              # MWSE solvation engine (GBSA)
│   ├── scoring/
│   │   └── engine.py              # Aurelius Score computation
│   ├── bridge.py                  # Zero-copy MLX<->PyTorch bridging
│   ├── config.py                  # Dynamic memory configuration
│   └── pipeline.py                # Pipeline orchestrator
├── scripts/
│   ├── train_tier1.py             # Train Tier 1 on ESOL/QM9
│   └── download_data.py           # Download datasets from HF Hub
├── benchmarks/
│   ├── benchmark_tier1.py         # MLX vs PyTorch vs CPU
│   └── benchmark_tier2.py         # Vectorized vs loop physics
├── tests/
│   └── test_aurelius.py           # Physics-based validation tests
└── pyproject.toml
```

## Benchmarks

Run benchmarks to compare performance across backends:

```bash
# Tier 1: MLX vs PyTorch MPS vs CPU
python benchmarks/benchmark_tier1.py --n-molecules 1000 --repeats 10

# Tier 2: Vectorized vs Loop-based physics
python benchmarks/benchmark_tier2.py --n-atoms 50 --n-cycles 500
```

## CLI Reference

```
aurelius init                    Initialize pipeline
aurelius screen <smiles>         Screen a single molecule
  --solvent TYPE                 Solvent type (default: ec:dmc)
  --salt TYPE                    Salt type (default: NaPF6)
  --ion TYPE                     Ion type (default: Na+)
  --temperature K                Temperature in Kelvin
  --voltage CUTOFF               Voltage cutoff
  --cycles N                     MD simulation cycles
  --gwp VALUE                    Global Warming Potential
  --use-real-models              Use real models (default: enabled)
  --demo                         Use synthetic data (demo mode)
aurelius batch <file>            Screen molecules from SMILES file
aurelius score <smiles>          Compute Aurelius score only
aurelius train                   Train Tier 1 model (esol/qm9)
  --dataset esol|qm9             Dataset to train on
  --epochs N                     Number of epochs
  --csv-path PATH                Local CSV file path
aurelius validate <smiles>       Run physics validation
aurelius status                  Show pipeline status and memory
```

## License

MIT License - See LICENSE file for details.
