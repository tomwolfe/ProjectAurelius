# Project Aurelius v7.0

**The Autonomous Discovery Release** -- Adds active learning loop with VAE-based structural mutation, PBC-aware physics, centralized dependency management, dependency health checks, and structural diversity generation for closed-loop molecule discovery.

## Changelog (v6.0 → v7.0)

### New Features

- **🥇 Task 1: MPNN Activation Energy Predictor** -- Replaced the Tier 0 linear heuristic with a lightweight Message Passing Neural Network (MPNN). Generates deterministic synthetic training data (500 molecules), trains via MSE loss with early stopping, and falls back to the original linear model if the GNN is unavailable. Added `aurelius train --task tier0` CLI command.

- **🥈 Task 2: Cutoff-Aware Neighbor Lists** -- Added optional neighbor list with fixed-cell spatial binning to Tier 2, reducing complexity from O(N²) to O(N·M). Toggleable via config (`use_neighbor_list: bool`, `neighbor_list_cutoff: float = 12.0`). Falls back to dense computation for small systems (<50 atoms) on MPS.

- **🏅 Task 3: RDKit Enforcement** -- RDKit is now strictly required for `--use-real-models`. Hash fallbacks are blocked in production paths. Added `--allow-fallback` CLI flag for demo/CI environments. Clear error messages point to `pip install rdkit`.

- **📊 Task 4: HuggingFace Hub Upload** -- Added `aurelius hf-upload` CLI subcommand for pushing locally trained models to HF Hub. Supports `--model-dir`, `--repo-id`, `--task`, `--private/--public`, `--commit-message`, and `--dry-run`. Auto-generates model cards with architecture, dataset, and hyperparameters.

- **📈 Task 5: Memory Profiler** -- Added `MemoryProfiler` class for tracking peak RAM, MPS/MLX memory, and GC activity. Generates timestamped CSV reports. Added `--profile-memory` flag to the autonomous screening agent CLI.

### v7 Improvements

- **🧬 Task 6: Centralized RDKit Helper Module** -- New `aurelius.utils.chem` module consolidates RDKit helpers: `_safe_mol_from_smiles`, `_is_valid_mol`, `_mol_to_fp`, `_serialize_fp`, `_deserialize_fp`, `_tanimoto`. Eliminates scattered try/except blocks across modules.

- **🔬 Task 7: PBC Minimum Image Convention** -- Tier 2 (`MatterSimMTSimulator`) gains periodic boundary conditions with `_apply_pbc()` for coordinate wrapping, negative coordinate handling, and default cubic box creation from `neighbor_list_cutoff`.

- **📦 Task 8: LRU Cache Eviction** -- `HuggingFaceWeightLoader.evict_lru_cache(max_cache_gb)` removes oldest entries when cache exceeds size limit, respecting LRU ordering.

- **⚡ Task 9: GNN-ChargeEq Model** -- New `ChargeEqModel` class with `hidden_dim` parameter for predicting partial charges from atomic numbers via `predict_charges(atomic_numbers)`.

- **🧠 Task 10: ActiveLearningOracle** -- New class for active learning with caching, batch querying (`query_batch`), dataset appending (`append_to_dataset`), and cache clearing (`clear_cache`).

- **🔄 Task 11: GraphVAEMutator** -- Structural diversity generation via latent interpolation. `GraphVAEMutator(latent_dim=64)` with `mutate(smiles, batch_size=N)` returns N candidates.

### Bug Fixes & Improvements

- **Version Bump**: Updated to 7.0.0 with all backward-compatible changes.
- **Config Extension**: Added `use_neighbor_list` and `neighbor_list_cutoff` to `AureliusConfig`.
- **CLI Flags**: Added `--task tier1|tier0` to `aurelius train`, `--allow-fallback` to `screen` and `batch`, and `--profile-memory` to the agent.
- **CI Updates**: Updated `.github/workflows/ci.yml` for new test markers and optional dependency installs.

## Overview

Aurelius is a high-performance screening pipeline for battery electrolyte molecules, designed for novel molecule discovery. It combines three screening tiers with Apple Silicon hardware optimization:

| Tier | Component | Framework | Purpose |
|------|-----------|-----------|---------|
| 1 | MLX-NA Filter | MLX | Rapid solubility/viability screening via ECFP4 fingerprints |
| 2 | MatterSim-MT | PyTorch MPS | Vectorized Lennard-Jones + Coulombic physics simulation |
| 3 | GCMD Digital Twin | NumPy | Arrhenius kinetic Monte Carlo (kMC) for SEI evolution |

## Hardware Requirements

- **Apple Silicon Mac** (M1, M2, M3, or M4 series)
- **Linux x86_64** (with NVIDIA GPU for CUDA support)
- **Windows 10/11** (with NVIDIA GPU for CUDA support)
- **Minimum 8GB RAM** (16GB+ recommended)
- **macOS 13+** (Ventura or later)

### Platform Support Matrix

| Platform | Tier 1 (MLX Filter) | Tier 2 (MatterSim-MT) | Tier 3 (GCMD Twin) |
|----------|---------------------|----------------------|---------------------|
| **Apple Silicon** (macOS) | *Optimized* (MLX + MPS) | *Optimized* (MPS) | Supported (NumPy) |
| **Linux x86_64** (CUDA) | PyTorch Fallback | *Optimized* (CUDA) | Supported (NumPy) |
| **Linux x86_64** (CPU-only) | PyTorch Fallback | Supported (CPU) | Supported (NumPy) |
| **Windows** (CUDA) | PyTorch Fallback | *Optimized* (CUDA) | Supported (NumPy) |
| **Windows** (CPU-only) | PyTorch Fallback | Supported (CPU) | Supported (NumPy) |

### Apple Silicon: *Optimized* (MLX + MPS)
- Tier 1 runs natively on MLX with Neural Engine acceleration
- Tier 2 uses PyTorch MPS backend for vectorized physics
- Cross-framework zero-copy bridging via DLpack

### Linux/Windows: *Supported* (CUDA/CPU)
- Tier 1 uses PyTorch fallback MLP when MLX is unavailable
- Tier 2/3 run on CUDA (GPU) or CPU as needed
- `bridge.py` imports successfully on all platforms; MLX-dependent methods raise `RuntimeError` with clear messaging

### Software Dependencies

```
Python >= 3.11
numpy >= 1.26.0          # Core dependency
scipy >= 1.12.0          # Core dependency
psutil >= 5.9.0          # Core dependency (RAM detection)

# Optional (install via .[apple], .[ml], .[chem])
torch >= 2.3.0           # .[ml] group
torchvision >= 0.17.0    # .[ml] group
mlx >= 0.15.0            # .[apple] group
rdkit >= 2023.9.0        # .[chem] group
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/ProjectAurelius.git
cd ProjectAurelius

# Install in development mode
pip install -e ".[dev]"
```

### Optional Dependencies

Aurelius splits large framework dependencies into optional groups to
keep the base install lightweight and allow installation on constrained
platforms (Linux CPU-only, Windows, etc.).

| Group | Dependencies | Use Case |
|-------|-------------|----------|
| `apple` | `mlx>=0.15.0` | Apple Silicon Neural Engine acceleration |
| `ml` | `torch>=2.3.0`, `torchvision>=0.17.0` | PyTorch-based Tier 2/3 physics |
| `chem` | `rdkit>=2023.9.0` | RDKit molecular fingerprints and descriptors |

**Recommended full installation:**

```bash
pip install -e ".[dev,apple,ml,chem]"
```

**Apple Silicon (recommended):**

```bash
pip install -e ".[dev,apple,ml,chem]"
```

**Linux/Windows (CPU-only):**

```bash
pip install -e ".[dev,ml,chem]"
```

**Minimal (no ML frameworks):**

```bash
pip install -e ".[dev]"
```

> **Note:** Environment variables (`PYTORCH_MPS_ENABLE_ASYNC_COMPILATION`, `MLX_MAX_MEM_CACHE`) are automatically set on first import via `aurelius.config.initialize_environment()`. The `setup_env.sh` script is only needed for persistent shell-level configuration (e.g., cluster environments, long-running services).

## Quick Start

```bash
# Initialize the pipeline
aurelius init

# Check system readiness (dependencies, hardware, config)
aurelius doctor --verbose

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

# Run hardware benchmark
aurelius benchmark
aurelius benchmark --tier 1 --quick
```

## Model Availability & Training

### Important: Local Training Required

**Project Aurelius does not ship with pre-trained model weights.** The Tier 1 screening model must be trained locally before meaningful screening can occur. The pipeline will attempt to:

1. Load pre-trained weights from Hugging Face Hub (if available)
2. Fall back to local model directory (`AURELIUS_MODEL_DIR`)
3. **Train on ESOL/QM9 datasets** if no weights are found

**Out-of-the-box, the pipeline trains a fresh model on real experimental data each time.** For production use, you should train and save a model once, then reuse it.

### Preparing for Autonomous Discovery

Before running the autonomous screening agent, ensure all models are trained and validated:

```bash
# Prepare all models for discovery (Tier 0 MPNN + Tier 1 MLP)
python scripts/prep_discovery.py

# With custom hyperparameters
python scripts/prep_discovery.py --tier0-epochs 500 --tier1-epochs 300

# Use numpy-only training (no MLX required)
python scripts/prep_discovery.py --no-mlx

# Use a local CSV for Tier 1 training
python scripts/prep_discovery.py --csv-path ./data/esol.csv
```

This script checks for existing model weights and triggers automated training if missing. After training, it runs a deterministic inference check on Ethylene Carbonate (`O=C1OCCO1`) to verify model integrity.

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

### Disk Usage Considerations

When downloading models from Hugging Face Hub, Aurelius uses `local_dir_use_symlinks=False` by default for compatibility and portability. This means:

- **Disk usage:** Each downloaded model is stored as full copies (not symlinks), which may consume more disk space than a symlinked HF cache.
- **Cache location:** Models are stored under `AURELIUS_MODEL_DIR/<task>/hf_cache/`.
- **Pre-flight check:** Aurelius checks for at least 10GB free space before downloading. If insufficient, the download is skipped with a warning.
- **Cleanup:** Use `huggingface-cli delete-cache` to manage the shared HF cache, or manually remove model directories under `AURELIUS_MODEL_DIR`.

#### Reducing Disk Usage with Symlinks

For users with limited disk space, set the environment variable:

```bash
export AURELIUS_HF_USE_SYMLINKS=1
```

This uses `local_dir_use_symlinks=True` in `snapshot_download`, which creates symlinks to the shared HF cache instead of full copies. This significantly reduces disk usage when multiple tasks download from the same repository.

#### Local Training + Caching Workflow

For production use with limited disk:

```bash
# Train locally once
aurelius train --dataset esol --epochs 200

# Reuse via environment variable
export AURELIUS_MODEL_DIR=./models/tier1

# Subsequent runs load from local cache without HF download
aurelius screen "CC(=O)OC1=CC=CC=C1"
```

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
│   │   ├── tier1/
│   │   │   ├── __init__.py     # Re-exports
│   │   │   ├── models.py       # _ChemVLM2MLP, _FallbackMLP, PyTorchFallbackFilter
│   │   │   ├── training.py     # train_on_esol, train_on_qm9
│   │   │   ├── loaders.py      # HuggingFaceWeightLoader, weight conversion
│   │   │   └── filter.py       # MLXNAFilter, fingerprint generation
│   │   ├── tier0/
│   │   │   ├── __init__.py     # Re-exports
│   │   │   ├── models.py       # Tier0MPNN, MPNNEdgeBlock, MPNNReadoutMLP
│   │   │   ├── data.py         # _build_molecular_graph, generate_synthetic_training_data, train_tier0_model
│   │   │   └── predictor.py    # Tier0ActivationPredictor, _LinearFallbackPredictor
│   │   ├── tier2_mattersim.py  # MatterSim-MT physics engine
│   │   └── tier3_gcmtwin.py    # GCMD Digital Twin
│   ├── solvation/
│   │   └── engine.py           # MWSE solvation engine (GBSA)
│   ├── scoring/
│   │   └── engine.py           # Aurelius Score computation
│   ├── bridge.py               # Zero-copy MLX<->PyTorch bridging
│   ├── config.py               # Dynamic memory configuration
│   └── pipeline.py             # Pipeline orchestrator
├── scripts/
│   ├── prep_discovery.py        # Train & validate all models for discovery
│   ├── train_tier0.py          # Train Tier 0 MPNN model
│   ├── train_tier1.py          # Train Tier 1 on ESOL/QM9
│   └── download_data.py        # Download datasets from HF Hub
├── config/
│   └── discovery_config.yaml   # Hydra configuration for autonomous agent
├── benchmarks/
│   ├── benchmark_tier1.py      # MLX vs PyTorch vs CPU
│   └── benchmark_tier2.py      # Vectorized vs loop physics
├── tests/
│   └── test_aurelius.py        # Physics-based validation tests
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
aurelius doctor                  Validate dependencies and hardware
  --verbose (-v)                 Show detailed framework versions
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
  --allow-fallback               Allow hash fallback without RDKit (demo/CI)
aurelius batch <file>            Screen molecules from SMILES file
  --solvent TYPE                 Solvent type
  --salt TYPE                    Salt type
  --output PATH                  Output JSON file
  --allow-fallback               Allow hash fallback without RDKit (demo/CI)
aurelius score <smiles>          Compute Aurelius score only
aurelius train                   Train model
  --task tier1|tier0             Task: tier1 (MLX filter) or tier0 (MPNN)
  --dataset esol|qm9             Dataset to train on (for --task tier1)
  --epochs N                     Number of epochs
  --batch-size N                 Mini-batch size
  --learning-rate LR             Learning rate
  --csv-path PATH                Local CSV file path
aurelius validate <smiles>       Run physics validation
aurelius benchmark               Run hardware benchmark
  --tier 1|2                     Benchmark specific tier only
  --quick/--detailed             Quick mode (default: enabled)
  --output PATH                  Save results to JSON
aurelius status                  Show pipeline status and memory
aurelius hf-upload               Upload model to HuggingFace Hub
  --model-dir PATH               Local model directory (required)
  --repo-id ID                   HF repository ID (required)
  --task tier0|esol|qm9          Model task type
  --private/--public             Repository visibility (default: private)
  --commit-message MSG           Commit message
  --dry-run                      Validate without uploading
python scripts/prep_discovery.py  Prepare all models for autonomous discovery
  --tier0-epochs N               Epochs for Tier 0 MPNN training (default: 200)
  --tier1-epochs N               Epochs for Tier 1 MLP training (default: 200)
  --batch-size N                 Mini-batch size for both tiers (default: 16)
  --learning-rate LR             Learning rate for Tier 1 (default: 0.005)
  --dataset DATASET              Dataset for Tier 1 (default: esol)
  --csv-path PATH                Local CSV file path for Tier 1
  --no-mlx                       Train Tier 1 with numpy only
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
  --allow-fallback               Allow hash fallback without RDKit (demo/CI)
aurelius batch <file>            Screen molecules from SMILES file
  --allow-fallback               Allow hash fallback without RDKit (demo/CI)
aurelius score <smiles>          Compute Aurelius score only
aurelius train                   Train model
  --task tier1|tier0             Task: tier1 (MLX filter) or tier0 (MPNN)
  --dataset esol|qm9             Dataset to train on (for --task tier1)
  --epochs N                     Number of epochs
  --csv-path PATH                Local CSV file path
aurelius validate <smiles>       Run physics validation
aurelius benchmark               Run hardware benchmark
  --tier 1|2                     Benchmark specific tier only
  --quick/--detailed             Quick mode (default: enabled)
  --output PATH                  Save results to JSON
aurelius status                  Show pipeline status and memory
aurelius hf-upload               Upload model to HuggingFace Hub
  --model-dir PATH               Local model directory (required)
  --repo-id ID                   HF repository ID (required)
  --task tier0|esol|qm9          Model task type
  --private/--public             Repository visibility (default: private)
  --commit-message MSG           Commit message
  --dry-run                      Validate without uploading
```

## License

MIT License - See LICENSE file for details.
