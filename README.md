# Project Aurelius v9.0

**The Bayesian Discovery Release** -- Introduces a Random Forest-driven active-learning loop: Morgan fingerprints are scored via Expected Improvement from a Random Forest surrogate, enabling intelligent candidate selection from mutation pools. The training data now uses real QM9 LUMO data, and all synthetic data generation has been removed.

## Changelog (v9.0 → v9.0)

### New Features

- **🥇 Task 1: Descriptor-Based Tier 0 Predictor** -- Replaced the MPNN activation energy predictor with a descriptor-based linear model using RDKit molecular descriptors (mol_weight, num_h_donors, num_h_acceptors, num_rotatable_bonds, logp, tpsa). Removed all hash-based synthetic data generation; training now requires real CSV data or QM9 LUMO datasets. Added `aurelius train --task tier0` CLI command.

- **🧠 Task 2: Bayesian-Guided Genetic Algorithm** -- New `GaussianProcessSurrogate` class enables Expected Improvement acquisition for active learning. The `FeedbackAdapter.update(X, y)` method retrains the GP with newly screened data, closing the active learning loop.

- **🧬 Task 3: Mutation Engine Propose Candidates** -- `MutationEngine.propose_candidates(n_candidates)` generates a large pool of unique variants for the GP to score, enabling the discovery loop to select high-EI candidates for expensive screening.

- **🏅 Task 4: HuggingFace Hub Upload** -- Added `aurelius hf-upload` CLI subcommand for pushing locally trained models to HF Hub and `--profile-memory` flag to the agent CLI.

### v7 Improvements

- **🧬 Active Learning Loop** -- The discovery loop now uses GP-guided candidate selection: mutation engine proposes 1000 candidates, GP surrogate scores via Expected Improvement, top candidates are screened, and results are fed back to the GP.

- **📦 LRU Cache Eviction** -- `HuggingFaceWeightLoader.evict_lru_cache(max_cache_gb)` removes oldest entries when cache exceeds size limit, respecting LRU ordering.

- **⚡ Device Consistency** -- All tensor operations ensure consistent device placement, preventing MPS/CPU device mismatch errors.

- **🔧 Dead Code Removal** -- Removed `_hash_descriptors`, `_ChemVLM2MLP`, `_FallbackMLP`, `generate_synthetic_training_data`, `_load_tier0_seed_smiles`, and all hash-based pseudo-random data generation logic.

### Bug Fixes & Improvements

- **Version Bump**: Updated to 7.0.0 with all backward-compatible changes.

- **CLI Flags**: Added `--task tier1|tier0` to `aurelius train`, and `--profile-memory` to the agent.
- **CI Updates**: Updated `.github/workflows/ci.yml` for new test markers and optional dependency installs.

## Overview

Aurelius is a high-performance screening pipeline for battery electrolyte molecules, designed for novel molecule discovery. The pipeline uses a Random Forest surrogate for Bayesian active learning:

| Component | Framework | Purpose |
|-----------|-----------|---------|
| 1 | MLX-NA Filter | Rapid solubility/viability screening via ECFP4 fingerprints |
| Surrogate | Random Forest | Expected Improvement acquisition for intelligent candidate selection |

## Overview

Aurelius is a high-performance screening pipeline for battery electrolyte molecules, designed for novel molecule discovery. The pipeline uses a Random Forest surrogate for Bayesian active learning:

| Component | Framework | Purpose |
|-----------|-----------|---------|
| 1 | MLX-NA Filter | Rapid solubility/viability screening via ECFP4 fingerprints |
| Surrogate | Random Forest | Expected Improvement acquisition for intelligent candidate selection |

## Hardware Requirements

- **Apple Silicon Mac** (M1, M2, M3, or M4 series)
- **Linux x86_64** (with NVIDIA GPU for CUDA support)
- **Windows 10/11** (with NVIDIA GPU for CUDA support)
- **Minimum 8GB RAM** (16GB+ recommended)
- **macOS 13+** (Ventura or later)

### Platform Support Matrix

| Platform | Tier 1 (MLX Filter) |
|----------|---------------------|
| **Apple Silicon** (macOS) | *Optimized* (MLX + MPS) |
| **Linux x86_64** (CUDA) | PyTorch Fallback |
| **Linux x86_64** (CPU-only) | PyTorch Fallback |
| **Windows** (CUDA) | PyTorch Fallback |
| **Windows** (CPU-only) | PyTorch Fallback |

### Apple Silicon: *Optimized* (MLX + MPS)
- Tier 1 runs natively on MLX with Neural Engine acceleration
- Cross-framework zero-copy bridging via DLpack

### Linux/Windows: *Supported* (CUDA/CPU)
- Tier 1 uses PyTorch fallback MLP when MLX is unavailable
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
| `ml` | `torch>=2.3.0`, `torchvision>=0.17.0` | PyTorch-based ML models |
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

## Aurelius Score v9.0

The composite Aurelius Score (S_A_v9.0) combines three components:

| Component | Weight | Description |
|-----------|--------|-------------|
| Sigma (σ) | 0.40 | Structural viability from Tier 1 screening |
| GWP Penalty | 0.10 | Global Warming Potential adjustment |

**Formula:** S_A_v9.0 = 0.40*σ + 0.10*MX - 0.10*GWP

Scores range from 0-100. Molecules with S_A_v9.0 >= 65 are considered viable.


## Scientific References

### Tier 1: Molecular Fingerprints & Solubility
- Morgan, H. L. "The Generation of a Unique Machine Description for Chemical Structures." *J. Chem. Doc.* 1965, 5, 107-117.
- Delaney, S. J. "ESOL: Estimating Aqueous Solubility Directly from Structure." *J. Chem. Inf. Model.* 2004, 44(6), 1947-1949. DOI: 10.1021/ci034236x
- Ramakrishnan, R. et al. "Quantum Chemistry Structures and Properties of 134 Kilo Molecules." *Sci. Data* 2014, 1, 140035. DOI: 10.1038/sdata.2014.35


```
ProjectAurelius/
├── src/aurelius/
│   ├── agent/
│   │   ├── __init__.py         # Re-exports
│   │   ├── loop.py             # DiscoveryLoop (Bayesian active learning)
│   │   ├── mutation.py         # SELFIES-based mutation engine
│   │   ├── reporting.py        # Report generation
│   │   ├── state.py            # Agent state management
│   │   └── surrogate.py        # Random Forest surrogate
│   ├── cli_scripts/
│   │   ├── __init__.py
│   │   ├── agent.py            # Autonomous screening agent
│   │   ├── download_data.py    # Download datasets from HF Hub
│   │   ├── prep_discovery.py   # Train & validate all models for discovery
│   │   └── train_tier1.py      # Train Tier 1 on ESOL/QM9
│   ├── data/
│   │   └── params.py           # Data parameters
│   ├── hub/
│   │   └── uploader.py         # Model upload to HF Hub
│   ├── memory/
│   │   ├── __init__.py
│   │   └── profiler.py         # Memory profiling utilities
│   ├── models/
│   ├── screening/
│   │   ├── tier1/
│   │   │   ├── __init__.py     # Re-exports
│   │   │   ├── models.py       # MLXBackend, PyTorchBackend, model_factory
│   │   │   ├── training.py     # train_on_esol, train_on_qm9
│   │   │   ├── loaders.py      # HuggingFaceWeightLoader, weight conversion
│   │   │   └── filter.py       # MLXNAFilter, fingerprint generation
│   │   ├── tier0/
│   │   │   ├── __init__.py     # Re-exports
│   │   │   ├── models.py       # _MPNNEdgeBlockBackend, _MPNNReadoutMLPBackend, model_factory
│   │   │   ├── data.py         # _build_molecular_graph, generate_synthetic_training_data, train_tier0_model
│   │   │   └── predictor.py    # Tier0ActivationPredictor, _LinearFallbackPredictor
│   ├── scoring/
│   │   ├── __init__.py
│   │   └── oracle.py           # MPNN-based property oracle
│   ├── pipeline.py             # Pipeline orchestrator
│   ├── config.py               # Dynamic memory configuration
│   └── bridge.py               # Zero-copy MLX<->PyTorch bridging
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
ProjectAurelius/
├── src/aurelius/
│   ├── screening/
│   │   ├── tier1/
│   │   │   ├── __init__.py     # Re-exports
│   │   │   ├── models.py       # MLXBackend, PyTorchBackend, model_factory
│   │   │   ├── training.py     # train_on_esol, train_on_qm9
│   │   │   ├── loaders.py      # HuggingFaceWeightLoader, weight conversion
│   │   │   └── filter.py       # MLXNAFilter, fingerprint generation
│   │   ├── tier0/
│   │   │   ├── __init__.py     # Re-exports
│   │   │   ├── models.py       # _MPNNEdgeBlockBackend, _MPNNReadoutMLPBackend, model_factory
│   │   │   ├── data.py         # _build_molecular_graph, generate_synthetic_training_data, train_tier0_model
│   │   │   └── predictor.py    # Tier0ActivationPredictor, _LinearFallbackPredictor
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
   --use-real-models              Use real models (default: enabled)
   --demo                         Use synthetic data (demo mode)
aurelius batch <file>            Screen molecules from SMILES file
  --solvent TYPE                 Solvent type
  --salt TYPE                    Salt type
  --output PATH                  Output JSON file
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

## License

MIT License - See LICENSE file for details.
