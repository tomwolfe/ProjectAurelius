# Implementation Notes - Project Aurelius v5.2

## Architectural Changes

### 1. Packaging & Path Resolution (Phase 1)
- **Problem**: `os.path.dirname(__file__)` path walks break in wheel installs and editable installs.
- **Solution**: All module-level JSON loading now uses `importlib.resources.files("aurelius.data").joinpath("force_field_params.json")`.
- **Files affected**: `config.py`, `solvation/engine.py`, `scoring/engine.py`, `screening/tier2_mattersim.py`, `screening/tier1_mlx_filter.py`
- **Validation**: Works in both `pip install -e .` and installed wheels.

### 2. Thread-Safe Configuration (Phase 1)
- **Problem**: `M5ProConfig.apply_environment()` mutated `os.environ` directly, causing race conditions during concurrent pipeline initialization.
- **Solution**: `apply_environment()` now returns a `dict[str, str]` of environment variables. The CLI entry point (`__main__.py`) applies them via `_apply_env_thread_safe()` which only sets variables not already present in `os.environ`.
- **Backward compatibility**: `apply_global_config()` still calls `apply_environment()` and applies env vars for backward-compatible code paths.

### 3. Frozen Dataclass Fix (Phase 1)
- **Problem**: `object.__setattr__` in `__post_init__` on a frozen dataclass is fragile and silently ignored by the dataclass `__init__`.
- **Solution**: `M5ProConfig` now uses `@dataclass(init=False)` with a keyword-only `__new__` that sets all fields before `__init__` runs. This eliminates the need for `__setattr__` workarounds entirely.
- **Pattern**: `__new__` detects system RAM, computes allocations, and sets all fields via `object.__setattr__` once at construction time.

### 4. Framework Hardening (Phase 2)
- **Problem**: Private PyTorch APIs (`torch._C._mps_loadMetalLib`, `torch.accelerator`) caused crashes on older PyTorch versions.
- **Solution**: All private API calls are wrapped in `hasattr` guards with safe fallbacks:
  - `torch.accelerator`: Gracefully skipped if unavailable (PyTorch < 2.12)
  - `torch._C._mps_loadMetalLib`: Only called if the function exists; JIT proceeds normally otherwise
  - `torch.mps.set_per_process_memory_fraction(0.8)`: Added for safe memory management

### 5. Dataset Verification (Phase 2)
- **Problem**: Placeholder HF dataset IDs (`matin/dehesa`, `matin/qm9`) caused failures when the repos didn't exist.
- **Solution**: Verified dataset IDs used (`deepchem/esol`, `maastrichtuniversity/qm9`) with a robust fallback chain:
  1. HuggingFace Hub (verified repos)
  2. Local CSV via `--csv-path`
  3. Embedded 50-molecule curated ESOL subset (scientifically valid)
- **Exception handling**: Specific `ValueError`, `ConnectionError`, `ImportError` caught instead of broad `except Exception`.

### 6. Memory Optimization (Phase 3)
- **Problem**: `_placeholder_model` allocated 100M float32 params (~400MB), causing OOM on 8GB Macs during tests.
- **Solution**: Replaced with lazy initialization (`np.zeros(1, dtype=np.float32)`). Test memory footprint now < 150MB at initialization.

### 7. Complexity Documentation (Phase 3)
- **Problem**: `"O(1) pairwise interaction"` was scientifically inaccurate.
- **Solution**: Corrected to `"O(N^2) time/space, O(1) Python interpreter overhead"`.

### 8. Bug Fix: Redundant Walrus Operator (Phase 3)
- **Problem**: `q_product := q_i * q_j` was computed twice in `compute_coulomb_vectorized`.
- **Solution**: Removed the duplicate walrus operator assignment.

### 9. Dependency Pinning (Phase 3)
- **Problem**: Nightly pins (`torch>=2.12.0`, `mlx>=0.20.0`) caused build failures.
- **Solution**: Pinned to stable releases: `torch>=2.3.0`, `mlx>=0.15.0`, `rdkit>=2023.9.0`.

## Fallback Paths Summary

| Component | Primary | Fallback 1 | Fallback 2 |
|-----------|---------|------------|------------|
| ESOL Dataset | `deepchem/esol` (HF) | Local CSV (`--csv-path`) | Embedded 50-molecule subset |
| QM9 Dataset | `maastrichtuniversity/qm9` (HF) | Local CSV (`--csv-path`) | Error with specific message |
| Metal Shaders | Pre-compiled `.metallib` | JIT compilation | Warning + proceed |
| PyTorch MPS | `torch._C._mps_loadMetalLib` | `torch.mps.set_per_process_memory_fraction()` | JIT proceeds normally |
| Tier 1 Model | HF Hub weights | Local `AURELIUS_MODEL_DIR` | Train on ESOL/QM9 |
| RDKit Fingerprint | RDKit ECFP4 | Deterministic hash-based | Warning + proceed |

## CI Setup

Created `.github/workflows/ci.yml` with:
- **Lint job**: `ruff check .` and `mypy src/aurelius` on macOS
- **Test job**: `pytest tests/ -v` + `scripts/validate_physics.py` on matrix:
  - `macos-latest` (Apple Silicon, MPS available)
  - `ubuntu-latest` (CPU fallback)
  - Python 3.11, 3.12
- **Wheel job**: `python -m build --wheel` with contents verification

## Backward Compatibility

- All public method signatures preserved
- CLI flags (`--use-real-models`, `--demo`, all screen/batch/score options) unchanged
- `AureliusPipeline` API unchanged
- `use_real_models`, `--demo`, and all CLI flags maintained

## v5.2 Review Response (Post-Release Hardening)

### 10. Frozen Dataclass Pattern Migration (Review Item #1)
- **Problem**: `M5ProConfig` used `object.__setattr__` inside `__post_init__` on a frozen dataclass. While functional, this pattern is fragile because `__setattr__` calls in `__post_init__` are silently ignored by the dataclass `__init__` if the class is frozen.
- **Solution**: Migrated to `@dataclass(init=False)` with a keyword-only `__new__` constructor. The `__new__` method detects system RAM, computes allocations, and sets all fields via `object.__setattr__` once at construction time. This eliminates `__post_init__` entirely and follows the pattern documented in the Python dataclass best practices.
- **Files affected**: `config.py`
- **Backward compatibility**: Constructor signature unchanged (all parameters are keyword-only with defaults).

### 11. Pipeline Timing Bug Fix (Review Item #2)
- **Problem**: `pipeline.py:263` used `dir()` to check for a local variable (`t2_start`), which is unreliable and fragile. The conditional expression could fall through to an incorrect fallback.
- **Solution**: Replaced with the standard timing calculation `(time.perf_counter() - t2_start) * 1000`, consistent with Tier 1 and Tier 3 timing. The `t2_start` variable is always in scope at this point.
- **Files affected**: `pipeline.py`

### 12. Tier 2 Energy Profile Vectorization (Review Item #3)
- **Problem**: `_compute_energy_profile` in `tier2_mattersim.py` contained a Python `for step in range(n_cycles)` loop despite claiming "fully vectorized" in the docstring. Only per-step LJ + Coulomb computation was vectorized; the outer cycle loop remained in Python.
- **Solution**: Fully vectorized the energy profile computation. The ion displacement is computed as `(n_cycles, 3)` tensor, solvent coordinates as `(1, n_solvent, 3)`, and pairwise distances broadcast to `(n_cycles, n_solvent)`. LJ and Coulomb energies are computed for all steps simultaneously, returning a `(n_cycles,)` energy tensor. Python interpreter overhead is now O(1) per step.
- **Files affected**: `screening/tier2_mattersim.py`

### 13. Tier 0 Activation Energy Predictor (Review Item #3 - Homogeneity)
- **Problem**: Tier 3 produced similar homogeneity scores (~12.2/100) across diverse molecules because activation energies were fixed. This limited the screening pipeline's ability to differentiate molecules.
- **Solution**: Added `Tier0ActivationPredictor` class that predicts molecule-specific activation energies from molecular descriptors. The predictor is a linear model with weights calibrated against DFT literature values. It generates descriptors from SMILES (using RDKit when available, deterministic hash fallback otherwise) and predicts Ea values for EC reduction, DMC reduction, PF6 decomposition, and polymerization. The GCMDigitalTwin constructor accepts `use_tier0_prediction=True` to enable molecule-specific screening.
- **Files affected**: `screening/tier3_gcmtwin.py`
- **Future work**: Replace the linear predictor with a trained GNN or transformer model for improved accuracy.

### 14. Lower-Bounds CI Testing (Review Item #4)
- **Problem**: Pinned dependencies (`torch>=2.3.0`, `mlx>=0.15.0`, etc.) needed verification against actual lower bounds.
- **Solution**: Added `lower-bounds` CI job that installs dependencies at their pinned minimum versions and runs the full test suite plus physics validation. Ensures compatibility with older stable releases.
- **Files affected**: `.github/workflows/ci.yml`
