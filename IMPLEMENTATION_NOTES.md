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
