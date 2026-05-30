# Refactoring Notes — Project Aurelius v9.0

## Architectural Changes

This document summarises the architectural changes made in the v9.0 refactoring
that closes the Bayesian active-learning loop for molecule discovery.

---

## Phase 1: Purge Zombie Physics Code

**File: `src/aurelius/screening/tier1/backend_mlx.py`**

- **Deleted** `backend_mlx.py` — the MLX-compatible 2-layer MLP and all weight-conversion
  logic are no longer needed. PyTorch is now the sole inference backend.

**Files: `tests/test_physics.py`, `tests/test_scoring.py`, `tests/test_tier2.py`, `tests/test_tier3.py`**

- **Deleted** all four test files. The fake physics engines, scoring engine,
  MatterSim, and GCMDigitalTwin were replaced by real ML oracles.
  These tests are no longer needed.

**File: `src/aurelius/config.py`**

- **Removed** `tier2_mattersim_enabled` and `tier3_gcmtwin_enabled` fields.
- **Removed** `mattersim_quantization` and `gcmd_quantization` fields.
- **Removed** GCMD kMC steps reference from `memory_report()`.

**File: `src/aurelius/screening/tier1/loaders.py`**

- **Renamed** `convert_mlx_to_torch_weights` → `convert_numpy_to_torch_weights`.
- **Renamed** `load_pytorch_fallback_with_mlx_weights` → `load_pytorch_fallback`.
- **Removed** `MLXBackend` import; now imports `PyTorchBackend` from `models`.
- **Updated** `_load_from_hf_hub` and `_load_from_local` to instantiate
  `PyTorchBackend` instead of `MLXBackend`.

---

## Phase 2: Scientific Validity (Oracle & Scoring)

**File: `src/aurelius/scoring/oracle.py`**

- **Replaced** `PropertyOracle`'s `RandomForest` backend with a proper
  MPNN (Message Passing Neural Network) using PyTorch Geometric.
- The oracle now operates on molecular graph structures (atom features + bond indices)
  rather than hashed ECFP4 fingerprints, enabling genuine extrapolation to
  novel chemical space.
- `evaluate(smiles)` still returns `homo_eV`, `lumo_eV`, `lumo_gap_eV`, `dipole_debye`,
  but predictions now come from a trained graph network.

**File: `src/aurelius/pipeline.py`**

- **Replaced** the toy linear formula `100.0 - lumo_gap * 5.0` with a proper
  Gaussian penalty-based scoring function:
  - Rewards LUMO ∈ [-1.5, -0.5] eV (SEI formation window).
  - Rewards HOMO < -6.0 eV (oxidative stability).
  - Applies Gaussian PDF-based penalties for out-of-range values.

---

## Phase 3: Close the Active Learning Loop

**File: `src/aurelius/agent/state.py`**

- **Fixed** `FeedbackAdapter.record()` to append actual 2048-bit ECFP4 fingerprint
  arrays (`result.fingerprint`) instead of raw SMILES strings.
- Added `fingerprint` field to `ScreeningResult` dataclass.

**File: `src/aurelius/agent/loop.py`**

- **Added** explicit GP retraining after each batch: `self.feedback.update(X_new, y_new)`
  is now called at the end of every generation's screening phase.
- The loop now properly closes the Bayesian active-learning cycle:
  1. Select candidates via Expected Improvement (or random for first batch).
  2. Screen selected candidates.
  3. **Update the GP surrogate** with new observations.
  4. Record feedback and convergence state.

---

## Dependencies

| File | Change |
|------|--------|
| `pyproject.toml` | Removed `mlx` optional dependency; version bumped to 9.0.0 |
| `src/aurelius/screening/tier1/backend_mlx.py` | Deleted |
| `src/aurelius/screening/tier1/loaders.py` | Removed MLX conversion, renamed functions |
| `src/aurelius/config.py` | Removed `tier2_mattersim_enabled`, `tier3_gcmtwin_enabled` |
| `src/aurelius/agent/loop.py` | Added `X_new, y_new = feedback.update()` call |
| `src/aurelius/agent/state.py` | Fixed `record()` to append fingerprint arrays |
| `src/aurelius/scoring/oracle.py` | Replaced RandomForest with MPNN-based oracle |
| `src/aurelius/pipeline.py` | Gaussian penalty scoring replaces linear formula |

## Files Deleted

| File | Reason |
|------|--------|
| `src/aurelius/screening/tier1/backend_mlx.py` | MLX backend purged; PyTorch is sole backend |
| `tests/test_physics.py` | Fake physics removed |
| `tests/test_scoring.py` | Fake scoring engine removed |
| `tests/test_tier2.py` | MatterSim replaced by real oracle |
| `tests/test_tier3.py` | GCMDigitalTwin replaced by real oracle |
