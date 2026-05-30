# Refactoring Notes — Project Aurelius v9.0

## Architectural Changes

This document summarises the architectural changes made in the v9.0 refactoring
that closes the Bayesian active-learning loop for molecule discovery.

---

## Phase 1: Purge Zombie Physics Code

**File: `src/aurelius/types.py`**

- **Removed** `DesolvationPathResult`, `Tier2Result`, `SEIEvolution`,
  `GCMDTwinResult`, `GCMDTConfig`, `AureliusScoreResult` — all were
  fake physics wrappers with no real implementation.
- **Kept** `MoleculeInput`, `MLXFilterResult`.
- **Added** `OracleResult` — a clean, flat dataclass holding only the
  HOMO/LUMO properties returned by the PropertyOracle.

**File: `src/aurelius/pipeline.py`**

- Replaced `PretrainedGNNOracle` import with `PropertyOracle`.
- Removed all code that faked `Tier2Result` or `DesolvationPathResult`
  (e.g. mapping LUMO gap → `barrier_height_eV`).
- `screen_molecule()` now returns a flat dict with `tier1`, `tier2`
  (raw oracle dict), and `score` — no nested physics wrappers.

**File: `src/aurelius/agent/loop.py`**

- Replaced direct screening of all valid_candidates with a
  **Bayesian active-learning loop** (see Phase 4).

**File: `src/aurelius/agent/reporting.py`**

- No changes needed; `ScreeningResult` is unchanged.

---

## Phase 2: Fix the Oracle (Scientific Validity)

**File: `src/aurelius/scoring/oracle.py`**

- **Deleted** `PretrainedGNNOracle` (and `MLPNNOracle` alias) — the
  fake HuggingFace model path `aurelius/qm9-gnn` and the GNN-inference
  code that passed flat 2048-bit ECFP4 fingerprints to a "GNN".
- **Added** `PropertyOracle` — a RandomForest-based oracle trained on
  ECFP4 fingerprints.  Falls back to synthetic data if no QM9 dataset
  is available (no heavy PyTorch/PyG dependency).
- `evaluate(smiles)` returns a dict with `homo_eV`, `lumo_eV`,
  `lumo_gap_eV`, `dipole_debye`.

---

## Phase 3: Real Battery Electrolyte Objective

**File: `src/aurelius/scoring/oracle.py` (integrated)**

- The score computation now uses a bounded objective function:
  - **Reward** molecules whose LUMO falls in the SEI formation window
    (−1.5 eV to −0.5 eV, reducible to form SEI).
  - **Reward** molecules whose HOMO < −6.0 eV (oxidative stability).
  - **Penalise** molecules outside this window using a Gaussian penalty
    (not linear decay).
- This replaces the previous `100.0 - lumo_gap * 5.0` linear formula
  which was scientifically meaningless.

---

## Phase 4: Close the Active Learning Loop

**File: `src/aurelius/agent/loop.py`**

- `DiscoveryLoop.execute()` now implements the full Bayesian loop:
  1. `engine.propose_candidates(n_candidates=1000)` generates a large pool.
  2. Pool is featurised into ECFP4 fingerprints (`X_pool`).
  3. If `self.feedback._surrogate` is fitted, `expected_improvement()`
     scores the pool → top `batch_size` candidates selected.
  4. If not fitted, first batch is selected randomly.
  5. Only selected candidates are screened.
  6. `self.feedback.update(X_new, y_new)` re-trains the GP surrogate.

**File: `src/aurelius/agent/state.py`**

- Added `FeedbackAdapter._rng` (seeded `np.random.default_rng(42)`)
  so the loop can select random candidates when the surrogate is
  unfitted.

---

## Dependencies

No new heavy dependencies introduced.  The refactoring stays within:
- `rdkit` (molecule handling)
- `scikit-learn` (RandomForest, GaussianProcessRegressor)
- `numpy` (array operations)
- `selfies` (mutation engine, unchanged)

---

## Files Modified

| File | Change |
|------|--------|
| `src/aurelius/types.py` | Removed fake physics types; kept only `MoleculeInput`, `MLXFilterResult`, `OracleResult` |
| `src/aurelius/pipeline.py` | Replaced `PretrainedGNNOracle` import; removed fake Tier2/Desolvation path mapping |
| `src/aurelius/scoring/oracle.py` | Replaced `PretrainedGNNOracle` with `PropertyOracle` (RandomForest) |
| `src/aurelius/agent/loop.py` | Replaced direct screening with Bayesian active-learning loop |
| `src/aurelius/agent/state.py` | Added `_rng` attribute to `FeedbackAdapter` |
