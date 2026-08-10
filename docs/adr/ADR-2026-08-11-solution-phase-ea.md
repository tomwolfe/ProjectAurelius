# ADR-2026-08-11: Solution-phase ΔSCF EA via xTB ALPB

## Status
Accepted — extends ADR-2026-08-10 (gas-phase ΔSCF EA).

## Context

ADR-2026-08-10 established ΔSCF electron affinity as the reduction axis,
reaching Spearman ρ = 0.91 against 40 gas-phase experimental EAs. That
validation is gas-phase. Battery reduction happens *in solvent* — the anion
produced by electron attachment is stabilised by the dielectric medium, and
the relevant quantity for SEI formation is the solution-phase reduction
potential, not the gas-phase EA.

The code already had a `solvent` parameter stubbed in `_run_xtb_energy` and
`compute_dscf_ea`, but two defects prevented it from being usable:

1. **Cache key collision.** `ReductionOracle.evaluate` keyed the cache by
   canonical SMILES alone. Evaluating the same molecule with
   `solvent=None` (gas) and `solvent="acetonitrile"` returned the gas-phase
   result for both — the second call hit the cache from the first. This was
   a silent correctness bug.

2. **No dynamic solvent selection.** The `solvent` parameter defaulted to
   `None` (gas phase). A chemist had to know to pass `--alpb acetonitrile`
   manually. The oracle already has access to the Kirkwood-Fröhlich
   dielectric proxy (`predict_dielectric_proxy`), which predicts ε on the
   physical scale (~2–90 for battery electrolytes).

## Decision

### 1. Cache key includes solvent

The cache key is now `f"{smiles}|{solvent}"`. Same molecule with different
solvents produces distinct cache entries. This is a correctness fix, not a
feature — the previous behaviour was a bug.

### 2. Dielectric-to-solvent mapping

Added `solvent_from_dielectric(epsilon)` — a nearest-neighbour lookup over
nine named xTB ALPB solvents spanning ε = 1.9 (hexane) to ε = 80 (water).
For battery electrolytes (ε ≈ 2–40) this maps to hexane–acetonitrile.

### 3. `ReductionOracle.with_auto_solvent()`

A classmethod that constructs an oracle with the solvent auto-selected from
the molecule's predicted dielectric. Falls back to "acetonitrile" when the
dielectric proxy is unavailable. This is the default path for wet-lab loop
integration — the oracle now evaluates EA in a physically relevant medium
without requiring the caller to know xTB's solvent table.

### 4. Solvent recorded on result

`ReductionResult.solvent` is persisted in the cache and returned in
`to_dict()`. Auditability: a chemist reading a cached result can see which
solvent produced it.

## Consequences

- **Positive.** Solution-phase EA is now the default when xTB is available.
  ALPB stabilises the anion, so solution-phase EA differs from gas-phase —
  the `test_gas_vs_solution_differ` test confirms this for EC.
- **Positive.** The cache collision bug is fixed; same-molecule cross-solvent
  evaluation is now correct.
- **Neutral.** Gas-phase EA is still available by passing `solvent=None`.
  The experimental calibration (`_EA_CALIBRATION`) was fitted on gas-phase
  data, so gas-phase predictions remain on the experimental scale.
- **Negative (known).** Solution-phase ALPB corrections are not yet calibrated
  against experimental reduction potentials in solvent. ALPB is a first-principles
  correction to the gas-phase scale — it improves physical realism but the
  absolute values are not yet anchored to CV onsets. This is the next calibration
  step (requires a small set of solution-phase CV measurements).

## Verification

- `pytest tests/test_reduction_oracle.py` — 23 passed (9 new tests added).
- `pytest tests/test_net_progress.py` — 1 passed ($P_{\text{net}} > 0$ maintained).
- New tests cover: dielectric mapping, cache key correctness, ALPB flag
  passing, gas vs. solution differentiation, auto-solvent selection.
