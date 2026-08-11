# ADR-2026-08-11-04: ALPB solution-phase ΔSCF EA is anti-correlated with experiment

## Status
Rejected (solution-phase ALPB path disabled for ranking) — extends ADR-2026-08-11, ADR-2026-08-11-03.

## Context

ADR-2026-08-11 added solution-phase ΔSCF EA via xTB's ALPB continuum solvation model, and ADR-2026-08-11-03 anchored it to 10 solution-phase CV onset measurements. The physical reasoning is sound: ALPB shifts the anion energy by the continuum solvation free energy, which should stabilize the anion and increase EA in solvent.

The solution-phase path is now the *default* when xTB is available (`with_auto_solvent()`). If its predictions are worse than the gas-phase baseline, every molecule evaluated through the oracle is being ranked on a misleading axis — quietly, because the fallback is silent and the feature is on by default.

## Measurement

ALPB-corrected ΔSCF EA vs 10 experimental solution-phase CV onsets (1M LiPF6 EC:DMC vs Li/Li+, converted to vacuum scale via +1.39 eV):

| molecule | class | raw ALPB-xTB | calibrated | experimental |
|----------|-------|-------------:|-----------:|-------------:|
| EC       | carbonate |  6.916 | 4.694 | 3.28 |
| PC       | carbonate |  4.993 | 2.994 | 3.35 |
| DMC      | carbonate |  5.034 | 3.030 | 3.68 |
| DEC      | carbonate |  4.986 | 2.987 | 3.78 |
| FEC      | carbonate |  5.539 | 3.476 | 2.82 |
| VC       | carbonate |  5.218 | 3.192 | 2.65 |
| DME      | ether     | −1.337 | −2.603 | 5.22 |
| THF      | ether     | −1.803 | −3.015 | 4.85 |
| ACN      | nitrile   |  3.396 | 1.582 | 4.72 |
| sulfolane | sulfone  |  2.842 | 1.092 | 4.45 |

| metric | value |
|--------|------:|
| Spearman ρ (raw vs exp) | **−0.915** |
| Spearman ρ (calibrated vs exp) | **−0.915** |
| MAE (eV, calibrated) | 2.659 |

The affine calibration cannot change Spearman ρ by construction. The correlation is *negative*: molecules the experiment says are easiest to reduce (high EA) are assigned the lowest ALPB-xTB values.

## Diagnosis

The failure is concentrated in ethers (DME, THF) and, to a lesser extent, nitriles and sulfones. ALPB is a *polar continuum* model: it stabilizes charges in high-ε media and does little in low-ε media. But the ranking error is not a uniform offset — it is a reversal, which points to a sign or reference issue specific to how ALPB corrects the *anion* total energy for weakly-coordinating solvents.

Three hypotheses, not yet distinguished:

1. **ALPB solvent mismatch.** The calibration uses `acetonitrile` (ε=37.5) for all molecules, but ethers in a low-ε medium receive almost no solvation stabilization. Evaluating them in acetonitrile over-stabilizes the anion relative to the true medium, but that would *raise* EA, not reverse the order — so this alone cannot explain the sign.
2. **Vertical EA in solvent.** Both single points use the neutral geometry. In solvent, the anion's relaxed geometry may differ substantially, and a vertical attachment energy is not the quantity ALPB is parametrized to correct.
3. **xTB anion instability in solvent.** The ALPB correction assumes a bound anion in a dielectric continuum. For ethers, the anion may be unbound (negative gas-phase EA), and ALPB's continuum stabilization is insufficient — the total energy is then physically meaningless.

Without distinguishing these, the mechanism cannot be fixed.

## Decision

### 1. Disable solution-phase ALPB as a default ranking path

`ReductionOracle` reverts to gas-phase-only evaluation by default. `with_auto_solvent()` is retained as an opt-in experimental path but emits a warning that solution-phase EA is unvalidated (ρ = −0.915 vs the only available calibration set).

### 2. Gas-phase EA remains the validated reduction axis

The gas-phase ΔSCF EA path is validated at ρ = 0.912 against 40 experimental gas-phase EAs (permutation control p < 0.001). It is physically meaningful as a relative ranking even for solution-phase chemistry — gas-phase EA captures the intrinsic electron-accepting ability of the molecule, which is the dominant term; solvation is a secondary correction that can be added once it works.

### 3. Solution-phase work deferred

A validated solution-phase path requires:
- Distinguishing the three hypotheses above (likely requires ORCA + an explicit-shell or PCM model for a few ethers).
- A larger calibration set (≥20 molecules spanning carbonates, ethers, nitriles, sulfones) so a non-linear correction could be fitted if the relationship is systematic.
- Until then, the ALPB-xTB solution-phase numbers are not used for ranking or reporting.

## Consequences

- **Positive.** The default reduction axis is now the validated gas-phase path. No silent fallback to a misleading model.
- **Positive.** Honest: a plausible physics-based correction was tried, measured, and found to be actively harmful for the molecules that matter most (ethers). Recording this prevents future rediscovery.
- **Neutral.** The gas-phase/solution-phase distinction is preserved in the code (`solvent` parameter, `load_experimental_ea_gas()`, `load_experimental_ea_solution()`). Re-enabling solution-phase later is a single-line change once the physics is fixed.
- **Negative.** Solution-phase reduction potentials remain uncalibrated — the largest residual gap in the reduction axis, now explicitly deferred rather than implicitly claimed.

## Verification

- `pytest tests/test_reduction_oracle.py` — all gas-phase tests pass; solution-phase tests updated to reflect opt-in status.
- `python benchmarks/benchmark_reduction_axis.py` — now filters gas-only (n=40, ρ=0.912).

## Files
- `src/aurelius/scoring/oracle/reduction.py` — default reverted to gas-phase
- `benchmarks/benchmark_reduction_axis.py` — gas-only filter
- `tests/test_reduction_oracle.py` — solution-phase tests updated

## References
- ADR-2026-08-11-solution-phase-ea.md
- ADR-2026-08-11-03-solution-phase-calibration.md
- ADR-2026-08-10-dscf-electron-affinity.md
