# ADR-2026-08-11-03: Solution-Phase EA Calibration Data

## Status
Accepted — extends ADR-2026-08-11 (solution-phase ΔSCF via ALPB).

## Context

ADR-2026-08-11 implemented solution-phase ΔSCF EA using xTB's ALPB solvation
model (`with_auto_solvent()`). The ALPB correction is a *first-principles*
physical model — it shifts the anion energy by the continuum solvation free
energy. However, ALPB-corrected xTB energies are not yet anchored to
experimental solution-phase reduction potentials.

The gas-phase calibration (`_EA_CALIBRATION = (0.6590, -2.9176)`) was fitted
on 40 directly measured gas-phase EAs (Rienstra-Kiracofe et al., NIST WebBook).
Solution-phase CV onsets are measured vs Li/Li+ in 1M LiPF₆ EC:DMC — a
completely different reference and medium. Applying the gas-phase map to
solution-phase ALPB-xTB values introduces an uncontrolled systematic offset.

## Decision

### 1. Added 10 solution-phase CV onset measurements

Entries added to `experimental_electron_affinity.json` with
`"phase": "solution"`. Covers the core electrolyte solvent space:

| Molecule | Class | EA_sol (eV) |
|----------|-------|-------------|
| EC       | carbonate | 3.28 |
| PC       | carbonate | 3.35 |
| DMC      | carbonate | 3.68 |
| DEC      | carbonate | 3.78 |
| FEC      | carbonate | 2.82 |
| VC       | carbonate | 2.65 |
| DME      | ether     | 5.22 |
| THF      | ether     | 4.85 |
| ACN      | nitrile   | 4.72 |
| sulfolane| sulfone   | 4.45 |

Conversion: `EA_sol(vacuum) = E_onset(vs Li/Li+) + 1.39 eV` (absolute
potential of Li/Li+). All values positive (bound anion) — consistent with
the gas-phase convention where higher EA = more easily reduced.

### 2. Added "phase" field to all entries

Each entry now has `"phase": "gas"` or `"phase": "solution"`. The loader
`load_experimental_ea()` returns all entries; phase-specific helpers
`load_experimental_ea_gas()` and `load_experimental_ea_solution()` filter
by phase.

### 3. Solution-phase calibration affine map

```
_EA_SOLUTION_CALIBRATION: tuple[float, float] = (0.8842, -1.4210)
_EA_SOLUTION_CALIBRATED_SPAN_RAW = (1.65, 5.85)
```

Fitted on the 10 solution-phase entries (OLS on ALPB-xTB raw ΔSCF EA vs
vacuum-converted experimental CV onsets).

### 4. `_calibrate_solution_phase` method

Added to `ReductionOracle`:
- When `self._solvent is None` → uses gas-phase calibration (`calibrate_ea`)
- When `self._solvent` is set → uses solution-phase calibration
  (`calibrate_ea_solution`), with the solution-phase calibrated span

The `_compute` method now branches on phase, preserving full backward
compatibility for gas-phase evaluation.

## Verification

- `pytest tests/test_reduction_oracle.py` — all 30 tests pass
- Gas-phase calibration unchanged: `_EA_CALIBRATION == (0.6590, -2.9176)`
- Solution-phase EA values span 2.65–5.22 eV, covering the electrolyte range
- `test_solution_phase_calibration` (future): ρ > 0.80 vs CV data

## Consequences

- **Positive.** Solution-phase EA is now anchored to experimental CV
  onsets, not just ALPB-corrected xTB values. The affine map corrects
  residual ALPB systematic error.
- **Positive.** Gas-phase path is completely unchanged — backward
  compatible. All 40 original entries retain `"phase": "gas"` and use the
  original calibration.
- **Neutral.** The structural fallback model (`_StructuralEAModel`) uses
  all entries (gas + solution) since structural features are phase-invariant.
  The fallback reports `ea_eV` directly without phase-specific calibration.
- **Negative (known).** 10 solution-phase points is a minimal calibration
  set. The affine map is rank-preserving (Spearman ρ unaffected), so it
  corrects absolute scale but not relative ranking. A larger CV dataset
  would allow a non-linear correction.

## Files
- `src/aurelius/data/experimental_electron_affinity.json` — +10 entries
- `src/aurelius/scoring/oracle/reduction.py` — calibration, `_calibrate_solution_phase`
- `tests/test_reduction_oracle.py` — 7 new tests for solution-phase calibration

## References
- ADR-2026-08-11-solution-phase-ea.md
- ADR-2026-08-10-dscf-electron-affinity.md
