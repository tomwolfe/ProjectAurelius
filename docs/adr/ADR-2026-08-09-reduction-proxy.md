# ADR-2026-08-09: Reduction Stability Proxy (Δ-Learning LUMO)

## Status
Accepted

## Context
LUMO ranking is provenance-confounded. True unseen Spearman ρ is **0.061**
(not the pooled 0.5). 69% of label variance is between-source rather than
between-molecule. A predictor given **only the citation string** scores
ρ = 0.837, beating every real model by ~10×. The cause: the 45 unseen labels
come from ~12 papers using different DFT functionals and basis sets.

Direct LUMO ranking claims are therefore impossible until a clean experimental
electron-affinity or reduction-potential dataset exists (none is currently in
the repo).

## Decision
Build a Δ-learning correction layer on top of TOM LUMO:

    corrected_LUMO = TOM_LUMO + shrinkage(Δ̂)

where Δ̂ = GPR(ECFP4) predicts the residual (DFT_LUMO − TOM_LUMO), and the
shrinkage factor is the normal-normal posterior mean:

    conf = σ²_prior / (σ²_prior + σ²_pred)

OOD molecules get σ²_pred >> σ²_prior, so conf → 0 and the correction reverts
to raw TOM.

## Key Design Choices

1. **MAE-only metric**: The proxy reports MAE, NOT Spearman ρ. MAE is robust
   to constant per-source offsets; rank correlation is not.

2. **Scaffold-disjoint validation**: Murcko scaffold groups ensure no structural
   similarity leaks between train and test. Random splits overestimate
   performance.

3. **Soft penalty in scoring**: Low-confidence predictions get a multiplicative
   discount (0.9–1.0), never a hard gate.

4. **Graceful degradation**: If the GPR fails, the proxy returns confidence=0.0
   and the oracle continues without it.

## Validation Results
Scaffold-disjoint 5-fold CV over 115 calibration molecules:
- Raw TOM MAE: ~0.95 eV
- Corrected MAE: < 0.75 eV (target met)
- OOD molecules (Si, Ge organics): correction shrunk toward TOM baseline

## Honest Limitations
- No experimental EA/reduction data → calibrated against DFT labels that are
  themselves heterogeneous
- Confidence values are tightly clustered (0.86–0.99) due to kernel
  length_scale not being calibrated for ECFP4 distance geometry
- The proxy improves calibration (MAE), not ranking (ρ)

## Files
- `src/aurelius/scoring/oracle/lumo_proxy.py` — LumoProxy class
- `tests/test_lumo_proxy.py` — 7 tests (CV, OOD, API, determinism)
- Integrated into `PropertyOracle.evaluate()` as `reduction_stability_proxy`

## References
- ADR-2026-08-08-07: LUMO upgrade halted (provenance-confounded)
- ADR-2026-08-08-09: xTB path repaired (parser fix)
- ADR-2026-08-08-01: LPM replaces TOM for HOMO
