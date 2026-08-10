# ADR-2026-08-09: Reduction Stability Proxy (Δ-Learning LUMO)

## Status
Accepted — amended by ADR-2026-08-09-02 (xTB calibration set)

## Context
LUMO ranking on the *external* benchmark is provenance-confounded. True unseen
Spearman ρ is **0.061** (not the pooled 0.5). 69% of label variance is
between-source rather than between-molecule. A predictor given **only the
citation string** scores ρ = 0.837, beating every real model by ~10×. The cause:
the 45 unseen labels come from ~12 papers using different DFT functionals and
basis sets.

ADR-2026-08-07: The halted LUMO work (ADR-2026-08-08-07) was based on
`orbital_calibration.json`, which suffers the same confound (53 DFT sources,
citation-only ρ = 0.680).

ADR-2026-08-09-02 (this amendment): A new internally consistent calibration
set `lumo_calibration_xtb.json` has been generated — 231 molecules with all
LUMO values computed by the same GFN2-xTB method, calibrated to the
B3LYP/6-311++G** scale via an OLS affine map. The confound audit confirms
zero provenance signal (citation-only ρ = 0.000, between-source fraction = 0.00).

## Decision
Build a Δ-learning correction layer on top of TOM LUMO:

    corrected_LUMO = TOM_LUMO + shrinkage(Δ̂)

where Δ̂ = GPR(ECFPP4) predicts the residual (xTB_LUMO − TOM_LUMO), and the
shrinkage factor is the normal-normal posterior mean:

    conf = σ²_prior / (σ²_prior + σ²_pred)

OOD molecules get σ²_pred >> σ²_prior, so conf → 0 and the correction reverts
to raw TOM.

### Split Calibration (ADR-2026-08-09-02)
- **HOMO (Δ-correction)**: Trained on `orbital_calibration.json` (115 DFT-B3LYP
  entries). The HOMO model is MAE-robust; the LPM HOMO already correlates
  ρ = 0.91 against NIST experimental IPs, so ranking is not the bottleneck.
- **LUMO (Δ-correction + LumoProxy)**: Trained on `lumo_calibration_xtb.json`
  (231 xTB entries). All values from the same quantum-chemical method, so
  ranking on this set reflects chemistry, not methodology.

## Key Design Choices

1. **MAE + internally consistent ranking**: On the xTB set, both MAE and
   Spearman ρ are valid metrics. The external benchmark remains MAE-only.

2. **Scaffold-disjoint validation**: Murcko scaffold groups ensure no structural
   similarity leaks between train and test. Random splits overestimate
   performance.

3. **Soft penalty in scoring**: Low-confidence predictions get a multiplicative
   discount (0.9–1.0), never a hard gate.

4. **Graceful degradation**: If the GPR fails, the proxy returns confidence=0.0
   and the oracle continues without it.

5. **xTB scale alignment**: xTB eigenvalues are mapped onto the B3LYP/6-311++G**
   scale via the OLS affine map in `quantum.py`. This is a change of units
   (not a ranking fit — an affine map cannot change Spearman ρ).

## Validation Results
Scaffold-disjoint 5-fold CV over 231 xTB calibration molecules:
- Raw TOM LUMO MAE: ~0.95 eV
- Corrected MAE: < 0.75 eV (target met)
- Citation-only ρ on this set: 0.000 (confound-free — rank is meaningful)

On the external benchmark (confounded):
- Corrected ρ: ~0.06 (unseen) — ranking is provenance, not chemistry
- Corrected MAE: valid metric, reflects calibration quality

## Honest Limitations
- No experimental EA/reduction data → calibrated against xTB values, not ground
  truth. The xTB method is internally consistent but tight-binding; the affine
  map to B3LYP scale adds systematic error.
- Confidence values are tightly clustered (0.86–0.99) due to kernel
  length_scale not being calibrated for ECFP4 distance geometry.
- The LUMO model improves calibration (MAE) and ranking (ρ on xTB set), but
  external benchmark ranking remains confounded.
- HOMO Δ-correction uses orbitally_calibration.json DFT labels (53 sources);
  this is MAE-valid but not ranking-valid. HOMO ranking comes from LPM (ρ=0.91
  against NIST IPs), not the Δ-correction.

## Files
- `src/aurelius/data/lumo_calibration_xtb.json` — 231-molecule xTB calibration set
- `src/aurelius/scoring/oracle/lumo_proxy.py` — LumoProxy class (xTB calibration)
- `src/aurelius/scoring/oracle/delta_correction.py` — DeltaCorrection (split HOMO/LUMO)
- `tests/test_lumo_proxy.py` — 7 tests (CV, OOD, API, determinism)
- `tests/test_label_confound.py` — audit guard tests
- `benchmarks/audit_label_confound.py` — includes xTB set in audit
- Integrated into `PropertyOracle.evaluate()` as `reduction_stability_proxy`

## References
- ADR-2026-08-08-07: LUMO upgrade halted (provenance-confounded)
- ADR-2026-08-08-09: xTB path repaired (parser fix)
- ADR-2026-08-08-01: LPM replaces TOM for HOMO
