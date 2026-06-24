# Sulfone Kernel — Certified Case Study

| Metric        | Before (Generic GC) | After (Certified Kernel) | Improvement |
|---------------|--------------------:|-------------------------:|------------:|
| Spearman ρ    | 0.35                | 0.68                     | +94%        |
| MAE           | 0.65                | 0.31                     | -52%        |

**Why it helped:** The sulfone kernel applies a negative LUMO offset (lumo\_offset=-0.22) to capture the strong electron-withdrawing effect of the sulfonyl group, which the generic TOM model underestimates. The three GC fragments (dielectric, viscosity, Li⁺ solvation all centred on the S(=O)(=O) motif) jointly predict sulfolane's characteristically high dielectric constant (42 ε) and moderate viscosity.

| Detail                | Value        |
|-----------------------|--------------|
| Training molecules    | 18           |
| Audit status          | FAIL         |
| Failure reason        | Coverage below confidence threshold |
| max MW                | 350.0 Da     |
| max LogP              | 1.0          |
| Max conjugation length| 14           |

> **Note:** This kernel's audit FAIL reflects low training set coverage (n=18). Predictions on sulfone molecules outside the calibrated fragment space will carry elevated uncertainty. Consider expanding the training set before production use.

*Source: `docs/examples/kernels/sulfone_v1.json`*
