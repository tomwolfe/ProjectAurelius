# Carbonate Kernel — Case Study

| Metric        | Before (Generic GC) | After (Tuned Kernel) | Improvement |
|---------------|--------------------:|-------------------------:|------------:|
| Spearman ρ    | 0.45                | 0.72                     | +60%        |
| MAE           | 0.55                | 0.28                     | -49%        |

**Why it helped:** The tuned TOM offsets (homo\_offset=0.15, lumo\_offset=-0.10) correct for the systematic overstabilisation of carbonate HOMO levels by the generic particle-in-a-box model. The three dielectric-specific GC fragments (cyclic carbonate, linear carbonate, ethyl carbonate) raise the dielectric proxy from 1.9 to a physically realistic range (8–16 ε).

| Detail                | Value        |
|-----------------------|--------------|
| Training molecules    | 28           |
| Audit status          | PASS         |
| max MW                | 250.0 Da     |
| max LogP              | 1.5          |
| Max conjugation length| 16           |

*Source: `docs/examples/kernels/carbonate_v1.json`*
