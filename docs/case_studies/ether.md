# Ether Kernel — Case Study

| Metric        | Before (Generic GC) | After (Tuned Kernel) | Improvement |
|---------------|--------------------:|-------------------------:|------------:|
| Spearman ρ    | 0.50                | 0.81                     | +62%        |
| MAE           | 0.45                | 0.19                     | -58%        |

**Why it helped:** The ether-specific kernel adjusts TOM offsets (homo\_offset=0.22, lumo\_offset=0.05) to account for the electron-rich oxygen lone pairs that the generic model systematically overestimates. The Li⁺ solvation GC fragment (glyme chelation) and two dielectric fragments (linear ether, cyclic ether) capture the solvent's dual role as a low-viscosity diluent and weak Li⁺ binder.

| Detail                | Value        |
|-----------------------|--------------|
| Training molecules    | 22           |
| Audit status          | PASS         |
| max MW                | 200.0 Da     |
| max LogP              | 2.0          |
| Max conjugation length| 12           |

*Source: `docs/examples/kernels/ether_v1.json`*
