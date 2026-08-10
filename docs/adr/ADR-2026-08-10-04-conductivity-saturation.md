# ADR-2026-08-10-04: Walden conductivity proxy was saturating on the best solvents

## Status
Accepted

## Context

`predict_ionic_conductivity_proxy` computes a Walden-style figure of merit

    sigma ~ (eps - 1) * gaussian(li_solvation) / eta

and clamped the result to `[0, 10]`. That clamp was calibrated when
`dielectric` was a compressed 1–15 unitless proxy. ADR-2026-08-07-04 replaced
that proxy with the Kirkwood-Fröhlich model, which reports the true epsilon
scale (EC ≈ 82, PC ≈ 66), roughly a five-fold increase in the numerator — and
the clamp was never re-derived.

Measured over the 51 known electrolytes:

- **15 of 51 (29%) returned exactly 10.000.**
- Among them: EC, FEC, PC, ethylene sulfate, succinonitrile — i.e. the
  high-dielectric solvents a battery campaign is actually trying to rank.
- Only 35 distinct values across 51 molecules.

The existing docstring already flagged this honestly and noted that nothing in
`pipeline.py` consumes `conductivity_proxy`, so no ranking claim was corrupted.
But a figure of merit that cannot distinguish EC from PC is not usable for the
mixture work it was built for.

## Decision

Replace the hard clamp with a smooth saturating map:

    sigma = C_MAX * w / (1 + w),     w = walden / WALDEN_HALF_SATURATION

with `C_MAX = 10.0` (preserving the reported scale) and
`WALDEN_HALF_SATURATION = 4.89`, the **median raw Walden product** over the
known-electrolyte set, so half the output range is spent where real solvents
live rather than on the long tail.

The map is strictly increasing on `[0, inf)`, so **it cannot reorder any two
candidates**. This is a resolution fix, not a ranking change, and no previously
reported ranking can be affected by it. `test_monotone_in_the_walden_product`
enforces this property directly.

The batch implementation was updated identically and is pinned to the scalar
path by `test_batch_matches_scalar`.

## Results

| metric (51 known electrolytes) | before | after |
|--------------------------------|-------:|------:|
| pinned at the ceiling          |  15/51 | **0/51** |
| distinct values                |     35 |    48 |
| std                            |  3.809 | 2.675 |
| max                            | 10.000 | 8.855 |

Top of the ranking is now resolved: EC 8.86, maleic anhydride 8.61, ethylene
sulfate 8.55, FEC 8.17, PC 8.14 — previously all exactly 10.000.

## Honest limitations

- This fixes *dynamic range*, not *accuracy*. The Walden proxy is still an
  uncalibrated figure of merit with no experimental conductivity validation
  behind it. Nothing here justifies quoting it as a predicted mS/cm.
- The half-saturation constant is fitted to the known-electrolyte distribution.
  If the search moves far outside that chemistry the constant should be
  re-derived, and the value being distribution-dependent is a weakness.
- Mixture conductivity still assumes ideal mixing for dielectric and Li⁺
  solvation and Grunberg-Nissan for viscosity. Non-ideality is only partially
  captured by the existing Hildebrand miscibility gate.

## Files
- `src/aurelius/scoring/oracle/gc.py`
- `tests/test_conductivity_saturation.py`
