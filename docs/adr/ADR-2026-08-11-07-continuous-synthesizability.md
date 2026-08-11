# ADR-2026-08-11-07: Continuous synthesizability score replaces discrete depth

## Status
Accepted — extends ADR-2026-08-10-02 (grounding degeneracy fix).

## Context

The `combined_grounding_score()` in `brics.py` applied a depth-dependent penalty
based on `brics_retrosynthetic_depth()`, which returns discrete values {1,2,3,4,5}.
This translated to only 4 distinct penalty values {1.0, 0.9, 0.8, 0.7}, providing
coarse selection pressure. The `continuous_synthesizability_score()` function
already existed in `retrosynthetic.py` but was not wired into the main scoring path.

## Decision

Replace the discrete depth penalty in `combined_grounding_score()` with the
continuous synthesizability score as a fourth weighted component:

    0.30 × BRICS_coverage + 0.25 × template_feasibility
    + 0.25 × route_confidence + 0.20 × continuous_synthesizability

The continuous score combines:
1. **Direct precursor match** (weight 0.40 within the score) — how well the
   molecule itself aligns with commercial precursors.
2. **Fragment coverage** (weight 0.35) — weighted precursor coverage of BRICS
   fragments after one decomposition.
3. **Decomposition efficiency** (weight 0.25) — how quickly decomposition
   converges to purchasable fragments.

## Results

Measured on known electrolytes vs adversarial Frankensteins:

| set | mean score | std |
|-----|-----------|-----|
| Known electrolytes (10) | 0.973 | 0.035 |
| Frankensteins (5) | 0.327 | 0.230 |
| **Gap** | **0.647** | — |

The continuous score provides smooth selection pressure across [0,1], replacing
the previous 4-level discrete penalty. Known electrolytes cluster near 1.0 while
adversarial structures (triepoxides, peroxides, azides, silyl ethers) are
strongly penalized (0.025-0.573).

## Files
- `src/aurelius/agent/mutation/brics.py` — updated `combined_grounding_score()`
- `src/aurelius/agent/mutation/retrosynthetic.py` — `continuous_synthesizability_score()` (already existed)

## References
- ADR-2026-08-10-02-grounding-degeneracy.md
