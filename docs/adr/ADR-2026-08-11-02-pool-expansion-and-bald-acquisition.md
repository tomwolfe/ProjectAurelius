# ADR-2026-08-11-02: Pool Expansion + BALD/Pareto-UCB Acquisition

## Status
Accepted — extends ADR-2026-08-10-03 (acquisition negative result),
ADR-2026-08-11 (solution-phase EA), ADR-2026-08-07-04 (R_g bottlenecks).

## Context

ADR-2026-08-10-03 measured the closed-loop benchmark at p = 0.111 — the
acquisition function (Expected Impact) was indistinguishable from random at
the available budget (~61 unmeasured molecules). Two root causes were
identified:

1. **Pool too small.** A 61-molecule candidate pool is saturated: any
   diversity-weighted acquisition function converges to random because there
   is no room for selective exploration. ADR-2026-08-10-03 concluded that
   pool expansion must precede acquisition improvements.

2. **No epistemic uncertainty targeting.** The acquisition was built on
   conformal interval width (aleatoric) plus expected impact (decision
   relevance), but neither targets Bayesian epistemic variance — the quantity
   BALD acquisition minimises.

## Decision

### 1. Pool expansion to ≥200 via BRICS harvesting (Phase 1)

Added `_harvest_from_top_candidates(top_n=10, max_products=150)` that:
- Pre-ranks candidates by a lightweight oracle quality score (no conformal
  intervals, no DoA — just point predictions)
- Decomposes the top-10 into BRICS fragments (keeping dummy-atom labels
  for valid recombination)
- Recombines via `BRICSBuild` (complementary pair filtering native to RDKit)
- Deduplicates against the existing pool by canonical SMILES

`expand_candidate_pool` now delegates to this function. `MIN_POOL_SIZE`
raised from 100 to 200.

```
# Before: pool stays at ~61 → acquisition ≈ random
# After:  pool grows to ≥200 → diversity-based acquisition has room
```

### 2. BALD acquisition score (Phase 2)

Implemented `bald_acquisition_score(mol, gpr_model)`:
- For a Gaussian posterior, BALD = H[p(y|x,D)] − E[H[p(y|x,θ)]]
  reduces monotonically to the GPR posterior variance σ*²(x)
- Extracts posterior std from the `MLXGPRSurrogate` (M5 Pro GPU) or
  falls back to sklearn
- Normalised via x/(1+x) to [0, 1]

Added `_compute_bald_scores` that batches all candidates in a single
MLX GPU call, reusing the variance already computed in
`MLXGPRSurrogate.predict_batch`.

### 3. Pareto UCB score (Phase 2)

Implemented `pareto_ucb_score(predictions, uncertainties, all_predictions)`:
- Computes UCB per objective: μ + β·σ (β = 1.5)
- Orients objectives so higher is always better:
  - HOMO: negated (high |homo| = stable)
  - LUMO: negated (high LUMO = hard to reduce, stable)
  - dielectric: as-is (higher = better solvation)
  - viscosity: negated (low viscosity = fast ion transport)
- A candidate scores 1.0 if it is non-dominated on the UCB frontier, 0.0
  if dominated

### 4. New weight schema

```
DEFAULT_WEIGHTS = {
    "uncertainty":     0.20,   # conformal interval width
    "expected_impact": 0.15,   # decision-boundary crossing probability
    "novelty":         0.15,   # distance to calibration set
    "doa_proximity":   0.10,   # domain-of-applicability boundary
    "bias":            0.05,   # systematic bias magnitude
    "bald":            0.20,   # epistemic variance reduction (new)
    "pareto_ucb":      0.15,   # Pareto-frontier UCB score (new)
}
```

BALD (0.20) and Pareto UCB (0.15) together contribute 35% of the
acquisition score, ensuring the suggester targets epistemic
uncertainty on the multi-objective frontier rather than aleatoric
interval width alone.

## Consequences

- **Positive.** Pool expansion enables all downstream acquisition to have
  a meaningful search space — the prerequisite for any acquisition function
  to beat random.
- **Positive.** BALD targets the GPR posterior variance, which directly
  measures how much a measurement would tighten the model's confidence in
  a region of interest.
- **Positive.** Pareto UCB balances exploration and exploitation across the
  multi-objective electrolyte design space (stability, solvation, viscosity).
- **Neutral.** The pre-scoring pass adds one lightweight evaluation round
  before expansion; it uses only point predictions (fast, no GPR), so the
  overhead is negligible.
- **Negative.** The BALD/GPR path requires a fitted sklearn GPR model.
  When unavailable (e.g. CI without sklearn), BALD scores gracefully
  degrade to 0.0 and the acquisition falls back to the existing terms.

## Files
- `src/aurelius/agent/experiment_suggester.py` — pool expansion, BALD, Pareto UCB
- `tests/test_experiment_suggester.py` — 10 new tests for Phases 1-2
- `benchmarks/benchmark_closed_loop.py` — uses expanded pool

## References
- ADR-2026-08-10-03-acquisition-negative-result.md
- ADR-2026-08-11-solution-phase-ea.md
- ADR-2026-08-07-04-radius-of-gyration-profiling.md
