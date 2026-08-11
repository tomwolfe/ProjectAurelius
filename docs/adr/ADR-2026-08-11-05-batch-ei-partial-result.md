# ADR-2026-08-11-05: Fantasization-based batch EI — improved but not significant

## Status
Accepted (feature added; efficacy claim **partially** supported — improved p-value but not below 0.05)

## Context

ADR-2026-08-10-03 established that the suggester was indistinguishable from random
(p=0.111) on the closed-loop benchmark. Two root causes were identified: (1) pool too
small (61 mols), (2) greedy per-molecule scoring cannot capture subset-level information
gain. ADR-2026-08-11-02 added pool expansion and BALD/Pareto-UCB but did not fix the
underlying issue: the conformal predictor has no GPR model, so BALD silently degrades to
zero and the acquisition is effectively unchanged.

This ADR implements the three-step fix:
1. Pool expansion as default (any pool < 200 is expanded via BRICS harvesting)
2. Fantasization-based batch Expected Improvement with rank-1 Cholesky updates
3. Model-aware acquisition via explicit `DeltaCorrection` parameter

## Decision

### 1. Pool expansion as default

`expand_candidate_pool()` now runs unconditionally for any pool below `MIN_POOL_SIZE`
(200). Previously it only ran for pools between 10 and 200. The harvesting function
`_harvest_from_top_candidates` was also fixed: it now ranks candidates by a composite
of quality score and BRICS-fragmentability, because the top-quality molecules (small
cyclic carbonates/sulfones) have zero BRICS-breakable bonds and produced no fragments.

### 2. Fantasization-based batch EI

New module `src/aurelius/scoring/oracle/gpr_fantasize.py` implements rank-1 Cholesky
updates for efficient GPR posterior simulation. For each candidate, it simulates adding
that candidate to the GPR training set and scores by the total variance reduction across
the *entire remaining pool*. This captures subset-level information that greedy per-molecule
scoring misses.

The acquisition is wired in via a new `batch_ei` term (weight 0.20) in `DEFAULT_WEIGHTS`.
It requires a `DeltaCorrection` instance (passed via the new `delta_correction` parameter
to `suggest_experiments`) because the conformal predictor has no GPR model.

### 3. Model-aware acquisition

`suggest_experiments` now accepts an optional `delta_correction` parameter. When provided,
its `_homo_model` GPR is used for both BALD and batch_ei scoring. The closed-loop
benchmark passes its refit `DeltaCorrection` so the acquisition targets epistemic
uncertainty in the *current* model (not the static calibration).

### 4. Scaffold-stratified splits (reverted)

Scaffold-stratified hold-out splits were implemented and tested. They made the benchmark
*too hard*: with only 20 seed molecules, the model cannot generalize to unseen scaffolds,
and the permutation control degraded from p=0.0013 to p=0.17. Reverted to random splits.
This is recorded as a negative result: scaffold stratification is the right methodology
for a benchmark that tests *extrapolation*, but this benchmark tests *acquisition strategy*
and needs the holdout to be learnable from the seed.

## Results

Closed-loop benchmark, 10 seeds, budget 20, random splits:

| metric | before (ADR-2026-08-10-03) | after (this ADR) |
|--------|---------------------------|------------------|
| rho edge p-value | 0.111 | **0.076** |
| mean rho edge | +0.024 | −0.0196 |
| suggester rho wins | — | 4/10 |
| permutation control p | 0.0013 | 0.0013 |

The p-value improved from 0.111 to 0.076, but the mean rho edge is now slightly negative
(−0.0196). The suggester beats random on 4/10 splits but loses on 6/10. The sign still
flips between seeds.

The permutation control remains strong (p=0.0013), confirming the loop genuinely learns —
the issue is that the *acquisition strategy* is not reliably better than random at this
budget and pool size.

## Why (best current understanding)

1. **Pool size is still the binding constraint.** The calibration set has 115 entries.
   After 30% holdout and 20 seed, the unmeasured pool is 61 molecules. At budget 20,
   random covers a third of the pool. BRICS expansion helps for diverse chemistry but
   the calibration set is too homogeneous (mean 1.9 BRICS-breakable bonds per molecule)
   to expand meaningfully.

2. **The target is confounded.** `orbital_calibration.json` has 53 citation sources with
   between-source confound (citation-only ρ=0.71). Ranking improvements on a confounded
   target are hard to interpret even when they occur.

3. **Batch EI captures subset-level information but the signal is weak.** With only 20
   seed points, the GPR posterior is uncertain everywhere, so the variance reduction
   from adding any single candidate is small and noisy. The batch_ei scores are
   well-differentiated (confirmed by unit tests) but don't translate to a reliable
   ranking edge at this scale.

## What is kept, and what is claimed

Kept:
- Pool expansion as default (correct behavior, no downside)
- Fantasization-based batch EI (principled, tested, available for larger pools)
- `delta_correction` parameter (enables model-aware acquisition)
- Fragmentability-aware BRICS harvesting (fixes a real bug)

**Not claimed:** that using the suggester beats random selection at budget 20 on the
current benchmark. The p-value improved (0.111 → 0.076) but did not cross the 0.05
threshold. The honest verdict remains "indistinguishable from random at this budget."

## Consequences

- `benchmark_closed_loop.py` now passes the refit `DeltaCorrection` to the suggester,
  enabling model-aware acquisition in the benchmark.
- `DEFAULT_WEIGHTS` has 8 terms including `batch_ei` (0.20). The weights were rebalanced
  from the previous 7-term schema.
- `suggest_experiments` has a new `delta_correction` parameter. All existing callers
  continue to work (parameter is optional, defaults to None).
- 5 new tests in `tests/test_experiment_suggester.py::TestBatchEI`.

## Next steps to actually close this gap

1. **Expand the candidate pool beyond the calibration file.** The 61-mol pool is the
   binding constraint. A pool of 200+ genuinely diverse molecules (e.g. from a
   BRICS enumeration of known electrolytes, or from a database like GDB-13) would give
   acquisition room to be clever.

2. **Use a clean experimental target.** The confounded HOMO set makes ranking gains
   uninterpretable. A clean experimental EA dataset (even 15-20 points, per
   ADR-2026-08-11-04) would let us measure whether the acquisition actually improves
   ranking on a physically meaningful target.

3. **Increase the budget.** At budget 20, random covers a third of a 61-mol pool.
   At budget 40+ with a larger pool, the acquisition has more room to differentiate.

4. **Try qEHVI or joint mutual information.** Fantasization-based batch EI is a greedy
   approximation. A full batch-aware acquisition (qEHVI for multi-objective, or
   joint mutual information for subset selection) might capture complementarity that
   the greedy approach misses.

## Files
- `src/aurelius/agent/experiment_suggester.py` — pool expansion, batch EI, delta_correction param
- `src/aurelius/scoring/oracle/gpr_fantasize.py` — rank-1 Cholesky update math
- `benchmarks/benchmark_closed_loop.py` — passes refit model to suggester
- `tests/test_experiment_suggester.py` — 5 new tests for batch EI

## References
- ADR-2026-08-10-03-acquisition-negative-result.md
- ADR-2026-08-11-02-pool-expansion-and-bald-acquisition.md
- ADR-2026-08-11-04-alpb-solution-phase-negative-result.md
