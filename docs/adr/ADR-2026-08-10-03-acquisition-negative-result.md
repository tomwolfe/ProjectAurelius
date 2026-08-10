# ADR-2026-08-10-03: Expected-impact acquisition, and an honest negative result

## Status
Accepted (feature added; efficacy claim **not** supported)

## Context

The closed loop demonstrably works: ingesting measurements improves holdout MAE
by ~0.13–0.15 eV, and the permutation control confirms the gain requires correct
labels (correct labels win on 9/10 splits, one-sided p = 0.0013). That part is
real learning, not recalibration.

What was never shown is that the *suggester* is better than picking molecules at
random. The four existing acquisition terms (uncertainty, novelty, DoA
proximity, bias) are all model-centric: they maximise information about the
oracle while being indifferent to whether that information changes which
molecules get made.

## Decision

Add a fifth term, `expected_impact`: the probability that a measurement moves
the molecule across the current top-k decision boundary. The conformal interval
is treated as a 90% predictive interval, approximated by a Gaussian with
matching coverage, and scored as `2·min(P(y>t), P(y≤t))` — maximal when the
prediction sits exactly on the shortlist cut-off, decaying to zero when the
molecule is unambiguously in or out.

This required restructuring `suggest_experiments` into two passes so the
decision thresholds reflect the whole candidate pool rather than one molecule.

## Results — the term helps ranking, and nothing beats random

Ablation over 10 frozen splits, budget 20, holdout never trained on:

| acquisition strategy | ΔMAE (eV) | Δρ |
|----------------------|----------:|-----:|
| suggester, no expected_impact |  −0.166 | −0.013 |
| suggester, with expected_impact (default) | −0.123 | **+0.024** |
| expected_impact only | −0.134 | **+0.043** |
| uncertainty only     | −0.173 | −0.017 |
| novelty only         | −0.101 | +0.013 |
| **random**           | −0.148 | +0.027 |

The new term does what it was designed to do — it is the only term that moves
holdout *ranking* in the right direction, and it converts the suggester's Δρ
from negative to positive. But the honest headline is the last row: **random
acquisition is statistically indistinguishable from every strategy tried.**

Across budgets 5/10/20/40 the suggester-vs-random edge never approaches
significance (best p = 0.111) and the sign flips between budgets.

Two further strategies were tried and also failed:

| strategy | MAE edge vs random | p |
|----------|-------------------:|--:|
| GPR posterior variance (model-aware uncertainty) | −0.011 | 0.507 |
| Max-min Tanimoto diversity from the calibration set | +0.008 | 0.619 |

### There is real headroom, and nothing captures it

To rule out "the problem is saturated", the ceiling was measured directly: for
each split, 12 random subsets of size 20 were drawn and the best compared to the
mean.

    mean achievable gain from perfect subset choice vs average random: 0.0665 eV

So a 0.066 eV prize exists and every acquisition function tested captures
approximately none of it. This is a genuine capability gap, not a saturated
benchmark, and it is recorded as such rather than papered over.

## Why (best current understanding)

- The pool is 61 molecules from a 115-molecule calibration file. At budget 20 a
  random draw already covers a third of the pool, so there is little room for
  cleverness to distinguish itself.
- The refit target (`orbital_calibration.json` HOMO) is provenance-confounded
  (53 sources, citation-only ρ = 0.71). Ranking improvements on a confounded
  target are hard to interpret even when they occur — this benchmark's ρ column
  should be read with the same caution as everywhere else in the project.
- The suggester's uncertainty term uses the *conformal* interval, which is
  molecule-agnostic for most properties, whereas the quantity that determines
  refit benefit is the *GPR posterior* on the specific residual being learned.
  Testing the GPR posterior directly (above) did not help either, which weakens
  this explanation but does not eliminate it.

## What is kept, and what is claimed

Kept: the `expected_impact` term, because it is principled, it is the only term
that improves holdout ranking, and it makes the suggester's output more
decision-relevant and better diversified (batch redundancy 0.064 vs 0.074 for
random).

**Not claimed:** that using the suggester beats random selection. The benchmark
verdict string now reports ranking significance explicitly and prints
"indistinguishable from random at this budget" when that is what the data says,
which is currently the case.

## Consequences

- `benchmark_closed_loop.py` now reports Δρ edge and a paired t-test alongside
  ΔMAE, so a ranking claim can never again rest on MAE alone.
- `DEFAULT_WEIGHTS` gained a key. Any test overriding weights must zero *all*
  terms explicitly; one existing test was silently under-specified and correctly
  failed when the new term defaulted in.

## Next steps to actually close this gap
1. Re-run acquisition against the clean experimental EA target (ADR-2026-08-10)
   instead of the confounded HOMO set, so ranking gains are interpretable.
2. Enlarge the candidate pool well beyond the calibration file, so acquisition
   has somewhere to be clever.
3. Try batch-aware acquisition (joint mutual information / Fisher-information
   subset selection) rather than greedy per-molecule scoring, since the measured
   ceiling is a property of *subsets*, not of individual molecules.

## Files
- `src/aurelius/agent/experiment_suggester.py`
- `benchmarks/benchmark_closed_loop.py`
- `tests/test_experiment_suggester.py`
