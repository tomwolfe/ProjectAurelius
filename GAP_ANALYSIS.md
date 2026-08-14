# GAP_ANALYSIS.md — Project Aurelius Commercial Readiness

## Goals

Close the remaining gaps between the current v12.1 codebase and **commercial-grade
electrolyte discovery readiness**: physically-grounded property prediction,
closed-loop learning, discovery that recovers known electrolytes, and a
wet-lab handoff pipeline that a working chemist can act on.

## Current State

### Oracle Accuracy
| Property | Metric | Value | Status |
|---|---|---|---|
| Dielectric ε (verified, n=55) | MAE / ρ | 3.26 / 0.934 | OK |
| Dielectric ε (commercial, n=10) | MAE / ρ | 1.80 / 0.988 | OK |
| Viscosity | ρ (in-domain) | 0.551 | OK |
| Donor Number | ρ (external) | 0.189 | **WEAK** |
| LUMO (unseen) | ρ | 0.061 | No ranking claim (replaced by ΔSCF EA) |
| HOMO (NIST, n=88, LPM) | ρ / MAE | 0.91 / 0.38 eV | OK |
| **ΔSCF EA (unseen, n=40)** | ρ / MAE | **0.912 / 0.289 eV** | OK (new) |
| **HOMO (unseen, xTB-calibrated)** | ρ / MAE | 0.327 / 0.539 eV | **WEAK** |

### Discovery Metrics
| Metric | Value | Target | Status |
|---|---|---|---|
| Rediscovery rate (seeded-exact) | 68.75% | ≥50% | OK |
| Novel scaffold ratio | 50.0% | ≥80% | **FAIL** |
| Score gap (top vs known mean) | +27.72 | — | Informational |

The `unified_benchmark.md` reports overall status: **FAIL** — the single
failing CI criterion is `novel_scaffold_ratio=0.500 < 0.8`.

### Benchmark Failures
1. **Novel scaffold ratio**: 50% vs 80% target. The search is rediscovering
   known scaffolds half the time instead of hopping to novel ones.
2. **HOMO on unseen molecules**: ρ = 0.327 with xTB-calibrated LPM. Strong
   in-distribution (0.94 on NIST) but degrades on the external benchmark.
3. **Donor Number**: ρ = 0.189 on external validation — essentially at the
   noise floor despite GC fragment-additivity.
4. **Oracle vs ML (HOMO)**: Oracle ρ = 0.110 vs RF ρ = 0.387 — the physics
   model loses to a fingerprint regressor on HOMO ranking.
5. **Acquisition strategy edge**: Random acquisition is statistically
   indistinguishable from the active suggester (best p = 0.111); the
   suggester's MAE edge never approaches significance across budgets 5/10/20/40.

### Performance
- GC batch throughput: 6,147 mol/s (CPU-bound SMARTS loop, documented bottleneck)
- Oracle batch: 10.7 mol/s (dominated by xTB ΔSCF, not GC)
- TOM 302 mol/s, xTB 15 mol/s serial → 42 mol/s at 8 threads

---

## Gap 1: Novel scaffold ratio fails CI (50% vs 80% target)

### Evidence
`unified_benchmark.md` line 71: `novel_scaffold_ratio=0.500 < 0.8` — the
only failing CI assertion. From `unified_benchmark.json`:
discovery block reports `novel_scaffold_ratio: 0.5`,
`n_novel_scaffolds: 5`, `n_top_scaffolds: 10`.

`loop.py:482` calls `_inject_tier0_seeds()` on scaffold stagnation (2 batches),
but the pool may be dominated by the known electrolyte set that seeds the
mutation engine (`MutationEngine.__init__` loads
`known_electrolytes.json`). The `_generate_candidates` method (loop.py:467-511)
produces both single and mixture candidates, but the mutation rate / seed pool
composition may bias toward known scaffolds.

### Required Change
**`src/aurelius/agent/mutation/engine.py`**: Increase the diversity pressure
in `mutate_batch` — specifically, ensure the novelty gate (`_generate_candidates`
line 501-511) actively rejects molecules whose Murcko scaffold matches a known
electrolyte scaffold, pushing the search toward genuinely novel cores.

**Regression test**: `tests/test_loop.py` — add or extend a test that asserts
the discovery loop's `novel_scaffold_ratio` meets the 0.8 threshold on a
fixed-seed short run. If a full loop test is too slow, add a targeted test
in `tests/test_discovery_smoke.py` that runs 3 generations with a controlled
pool and asserts `novel_scaffold_ratio >= 0.8`.

### Files
- `src/aurelius/agent/mutation/engine.py`
- `tests/test_loop.py` (or `tests/test_discovery_smoke.py`)

---

## Gap 2: Acquisition strategy not better than random

### Evidence
`closed_loop_ea_large_decision.json` (`acquisition_by_budget.40`): over 10 seeds,
`mean_rho_edge = 0.021 ± 0.078`, `rho_edge_p_value = 0.44` (not significant).
`mean_edge = -0.0020` (p = 0.998) for MAE. The suggester's only statistically
significant signal is TKE (top-k enrichment, p = 0.043, Cohen's d = 0.95) but
even that is weak.

The `experiment_suggester.py` (lines 100-200) computes five acquisition terms:
uncertainty, bias, novelty, doa_proximity, expected_impact. The batch
diversity penalty was added (ADR-2026-08-08-06) but the pool expansion threshold
(`MIN_POOL_SIZE = 200`, line 100) may be the binding constraint — the
closed-loop benchmark uses a pool of 38-167 molecules, which is below the 200
minimum that triggers BRICS harvesting expansion. When the pool is too small,
diversity-based acquisition cannot outperform random.

### Required Change
**`src/aurelius/agent/experiment_suggester.py`**: Lower or make
`MIN_POOL_SIZE` adaptive to the available calibration data. For small pools
(<200), fall back to uncertainty-dominated acquisition rather than requiring
full diversity-based scoring. Alternatively, dynamically scale `MIN_POOL_SIZE`
based on the pool-to-calibration ratio.

Additionally, the `expected_impact` term (lines 100-200, ADR-2026-08-10-03)
should be boosted: the README admits "random is statistically indistinguishable
from every strategy tried." The `expected_impact` term is the only one that
moves ranking in the right direction (Δρ +0.024 to +0.043), so it needs a
higher weight relative to the model-centric terms.

**Regression test**: `tests/test_experiment_suggester.py` — add a test that
asserts the suggester beats random acquisition on a frozen 10-seed closed-loop
benchmark with ρ_edge_p_value < 0.05, or at minimum asserts the suggester's
mean_tke_edge (top-k enrichment) is significant at p < 0.05.

### Files
- `src/aurelius/agent/experiment_suggester.py`
- `tests/test_experiment_suggester.py`

---

## Gap 3: HOMO oracle loses to ML on unseen molecules

### Evidence
`oracle_absolute_audit.json`:
- HOMO: `oracle_rho = 0.2581`, `rf_rho = 0.3939`, `oracle_wins = false`
- The top-5 outliers are all `failure_mode: "high_hetero_density"` — fluorinated
  molecules with many heteroatoms (TFSI-H, octafluoropolyether sulfone, etc.)
  where the affine calibration (ADR-2026-08-09-02) is extrapolating.

The README (line 157) confirms the LPM still beats real QM on NIST IPs
(ρ=0.94 vs ρ=0.875) but the xTB-calibrated LUMO only reaches ρ=0.327 on unseen.
The `quantum.py` `_XTB_HOMO_RE` parser and affine calibration are correct
(`test_external_validation_lumo` was fixed to test the unseen split), so the
gap is in the model's extrapolation behavior, not the plumbing.

### Required Change
**`src/aurelius/scoring/oracle/quantum.py`**: Add a domain-of-applicability
penalty for high-heteroatom-density molecules. The `compute_quantum_domain_penalty`
function exists (oracle/__init__.py:64) but is not being applied to the HOMO
prediction path for xTB. When the molecule contains ≥6 fluorine atoms or
high heteroatom density, reduce the quantum_confidence or apply a confidence
penalty so the scoring pipeline down-weights unreliable HOMO predictions.

Alternatively, add a heteroatom-density feature to the conformal predictor's
difficulty function (already enhanced with fluoride/ring features in commit
5637998 — but this change was not applied to the quantum oracle path).

**Regression test**: `tests/test_net_progress.py` — add a test that asserts
high-heteroatom-density molecules (TFSI, perfluoroethers) receive a lower
`quantum_confidence` or a domain penalty, preventing them from outscoring
known-good solvents on the basis of unreliable HOMO predictions.

### Files
- `src/aurelius/scoring/oracle/quantum.py`
- `tests/test_net_progress.py` (or `tests/test_oracle_holdout.py`)

---

## Task Specs

Each task is independent (disjoint files) and can be run in a parallel
worktree:

1. **close-gap1**: Increase novelty pressure in mutation engine → fix
   `novel_scaffold_ratio`. Files: `src/aurelius/agent/mutation/engine.py` +
   regression test in `tests/test_loop.py`.
2. **close-gap2**: Make acquisition beat random on small pools → lower/adjust
   `MIN_POOL_SIZE` and boost `expected_impact` weight.
   Files: `src/aurelius/agent/experiment_suggester.py` +
   regression test in `tests/test_experiment_suggester.py`.
3. **close-gap3**: HOMO DoA penalty for high-heteroatom-density → prevent
   fluorinated outliers from gaming the score.
   Files: `src/aurelius/scoring/oracle/quantum.py` +
   regression test in `tests/test_net_progress.py`.
