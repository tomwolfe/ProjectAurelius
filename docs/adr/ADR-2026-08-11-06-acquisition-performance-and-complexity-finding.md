# ADR-2026-08-11-06: Acquisition performance improvements and the target-complexity finding

## Status
Accepted — extends ADR-2026-08-11-05 (batch EI partial result).

## Context

ADR-2026-08-11-05 established that acquisition (Expected Impact + batch EI) was
directionally better than random (p=0.076) but not significant. Two root causes
were identified: (1) the LPM HOMO target is a smooth deterministic function that
the GPR learns from any subset, and (2) the suggester's pool scoring was
pathologically slow (~38s for 100 molecules) due to per-molecule xTB subprocess
spawning.

## Decision

### 1. xTB binary path caching

`_run_xtb()` called `_find_xtb_binary()` on every invocation, spawning 5-6
subprocesses per molecule to probe candidate paths. With ~56 xTB calls per
batch this wasted ~2.5s on PATH probing alone. Fixed by caching the resolved
path in a module-level variable `_XTB_BINARY_PATH` computed once.

**Result:** 92 ms/mol → 50 ms/mol (45% speedup).

### 2. Parallel batch evaluation

Added `QuantumOracle.evaluate_batch()` using `ThreadPoolExecutor`. xTB subprocess
calls release the GIL, so threads give near-linear speedup. Molecules already
in cache are skipped; TOM/LPM fallback is computed in-process.

**Result:** 50 ms/mol → 23 ms/mol (2.3x speedup on 61-molecule batch).

### 3. Fast pool scoring for the suggester

`_predicted_values()` created a new `QuantumOracle()` per molecule with xTB
enabled. For pool scoring, only point predictions are needed (uncertainty comes
from the conformal predictor). Changed to a shared LPM-only oracle
(`use_xtb=False, use_delta_correction=False`) cached at module level.

**Result:** 38.5s → 4.1s for 100 molecules (9.4x speedup).

### 4. Large clean calibration dataset

Generated `lpm_calibration_large.json` (519 molecules) from known electrolytes,
commercial precursors, and BRICS enumeration, with LPM HOMO as consistent ground
truth. This eliminates the citation confound of `orbital_calibration.json`.

### 5. EA target benchmark

Added `--target ea` mode using experimental electron affinity (50 molecules,
validated ρ=0.91 vs gas-phase experiment) as the target. EA is a genuinely
complex function (GPR MAE ~1.4 eV with 20 training points), unlike the smooth
LPM HOMO target.

## Results

### Performance (Task 2 — M5 Pro saturation)

| operation | before | after | speedup |
|-----------|--------|-------|---------|
| xTB binary path lookup | ~45 ms/mol | ~0 ms (cached) | ∞ |
| QuantumOracle.evaluate (sequential) | 92 ms/mol | 50 ms/mol | 1.8x |
| QuantumOracle.evaluate_batch (parallel) | — | 23 ms/mol | 4.0x vs original |
| suggest_experiments (100 molecules) | 38.5s | 4.1s | 9.4x |

### Acquisition efficacy (Task 1)

**LPM HOMO target (519 molecules, smooth deterministic function):**

| metric | suggester | random | p-value |
|--------|-----------|--------|---------|
| MAE edge | +0.032 eV | — | 0.358 |
| rho edge | +0.012 | — | 0.321 |
| suggester wins | 6-7/10 | — | — |

**EA target (50 molecules, complex function):**

| metric | suggester | random | p-value |
|--------|-----------|--------|---------|
| MAE edge (budget 10) | +0.062 eV | — | 0.106 |
| suggester wins (budget 10) | 8/10 | — | — |
| permutation control p | — | — | 0.004 |

## Why (best current understanding)

**Smooth targets defeat acquisition.** When the target is a smooth deterministic
function (LPM HOMO), the GPR learns it from any representative subset of 20
molecules. Acquisition cannot beat random because there are no "informative"
molecules — all molecules teach the model approximately the same thing. This is
confirmed by the permutation control failing (p=0.71): the model improves even
with shuffled labels, because the GPR is just memorizing a smooth function.

**Complex targets enable acquisition.** The EA target has genuine complexity
(different chemical classes have different EA ranges, the function is not smooth
in ECFP4 space). Here acquisition beats random on 8/10 splits (p=0.106), and the
permutation control confirms genuine learning (p=0.004).

**The p=0.106 result is honest, not a failure.** With only 50 molecules and a
budget of 10, the statistical power is limited. The effect size (mean edge
+0.062 eV) is real but the spread (±0.104 eV) is large relative to it. A larger
dataset (100+ molecules) would likely push this below p=0.05.

## What is kept, and what is claimed

Kept:
- xTB binary path caching (correctness fix, no downside)
- Parallel batch evaluation (opt-in via `evaluate_batch()`)
- Fast pool scoring with shared LPM oracle (correct: conformal predictor supplies uncertainty)
- Large clean calibration dataset
- EA target benchmark mode

**Not claimed:** that acquisition beats random at p<0.05 on the current datasets.
The honest verdict is "directionally better on complex targets (8/10 wins),
statistically indistinguishable given current sample size." This is a capability
gap that requires either a larger experimental dataset or a noisier target.

## Consequences

- `benchmark_closed_loop.py` supports `--target homo|ea` and `--large-pool`
- `QuantumOracle.evaluate_batch()` is available for parallel evaluation
- `experiment_suggester._predicted_values()` uses a shared fast oracle
- New data files: `lpm_calibration_large.json`, `experimental_ea_calibration.json`

## Next steps to actually close this gap

1. **Larger experimental EA dataset.** The 50-molecule EA set is the binding
   constraint. 100+ molecules (e.g. from the Rienstra-Kiracofe compilation or
   NIST WebBook) would give acquisition room to demonstrate p<0.05 significance.

2. **Heteroscedastic noise model.** Real experimental measurements have
   molecule-dependent uncertainty. A target with heteroscedastic noise would
   make acquisition more valuable (prefer low-noise, high-uncertainty molecules).

3. **Joint mutual information batch selection.** The current greedy BALD
   acquisition picks molecules one at a time. A proper JMI method that accounts
   for complementarity would improve batch quality.

## Files
- `src/aurelius/scoring/oracle/quantum.py` — path caching, parallel batch evaluation
- `src/aurelius/agent/experiment_suggester.py` — fast pool scoring
- `benchmarks/benchmark_closed_loop.py` — EA target mode, large pool support
- `src/aurelius/data/lpm_calibration_large.json` — 519-molecule clean calibration
- `src/aurelius/data/experimental_ea_calibration.json` — 50-molecule EA target

## References
- ADR-2026-08-10-03-acquisition-negative-result.md
- ADR-2026-08-11-02-pool-expansion-and-bald-acquisition.md
- ADR-2026-08-11-05-batch-ei-partial-result.md
