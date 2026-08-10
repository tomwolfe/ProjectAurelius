# ADR-2026-08-10: Integrated release — what was wrong, what changed, what's proven

## TL;DR

The four largest structural defects in Project Aurelius were not in the science
layer but in the plumbing: a reversed substructure call, a binary statistic
that saturated at 1.000, a three-valued template score, a topology-blind
precursor lookup, and a constant-0 direct-match confidence. Together they made
the entire synthesizability signal exactly constant on realistic populations —
every one of the 15 molecules in `discoveries.sdf` scored **0.7731**. The
reduction axis was worse: the descriptor being ranked had a Spearman ρ of
**+0.34 against a 0.31 permutation-control bar**, i.e. essentially no
information, and no amount of label cleaning was going to help because the
problem was the *observable*, not the labels.

Both axes are now validated against clean experimental data and drive
selection. Measured before and after:

| what | before | after | evidence |
|------|--------|-------|----------|
| reduction axis (ρ vs measured gas-phase EA) | +0.34 (noise floor) | **+0.91** | 40 directly measured EAs, permutation p |
| reduction MAE (eV) | 0.68 | **0.29** | same |
| reduction SEI ordering (FEC>VC>EC>DMC>>DME) | wrong | **correct** | chemistry sanity test |
| grounding std on 51 known electrolytes | 0.217 | 0.192 | |
| distinct grounding values / 51 | 7 | **40** | |
| distinct grounding values / 15 discovered | 1 | **12** | |
| distinct template values / 51 | 2 | **28** | |
| known − Frankenstein grounding gap | +0.311 | **+0.527** | adversarial set, 0/12 escape |
| Frankensteins passing 0.75 report gate | 3/12 | **0/12** | |
| conductivity proxy pinned at ceiling | 15/51 | **0/51** | saturating map, monotone |
| ΔSCF cost | 114 ms/mol | **61 ms/mol** | parallelised, 2.0× |
| biggest scoring weight (0.23) ranked by | noise-floor LUMO | **validated EA** | wired into pipeline |

**Honest residual gaps (not papered over):**
- Acquisition: the new `expected_impact` term improves holdout ranking but does
  **not** beat random (best p = 0.111; ceiling 0.066 eV). Open capability gap.
- Retrosynthetic depth is still coarse (values 1, 2, 5).
- Solution-phase reduction potentials uncalibrated; the EA set is gas-phase.
- The confound audit still flags 6 shipped targets; LUMO ranking remains
  unavailable.

## What was released

Four ADRs, six files changed in the science layer, four new modules/tests:
- `docs/adr/ADR-2026-08-10-dscf-electron-affinity.md` (reduction axis)
- `docs/adr/ADR-2026-08-10-02-grounding-degeneracy.md` (synthesizability)
- `docs/adr/ADR-2026-08-10-03-acquisition-negative-result.md` (closed loop)
- `docs/adr/ADR-2026-08-10-04-conductivity-saturation.md` (Walden proxy)
- `src/aurelius/scoring/oracle/reduction.py` — ΔSCF electron-affinity oracle
- `src/aurelius/data/experimental_electron_affinity.json` — 40 clean EAs
- `benchmarks/benchmark_reduction_axis.py` — permutation-controlled ρ/MAE
- `tests/test_reduction_oracle.py`, `test_grounding_non_degenerate.py`,
  `test_conductivity_saturation.py`

**Reproducibility.** Every claim is backed by a benchmark or test that a fresh
checkout can run: `python benchmarks/benchmark_reduction_axis.py`,
`pytest tests/test_reduction_oracle.py tests/test_grounding_non_degenerate.py
tests/test_conductivity_saturation.py`.

## Recommended next experiments

**Computational (no wet lab):**
1. Re-run `benchmark_closed_loop.py` against the experimental EA target
   instead of the confounded HOMO set, so any ranking gain is interpretable.
2. Enlarge the candidate pool well beyond the calibration file; acquisition
   cannot distinguish itself from random when the pool is 61 molecules and the
   budget is 20.
3. Batch-aware acquisition (joint mutual information / Fisher-information
   subset selection) — the measured ceiling is a property of *subsets*, not
   individual molecules, and greedy scoring captured none of it.

**Wet lab (what to actually make):**
4. The screening run produced a shortlist where EC/PC/FEC/DTD and the
   glutaronitrile/succinonitrile families now rank at the top. Pick the 5–10
   molecules in `docs/` → validate with impedance spectroscopy. The validated
   EA axis means the reduction-stability ranking these are built on now tracks
   real chemistry — which was not true before.
5. Measure reduction potentials for 5–10 of these in the actual electrolyte
   solvent (not gas phase). That single dataset would replace the gas-phase EA
   calibration and close the largest remaining gap.
