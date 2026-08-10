# ADR-2026-08-10-02: Synthesizability grounding was constant, not weak

## Status
Accepted

## Context

The project notes described the synthesizability signal as "too weak" and
"depth often constant". Measurement showed something stronger than weak: on the
15 molecules in `discoveries.sdf` the grounding score was **exactly 0.7731 for
every single one** — standard deviation 0.000, one distinct value. Across the
51 known electrolytes it took only 7 distinct values.

A constant cannot rank. Synthesizability was wired into the scalar score, the
tournament weighting and the NSGA-II objectives, and in all three places it was
contributing precisely zero selection pressure while appearing to work.

Three independent defects produced this, each masking the others.

### Defect 1 — reversed substructure arguments

```python
if mol.HasSubstructMatch(bb):
    overlap = len(bb.GetSubstructMatch(mol))   # wrong direction
```

`bb.GetSubstructMatch(mol)` asks where the *whole molecule* sits inside the
precursor. Whenever the precursor is the smaller species — the normal case —
this returns `()`, so `overlap` was 0 and `direct_conf` was 0.000 for
essentially every candidate ever scored. Route confidence then degenerated to
`0.5 * depth_conf * brics_cov`, which is why so many molecules landed on the
same number.

### Defect 2 — binary coverage that everything passes

`_cached_coverage` computed the *fraction of BRICS fragments* passing
`_is_known_bb_precursor`, a binary test with a 50%-of-heavy-atoms threshold.
Small fragments of electrolyte-like molecules essentially always pass, so the
statistic sat at exactly 1.000 across the search space. A continuous,
heavy-atom-weighted, rarity-penalised version already existed in
`retrosynthetic._precursor_coverage` and was simply not being used by the
scoring path.

### Defect 3 — three-valued template score

`compute_synthesis_feasibility` returned 0.9 / 0.5 / 0.1 depending on whether
*any* core or functional template matched. Since virtually every candidate
contains an ether or a carbonyl, 96% of realistic molecules scored 0.9.

### Defect 4 (found while fixing 1–3) — topology-blind precursor lookup

With direct matching repaired, the strained triepoxide `C1OC1C1OC1C1OC1` scored
depth 1 and direct confidence 1.00, because plain `HasSubstructMatch` let the
*linear* precursor triglyme match it at 100% atom coverage. A three-membered
ring is not "an ether you can buy".

## Decision

1. Match substructures in the correct direction, and treat "molecule is a piece
   of a larger precursor" as weaker evidence (half weight) than "a precursor
   accounts for most of this molecule".
2. Delegate `_cached_coverage` to the existing continuous coverage function.
3. Grade template feasibility on *heavy-atom coverage* by recognised motifs plus
   a saturating *motif-diversity* term, instead of a three-way branch.
4. Add an explicit multiplicative `infeasibility_penalty` for motifs that defeat
   a plausible synthesis or an electrolyte application — peroxides, azides,
   silyl ethers, boroxines, polynitrile carbons, strained oxygen rings,
   anhydrides. Penalties compound.
5. Use `Chem.AdjustQueryProperties` with `adjustRingChain` for every precursor
   lookup so ring atoms only match ring atoms.
6. Add 38 stocked cyclic reagents (VC, TMC, FEC, sultones, DTD, lactones,
   dioxolane, THF, glymes, chloroformates) that the 223-entry precursor
   database was missing. Once ring-aware matching was enforced, their absence
   became visible as false negatives on real electrolytes such as vinylene
   carbonate and ethylene sulfate.

## Results

Measured on the 51 known electrolytes vs a 12-molecule adversarial set:

| metric                                   | before | after |
|------------------------------------------|-------:|------:|
| grounding std (known electrolytes)       |  0.217 | 0.192 |
| distinct grounding values / 51           |      7 |    40 |
| distinct grounding values (discoveries)  |      1 |    12 |
| distinct template values / 51            |      2 |    28 |
| known mean − Frankenstein mean           | +0.311 | **+0.527** |
| Frankensteins above known 25th pct       |   1/12 | **0/12** |
| Frankensteins passing the 0.75 report gate | 3/12 | **0/12** |

The `combined_grounding_score >= 0.75` wet-lab handoff gate is unchanged and
remains correctly positioned: known electrolytes now average 0.780, adversarial
structures 0.253.

## Honest limitations

- Retrosynthetic depth is still coarse (values 1, 2, 5) and remains constant at
  2 across the current `discoveries.sdf`. That particular constancy is
  legitimate — all 15 molecules are near-identical cyanomethyl ethers — but the
  three-valued output is genuinely low-resolution and is the next thing to
  improve if depth is to carry ranking weight.
- The infeasibility motif list is hand-curated and electrolyte-specific. It
  encodes chemical judgement, not a learned model, and will need extending as
  the search moves into new chemical space.
- Grounding is still a structural heuristic, not a retrosynthesis search. It
  answers "is this built from purchasable pieces by precedented chemistry",
  not "here is the route".

## Files
- `src/aurelius/agent/mutation/brics.py`
- `src/aurelius/agent/mutation/retrosynthetic.py`
- `src/aurelius/data/commercial_precursors.json` (223 → 261 entries)
- `tests/test_grounding_non_degenerate.py`
