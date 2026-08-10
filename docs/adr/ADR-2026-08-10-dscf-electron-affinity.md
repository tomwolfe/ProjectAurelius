# ADR-2026-08-10: Reduction axis moves from LUMO to ΔSCF electron affinity

## Status
Accepted — supersedes the *ranking* role of ADR-2026-08-09 and ADR-2026-08-09-02
(those remain in force for LUMO *calibration*/MAE only).

## Context

Two previous attempts to make the reduction axis usable both failed, and they
failed for the same underlying reason, which neither ADR identified.

**Attempt 1 (ADR-2026-08-09).** Train a Δ-learning correction on the DFT LUMO
labels in `orbital_calibration.json`. Abandoned for ranking: the labels come
from 53 different functional/basis-set combinations, so a predictor given only
the citation string reaches ρ = 0.68. Unseen ρ = 0.061.

**Attempt 2 (ADR-2026-08-09-02).** Replace those labels with 231 internally
consistent GFN2-xTB LUMO values. The confound audit went clean (citation
ρ = 0.000), but the unseen external ρ stayed at **0.061 → 0.097**. The audit
was satisfied and the science was not.

Attempt 2 also introduced a subtler problem: the target is now *self-generated*.
`LumoProxy` learns to reproduce the pipeline's own xTB backend, with no
experimental anchor anywhere in the loop. A perfect score on that target means
the GPR has memorised xTB, not that the molecule is hard to reduce.

### Root cause

The failure is not primarily a label-provenance problem. It is an **observable**
problem: for the saturated, closed-shell solvents this project searches over,
the frozen-orbital LUMO is not a physically meaningful reduction descriptor.

Koopmans' theorem for the *virtual* space is far weaker than for the occupied
space. The occupied analogue works well here — LPM reaches ρ = 0.94 against 88
NIST ionisation energies — because ionisation removes an electron from a bound,
well-defined lone pair. The virtual orbital of an alkyl carbonate or an ether is
not a bound anion state at all: it is a discretised continuum function whose
energy is determined by the basis set, not by the chemistry. Ranking molecules
by such an orbital energy is close to ranking them by how diffuse their basis
representation happens to be.

The empirical consequence, measured over 40 molecules with independently
determined gas-phase electron affinities (photoelectron spectroscopy, electron
transfer equilibria, and related direct methods):

| estimator                              | Spearman ρ | MAE after affine map |
|----------------------------------------|-----------:|---------------------:|
| TOM LUMO, negated (superseded input)   |     +0.342 |             0.682 eV |
| structural ridge, class-disjoint CV    |     +0.693 |             0.607 eV |
| **ΔSCF EA (GFN2-xTB, gas)**            |   **+0.912** |         **0.289 eV** |
| ΔSCF EA (GFN2-xTB, ALPB MeCN)          |     +0.808 |             0.397 eV |

Permutation control on the same set: |ρ| 95th percentile under shuffled labels
is **0.310**.

The reading is that TOM LUMO points in the *right direction* (ρ = +0.34 once
negated) but is statistically indistinguishable from the permutation bar — it
carries almost no usable ranking information, which is exactly consistent with
the 0.06 seen downstream. An earlier draft of this ADR claimed TOM LUMO was
anti-correlated; that was a sign-convention error on a smaller 26-molecule
probe (raw LUMO correlates at −0.34, which *is* the correct physical direction
once negated). The corrected conclusion is weaker but still decisive: the
descriptor is not wrong-signed, it is uninformative, and no amount of label
cleaning can add information a descriptor does not contain.

## Decision

Compute reduction stability as the **ΔSCF vertical electron affinity**

    EA = E(neutral, N electrons) − E(anion, N+1 electrons)

both at the neutral geometry, using GFN2-xTB single points. Higher EA means the
molecule accepts an electron more readily, i.e. it is *less* reduction-stable —
which is exactly the SEI-formation axis a battery electrolyte campaign cares
about.

The raw xTB ΔSCF value is affine-mapped onto the experimental EA scale by an
OLS fit on the clean experimental set (`experimental_electron_affinity.json`).
An affine map cannot change Spearman ρ, so this is a unit change, not a
ranking fit — the same discipline already applied to the xTB orbital scale.

### Why gas phase and not ALPB

ALPB ranks slightly *worse* (0.808 vs 0.890) on a gas-phase reference set. That
is expected — the reference values are gas-phase, so ALPB is being penalised for
modelling a solvent that was not present in the measurement. It is not evidence
that implicit solvation is wrong for the application. Since we have no clean
solution-phase reduction-potential set to calibrate against, we take the
defensible option: calibrate against gas-phase EA where clean experimental data
exists, and expose the ALPB variant as an optional, uncalibrated field for
downstream use. This is recorded as a known limitation rather than hidden.

### Fallback when xTB is absent

xTB is unavailable on CI and on many laptops, and graceful degradation is a hard
project constraint. The fallback is a ridge model on interpretable
electron-accepting structural features (π-acceptor count, nitro/cyano/carbonyl
counts, halogen inductive terms, aromatic ring count, conjugation extent),
fitted on the same clean experimental EA set. Validated leave-one-chemical-
class-out (ρ = 0.693, MAE 0.607 eV) rather than by random split, so quinones,
nitroaromatics and polyacenes are each predicted by a model that never saw a
member of their class. That is roughly twice the ranking signal of the TOM
LUMO it replaces, without requiring xTB.

The fallback reports its own confidence, and predictions carry a
`method` field (`xtb_dscf` / `structural_ridge`) so no consumer can silently
confuse the two.

## Consequences

- `reduction_stability_proxy` changes meaning: it now reports `ea_eV`
  (higher = easier to reduce = worse SEI stability) rather than `lumo_eV`.
  The old key is retained alongside it for one release for compatibility, but
  is no longer used for ranking.
- `LumoProxy` is retained for LUMO *calibration* (MAE) only and is explicitly
  documented as not a ranking input. It is no longer wired into selection.
- Ranking claims on the reduction axis are now backed by an experimental,
  single-measurement-class benchmark, so Spearman ρ is a legitimate metric
  for this target for the first time.

## Honest limitations

- The experimental EA set is gas-phase and skews toward molecules with
  *measurable* (positive) EAs — quinones, nitroaromatics, cyanocarbons,
  polyacenes. Most electrolyte solvents have negative EA and are therefore
  outside the calibrated span; for them the model provides ranking, not
  trustworthy absolute values.
- GFN2-xTB anion energies for species with unbound anions are basis-limited;
  the computed EA for such molecules is a bounded extrapolation, not a physical
  binding energy. The `ea_in_calibrated_span` flag marks these.
- Vertical, not adiabatic. Anion geometry relaxation is not included (xTB
  `--opt` on the anion is unstable for several test molecules and roughly 20×
  the cost). Vertical EA is the appropriate quantity for the fast
  electron-transfer step anyway.
- ALPB solution-phase reduction potentials remain uncalibrated pending a clean
  experimental reduction-potential set.

## Files
- `src/aurelius/data/experimental_electron_affinity.json` — clean EA set
- `src/aurelius/scoring/oracle/reduction.py` — ReductionOracle
- `benchmarks/benchmark_reduction_axis.py` — ρ/MAE on held-out molecules
- `tests/test_reduction_oracle.py`

## References
- ADR-2026-08-09 / -02: superseded for ranking
- ADR-2026-08-08-01: LPM (the occupied-space analogue that does work)
