"""Active experiment suggestion — close the wet-lab loop.

ADR-2026-08-08-02: Aurelius could already *ingest* measurements
(``experimental_ingestion.py``) but never *requested* any. The loop was
one-way: a chemist had to guess which measurement would help the model most.
This module inverts that — it ranks candidate molecule/property pairs by how
much measuring them would improve the oracle, and emits a structured,
rationalised worklist.

Selection criterion
-------------------
This is active learning, so the objective is *information gain per
measurement*, which is not the same as "measure the best-scoring molecule".
A molecule the model is already confident about teaches nothing. Four terms
are combined, each already computed elsewhere in the codebase:

``uncertainty`` — width of the split-conformal prediction interval for the
    property, normalised by the calibration width. A wide interval means the
    model's own distribution-free guarantee is loose here. This is the
    classical uncertainty-sampling term.

``bias`` — magnitude of the detected systematic deviation for that property
    (``FeedbackController.detect_systematic_bias``). Systematic error is not
    reducible by more measurements of the same kind, but it *is* the strongest
    signal that a physical constant needs re-derivation, so measurements that
    probe a biased property are worth more.

``novelty`` — one minus the maximum Tanimoto similarity to the calibration
    set. A molecule chemically identical to something already measured adds a
    near-duplicate row and little information; the conformal guarantee is an
    exchangeability argument, so broadening the calibration distribution is
    what actually tightens intervals on new chemistry.

``doa_proximity`` — closeness to the domain-of-applicability boundary.
    Measurements deep inside the domain confirm what is known; measurements
    far outside it cannot be modelled at all. The most informative point is
    near the boundary, where a measurement decides whether the domain should
    be extended. This term therefore peaks at the edge rather than increasing
    monotonically — see :func:`_doa_proximity_score`.

The weighted sum is deliberately transparent rather than a learned
acquisition function: a chemist deciding whether to spend a week on a
measurement should be able to read why it was suggested.

Honest limitations
------------------
* This ranks by *expected* information gain under the model's own
  uncertainty estimate. If the model is confidently wrong in a region it does
  not know about, no acquisition function built from its own uncertainty will
  point there. Novelty partially mitigates this; it does not solve it.
* Synthetic accessibility is applied as a hard filter, not a term: a
  suggestion nobody can make is worthless regardless of its information
  content. The threshold is deliberately permissive.
* No cost model. A HOMO measurement (photoelectron spectroscopy) and a
  viscosity measurement (falling-ball) differ by orders of magnitude in
  effort, and the ranking does not know that. Property-specific costs would
  need to come from the lab.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from rdkit import Chem

from aurelius.types import MoleculeContext

logger = logging.getLogger(__name__)

# Properties a wet lab can measure that the oracle also predicts. Keys match
# ConformalPredictor property names and the ingestion schema's
# ``measured_property`` values.
MEASURABLE_PROPERTIES: dict[str, str] = {
    "homo": "homo_eV",
    "lumo": "lumo_eV",
    "dielectric": "dielectric_constant",
    "viscosity": "viscosity_cP",
}

# Term weights. Uncertainty dominates because it is the term with a formal
# guarantee behind it; novelty is next because calibration-set coverage is
# what makes that guarantee transfer.
DEFAULT_WEIGHTS: dict[str, float] = {
    "uncertainty": 0.40,
    "novelty": 0.25,
    "doa_proximity": 0.20,
    "bias": 0.15,
}

# A suggestion must be plausibly synthesizable to be worth lab time.
MAX_SA_SCORE = 6.0


@dataclass
class ExperimentSuggestion:
    """One recommended measurement, with the reasoning that produced it."""

    smiles: str
    property_to_measure: str
    priority_score: float
    rationale: str
    predicted_value: float
    prediction_interval: tuple[float, float]
    components: dict[str, float] = field(default_factory=dict)
    units: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["prediction_interval"] = list(self.prediction_interval)
        return data


_UNITS = {
    "homo_eV": "eV",
    "lumo_eV": "eV",
    "dielectric_constant": "dimensionless",
    "viscosity_cP": "cP",
}


def _doa_proximity_score(doa_penalty: float) -> float:
    """Score a molecule by closeness to the domain-of-applicability edge.

    ``doa_penalty`` is 1.0 fully in-domain and falls toward 0.70 as the
    molecule leaves the calibration domain (see
    ``quantum.compute_quantum_domain_penalty``).

    Information is maximised at the *boundary*, not beyond it: a molecule the
    model handles comfortably teaches little, and one far outside the domain
    cannot be interpreted even once measured. This maps the penalty to a
    triangular score peaking at the midpoint of the penalty range.
    """
    floor = 0.70
    if doa_penalty >= 1.0:
        return 0.0
    if doa_penalty <= floor:
        return 0.5  # beyond the edge: informative, but hard to act on
    # Normalise to [0, 1] where 1 = at the floor, then peak at the midpoint.
    depth = (1.0 - doa_penalty) / (1.0 - floor)
    return round(1.0 - abs(depth - 0.5) * 2.0 + depth * 0.5, 6)


def _max_tanimoto_to_calibration(fp: Any, calibration_fps: list[Any]) -> float:
    """Highest Tanimoto similarity between a molecule and the calibration set."""
    if not calibration_fps:
        return 0.0
    from rdkit import DataStructs

    return float(max(DataStructs.BulkTanimotoSimilarity(fp, calibration_fps)))


def _load_calibration_fingerprints() -> list[Any]:
    """Morgan fingerprints of every molecule the oracle has been calibrated on."""
    from aurelius.scoring.oracle.conformal import (
        _load_external_benchmark,
        _load_orbital_calibration,
    )

    fps = []
    seen: set[str] = set()
    for entry in list(_load_orbital_calibration()) + list(_load_external_benchmark()):
        smiles = entry.get("smiles")
        if not smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol)
        if canonical in seen:
            continue
        seen.add(canonical)
        ctx = MoleculeContext.from_smiles(canonical)
        if ctx is not None:
            fps.append(ctx.get_ecfp4())
    return fps


def _bias_magnitudes(controller: Any | None) -> dict[str, float]:
    """Normalised systematic-bias magnitude per property, in [0, 1].

    Normalised by the detection threshold, so a value of 1.0 means the bias
    is exactly at the level the controller considers actionable.
    """
    if controller is None:
        return {}
    try:
        bias = controller.detect_systematic_bias()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Bias detection failed (%s); ignoring bias term.", exc)
        return {}
    thresholds = {"dielectric": 2.0, "viscosity": 0.5}
    out: dict[str, float] = {}
    for prop, info in bias.items():
        threshold = thresholds.get(prop, 1.0)
        out[prop] = min(1.0, float(info.get("magnitude", 0.0)) / threshold)
    return out


def _normalised_interval_width(
    predictor: Any, prop: str, point: float, mol: Any
) -> tuple[float, tuple[float, float]]:
    """Conformal interval width as a fraction of the property's natural spread.

    Normalising by the spread of the reference values — rather than by the
    calibration quantile — is what makes the term informative. Dividing by
    the quantile returns ~1.0 for every molecule by construction, because the
    quantile *is* the typical width; the resulting constant signal silently
    disables uncertainty sampling.
    """
    lower, upper = predictor.predict_interval(prop, point, mol=mol)
    width = upper - lower
    spread = getattr(predictor, "_reference_spread", {}).get(prop, 0.0)
    if spread <= 0:
        return 0.0, (lower, upper)
    ratio = width / spread
    # Saturating rather than clipped: hard-clipping at 1.0 collapses every
    # property whose interval exceeds its reference spread onto the same
    # value, which is exactly the regime the dielectric model is in, and
    # would re-introduce the constant-signal problem this term exists to fix.
    # x/(1+x) is monotone on [0, inf) and keeps the term in [0, 1).
    return ratio / (1.0 + ratio), (lower, upper)


def _predicted_values(ctx: MoleculeContext) -> dict[str, float]:
    """Oracle point predictions for every measurable property."""
    from aurelius.scoring.oracle.gc import (
        predict_dielectric_proxy,
        predict_viscosity_proxy,
    )
    from aurelius.scoring.oracle.quantum import QuantumOracle

    orbitals = QuantumOracle().evaluate(ctx.mol)
    return {
        "homo": float(orbitals["homo_eV"]),
        "lumo": float(orbitals["lumo_eV"]),
        "dielectric": float(predict_dielectric_proxy(ctx)),
        "viscosity": float(predict_viscosity_proxy(ctx)),
    }


def _build_rationale(
    prop: str, components: dict[str, float], interval: tuple[float, float], units: str
) -> str:
    """Explain, in a chemist's terms, why this measurement was requested."""
    drivers = sorted(components.items(), key=lambda kv: -kv[1])
    lead, lead_value = drivers[0]
    phrases = {
        "uncertainty": (
            f"the model's 90% conformal interval spans "
            f"{interval[1] - interval[0]:.2f} {units}, which is wide relative to "
            f"its calibration residuals"
        ),
        "novelty": "this scaffold is chemically distant from every calibration molecule",
        "doa_proximity": (
            "the molecule sits near the edge of the oracle's domain of "
            "applicability, so a measurement decides whether the domain extends here"
        ),
        "bias": (
            f"the {prop} model currently shows a systematic offset against "
            f"ingested measurements"
        ),
    }
    secondary = [name for name, value in drivers[1:] if value > 0.4]
    text = f"Measure {prop}: {phrases.get(lead, lead)} (weight {lead_value:.2f})."
    if secondary:
        text += " Also " + ", ".join(phrases.get(s, s) for s in secondary) + "."
    return text


def _score_property(
    prop: str,
    point: float,
    predictor: Any,
    mol: Any,
    novelty: float,
    doa: float,
    biases: dict[str, float],
    weights: dict[str, float],
) -> tuple[float, dict[str, float], tuple[float, float]]:
    uncertainty, interval = _normalised_interval_width(predictor, prop, point, mol)
    components = {
        "uncertainty": round(uncertainty, 4),
        "novelty": round(novelty, 4),
        "doa_proximity": round(doa, 4),
        "bias": round(biases.get(prop, 0.0), 4),
    }
    score = sum(weights.get(name, 0.0) * value for name, value in components.items())
    return round(score, 6), components, interval


def suggest_experiments(
    candidates: list[str],
    top_n: int = 10,
    controller: Any | None = None,
    properties: list[str] | None = None,
    weights: dict[str, float] | None = None,
    max_sa_score: float = MAX_SA_SCORE,
) -> list[ExperimentSuggestion]:
    """Rank molecule/property pairs by expected information gain.

    Args:
        candidates: SMILES to consider. Invalid entries are skipped.
        top_n: Number of suggestions to return.
        controller: Optional ``FeedbackController`` supplying systematic-bias
            estimates. Without one the bias term is zero.
        properties: Restrict to these property names (keys of
            ``MEASURABLE_PROPERTIES``). Defaults to all.
        weights: Override the term weights.
        max_sa_score: Reject candidates harder to make than this.

    Returns:
        Up to ``top_n`` suggestions, highest priority first.
    """
    from aurelius.scoring.oracle.conformal import get_conformal_predictor
    from aurelius.scoring.oracle.quantum import compute_quantum_domain_penalty
    from aurelius.utils.chem_utils import electrolyte_synthetic_accessibility

    active_weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    wanted = properties or list(MEASURABLE_PROPERTIES)
    unknown = set(wanted) - set(MEASURABLE_PROPERTIES)
    if unknown:
        raise ValueError(f"Unknown properties: {sorted(unknown)}")

    predictor = get_conformal_predictor()
    calibration_fps = _load_calibration_fingerprints()
    biases = _bias_magnitudes(controller)

    suggestions: list[ExperimentSuggestion] = []
    for smiles in candidates:
        ctx = MoleculeContext.from_smiles(smiles)
        if ctx is None:
            logger.debug("Skipping unparseable SMILES: %s", smiles)
            continue
        if electrolyte_synthetic_accessibility(ctx) > max_sa_score:
            continue

        novelty = 1.0 - _max_tanimoto_to_calibration(ctx.get_ecfp4(), calibration_fps)
        doa_penalty, doa_reason = compute_quantum_domain_penalty(ctx)
        doa = _doa_proximity_score(doa_penalty)

        try:
            predictions = _predicted_values(ctx)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Oracle failed for %s (%s); skipping.", smiles, exc)
            continue

        for prop in wanted:
            point = predictions[prop]
            score, components, interval = _score_property(
                prop, point, predictor, ctx.mol, novelty, doa, biases, active_weights
            )
            canonical_property = MEASURABLE_PROPERTIES[prop]
            units = _UNITS[canonical_property]
            rationale = _build_rationale(prop, components, interval, units)
            if doa_reason != "within domain":
                rationale += f" Domain note: {doa_reason}."
            suggestions.append(
                ExperimentSuggestion(
                    smiles=Chem.MolToSmiles(ctx.mol),
                    property_to_measure=canonical_property,
                    priority_score=score,
                    rationale=rationale,
                    predicted_value=round(point, 4),
                    prediction_interval=(round(interval[0], 4), round(interval[1], 4)),
                    components=components,
                    units=units,
                )
            )

    suggestions.sort(key=lambda s: (-s.priority_score, s.smiles, s.property_to_measure))
    return _diversify(suggestions, top_n)


# Each additional suggestion for an already-chosen molecule or property is
# discounted by this factor, compounding. 0.6 is strong enough to interleave
# properties without ever letting a weak suggestion outrank a much stronger one.
_REDUNDANCY_DISCOUNT = 0.6


def _diversify(ranked: list[ExperimentSuggestion], top_n: int) -> list[ExperimentSuggestion]:
    """Greedily pick a worklist that is not four copies of the same experiment.

    Ranking each (molecule, property) pair independently produces a list whose
    top entries are near-duplicates: the same property on similar scaffolds,
    because the terms that drive the score vary smoothly across chemistry.
    A batch of measurements is only worth its *joint* information, and
    near-duplicate measurements have highly correlated residuals, so the
    marginal value of the second one is much lower than its standalone score
    suggests.

    This applies a compounding redundancy discount to molecules and properties
    already represented in the batch — the standard greedy approximation to
    batch-mode active learning. It is a re-ordering only: nothing new enters
    the list, and the discount is reported so the effect is auditable.
    """
    selected: list[ExperimentSuggestion] = []
    remaining = list(ranked)
    seen_molecules: dict[str, int] = {}
    seen_properties: dict[str, int] = {}

    while remaining and len(selected) < top_n:
        best_index, best_value = 0, -1.0
        for i, candidate in enumerate(remaining):
            penalty = _REDUNDANCY_DISCOUNT ** (
                seen_molecules.get(candidate.smiles, 0)
                + seen_properties.get(candidate.property_to_measure, 0)
            )
            value = candidate.priority_score * penalty
            if value > best_value:
                best_index, best_value = i, value
        chosen = remaining.pop(best_index)
        seen_molecules[chosen.smiles] = seen_molecules.get(chosen.smiles, 0) + 1
        seen_properties[chosen.property_to_measure] = (
            seen_properties.get(chosen.property_to_measure, 0) + 1
        )
        chosen.components["batch_adjusted_score"] = round(best_value, 6)
        selected.append(chosen)

    return selected


def write_suggestions(suggestions: list[ExperimentSuggestion], path: str) -> None:
    """Write suggestions to JSON in a form the ingestion schema mirrors."""
    payload = {
        "n_suggestions": len(suggestions),
        "weights": DEFAULT_WEIGHTS,
        "suggestions": [s.to_dict() for s in suggestions],
        "note": (
            "Ranked by expected information gain, not by predicted performance. "
            "A high-priority molecule is one the oracle is least able to predict, "
            "which is what makes measuring it worthwhile."
        ),
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def default_candidate_pool() -> list[str]:
    """Candidate pool used when the caller supplies none.

    Drawn from the discovery seed pool, which is the chemistry the loop
    actually explores.
    """
    import os

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    try:
        with open(os.path.join(data_dir, "tier0_seed_smiles.json")) as fh:
            seeds = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []
    if seeds and isinstance(seeds[0], dict):
        return [s.get("smiles", "") for s in seeds if s.get("smiles")]
    return [s for s in seeds if isinstance(s, str)]
