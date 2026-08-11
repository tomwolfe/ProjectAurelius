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

``expected_impact`` — how much the measurement would move the *decision*, not
    just the model (ADR-2026-08-10-03). The four terms above are all
    model-centric: they maximise information about the oracle while being
    indifferent to whether that information changes which molecules get made.
    A wide interval on a molecule that is certainly mediocre is worth less
    than a narrow interval on a molecule sitting on the top-k boundary, where
    the measurement decides inclusion. This term estimates the probability
    that the true value falls on the other side of the current top-k
    threshold — the classic "expected change in the selected set" argument —
    using the conformal interval as the predictive distribution. See
    :func:`_expected_impact_score`.

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
from rdkit.Chem import BRICS

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

# Pool expansion target. Below this size, BRICS harvesting grows the pool
# before acquisition scoring. The 61-mol unmeasured pool in the closed-loop
# benchmark is too small for diversity-based acquisition to outperform random.
MIN_POOL_SIZE = 200

# Minimum pool size to attempt expansion. Below this, BRICS harvesting cannot
# produce meaningful fragments (the pool is too small/homologous), and the
# acquisition scores the candidates directly.
_MIN_EXPAND_POOL = 10

# Maximum number of BRICS fragments to harvest per candidate molecule.
_MAX_FRAGMENTS_PER_MOL = 10

# Maximum number of recombination products to generate.
_MAX_RECOMBINATION_PRODUCTS = 200

# Term weights. Uncertainty dominates because it is the term with a formal
# guarantee behind it; novelty is next because calibration-set coverage is
# what makes that guarantee transfer. BALD and pareto_ucb are the Phase 2
# acquisition terms targeting epistemic variance on the Pareto frontier.
DEFAULT_WEIGHTS: dict[str, float] = {
    "uncertainty": 0.20,
    "expected_impact": 0.15,
    "novelty": 0.15,
    "doa_proximity": 0.10,
    "bias": 0.05,
    "bald": 0.20,
    "pareto_ucb": 0.15,
}

# Exploration-exploitation tradeoff for Pareto UCB. Higher values favour
# exploration (uncertain candidates); lower values favour exploitation.
_PARETO_UCB_BETA = 1.5

# Fraction of the candidate pool treated as the "decision set" when computing
# expected impact. Measurements only change what gets made if they can move a
# molecule across this boundary.
TOP_K_FRACTION = 0.25

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


def _expected_impact_score(
    point: float,
    interval: tuple[float, float],
    threshold: float | None,
) -> float:
    """Probability the measurement moves this molecule across the top-k boundary.

    Uncertainty sampling asks "where is the model least sure?". That is the
    right question for building a model and the wrong one for running a
    discovery campaign, where the only measurements that pay for themselves are
    those that can change *which molecules get made*. A molecule predicted far
    from the decision boundary teaches the model something even when the answer
    is unsurprising, but it cannot alter the shortlist.

    The conformal interval is treated as a 90% predictive interval and
    approximated by a Gaussian with matching coverage (the half-width is
    1.645σ). The score is then

        2 · min(P(y > threshold), P(y ≤ threshold))

    which is 1.0 when the prediction sits exactly on the boundary — a coin
    flip, maximum decision-relevance — and decays to 0 as the molecule becomes
    unambiguously in or out. Doubling normalises the maximum to 1.0.

    Returns 0.0 when no threshold is available (single-candidate calls), so the
    term degrades to neutral rather than to a misleading constant.
    """
    if threshold is None:
        return 0.0

    half_width = (interval[1] - interval[0]) / 2.0
    if half_width <= 0:
        # A zero-width interval means the model claims certainty: the
        # measurement can only change the decision if the point is exactly on
        # the boundary, which is measure-zero.
        return 0.0

    sigma = half_width / 1.645
    from math import erf, sqrt

    z = (threshold - point) / (sigma * sqrt(2.0))
    p_below = 0.5 * (1.0 + erf(z))
    return round(2.0 * min(p_below, 1.0 - p_below), 6)


def _decision_thresholds(
    predictions_by_property: dict[str, list[float]],
    top_k_fraction: float = TOP_K_FRACTION,
) -> dict[str, float]:
    """Per-property top-k cut-off across the candidate pool.

    The threshold is the value a molecule must beat to enter the top
    ``top_k_fraction`` of the pool on that property. Properties where more is
    better (dielectric) and less is better (viscosity) both work, because the
    expected-impact score only cares about distance to the boundary, not which
    side is preferable.
    """
    import numpy as np

    thresholds: dict[str, float] = {}
    for prop, values in predictions_by_property.items():
        if len(values) < 4:
            continue
        thresholds[prop] = float(np.quantile(values, 1.0 - top_k_fraction))
    return thresholds


# ---------------------------------------------------------------------------
# Pool expansion via BRICS harvesting
# ---------------------------------------------------------------------------
# A small candidate pool (<100 mols) cannot support diversity-based acquisition
# — the search space is already saturated and any selection ≈ random. BRICS
# harvesting breaks high-scoring candidates into fragments and recombines them
# to grow the pool before acquisition scoring.


def _strip_brics_dummy_atoms(frag_smi: str) -> str | None:
    """Remove BRICS dummy-atom labels (*) from a fragment SMILES."""
    mol = Chem.MolFromSmiles(frag_smi)
    if mol is None:
        return None
    rw = Chem.RWMol(mol)
    for idx in sorted(
        (a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0),
        reverse=True,
    ):
        rw.RemoveAtom(idx)
    try:
        rw.UpdatePropertyCache()
        Chem.SanitizeMol(rw)
    except Exception:
        return None
    return Chem.MolToSmiles(rw)


def _harvest_fragments(candidates: list[str]) -> list[str]:
    """Decompose candidates into BRICS fragments, strip dummies, deduplicate."""
    seen: set[str] = set()
    fragments: list[str] = []
    for smi in candidates:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            for frag in BRICS.BRICSDecompose(mol):
                core = _strip_brics_dummy_atoms(frag)
                if core is None or core in seen:
                    continue
                core_mol = Chem.MolFromSmiles(core)
                if core_mol is None or core_mol.GetNumHeavyAtoms() < 2:
                    continue
                seen.add(core)
                fragments.append(core)
        except Exception:
            continue
    return fragments


def _recombine_fragments(fragments: list[str], max_products: int = _MAX_RECOMBINATION_PRODUCTS) -> list[str]:
    """Recombine BRICS fragments into novel candidate molecules.

    Uses the project's BRICS infrastructure (``find_complementary_pairs``,
    ``inject_linkers``) to prepare a fragment pool with valid attachment
    points, then ``BRICSBuild`` to generate products. Returns up to
    ``max_products`` unique SMILES.

    Fragments without complementary partners are bridged by universal linker
    fragments (ether, ester, methylene, …) so the pool is never empty even
    when the harvested fragments share no bond types.
    """
    from rdkit.Chem.BRICS import BRICSBuild

    from aurelius.agent.mutation.brics import (
        find_complementary_pairs,
        inject_linkers,
    )

    mols: list[Any] = []
    for f in fragments:
        m = Chem.MolFromSmiles(f)
        if m is not None:
            mols.append(m)
    if len(mols) < 2:
        return []

    # Inject universal linkers when the harvested fragments have no
    # complementary pairs — without this, BRICSBuild produces nothing.
    pairs = find_complementary_pairs(mols)
    if not pairs:
        inject_linkers(mols)

    seen: set[str] = set()
    products: list[str] = []
    try:
        for product in BRICSBuild(mols):
            smi = Chem.MolToSmiles(product)
            if smi in seen:
                continue
            seen.add(smi)
            products.append(smi)
            if len(products) >= max_products:
                break
    except Exception:
        pass
    return products


def _quick_score_candidates(
    candidates: list[str],
) -> list[tuple[str, float]]:
    """Lightweight pre-scoring to rank candidates for pool expansion.

    Uses oracle point predictions only (no conformal intervals, no DoA) to
    produce a composite quality score. This ranks candidates so that
    `_harvest_from_top_candidates` harvests fragments from the most promising
    molecules, yielding higher-quality recombination products.

    Returns list of (canonical_smiles, score) sorted descending by score.
    """
    scored: list[tuple[str, float]] = []
    for smi in candidates:
        ctx = MoleculeContext.from_smiles(smi)
        if ctx is None:
            continue
        try:
            predictions = _predicted_values(ctx)
        except Exception:
            continue
        score = (
            -0.30 * predictions["homo"]
            - 0.25 * predictions["lumo"]
            + 0.25 * predictions["dielectric"]
            - 0.20 * predictions["viscosity"]
        )
        canonical = Chem.MolToSmiles(ctx.mol)
        scored.append((canonical, score))
    scored.sort(key=lambda x: -x[1])
    return scored


def _has_brics_dummy(frag_smi: str) -> bool:
    """True if the fragment SMILES contains a BRICS dummy atom ([N*])."""
    mol = Chem.MolFromSmiles(frag_smi)
    if mol is None:
        return False
    return any(a.GetAtomicNum() == 0 for a in mol.GetAtoms())


def _harvest_labeled_fragments(top_candidates: list[str]) -> list[str]:
    """Decompose top candidates into BRICS fragments with dummy-atom labels intact.

    Unlike :func:`_harvest_fragments`, the labels are kept so the fragments
    remain recombinable via BRICSBuild. Duplicates are removed by stripped core.

    Fragments without dummy atoms (whole molecules BRICS could not decompose)
    are skipped — they have no attachment points and cannot participate in
    recombination.
    """
    seen_cores: set[str] = set()
    labeled_frags: list[str] = []
    for smi in top_candidates:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            for frag in BRICS.BRICSDecompose(mol):
                if not _has_brics_dummy(frag):
                    continue
                core = _strip_brics_dummy_atoms(frag)
                if core is None or core in seen_cores:
                    continue
                core_mol = Chem.MolFromSmiles(core)
                if core_mol is None or core_mol.GetNumHeavyAtoms() < 2:
                    continue
                seen_cores.add(core)
                labeled_frags.append(frag)
        except Exception:
            continue
    return labeled_frags


def _dedup_pool(candidates: list[str], products: list[str]) -> list[str]:
    """Merge candidates and generated products, deduplicating by canonical SMILES."""
    seen: set[str] = set()
    merged: list[str] = []
    for smi in list(candidates) + products:
        canonical = Chem.CanonSmiles(smi) if Chem.MolFromSmiles(smi) else None
        if canonical is None or canonical in seen:
            continue
        seen.add(canonical)
        merged.append(canonical)
    return merged


def _harvest_from_top_candidates(
    candidates: list[str],
    top_n: int = 10,
    max_products: int = 150,
) -> list[str]:
    """Harvest BRICS fragments from top-scoring candidates and recombine them.

    Takes the top-`top_n` candidates by a lightweight oracle quality score,
    decomposes each into BRICS fragments (keeping dummy-atom labels for valid
    recombination), recombines via BRICSBuild, and deduplicates against the
    existing pool.

    Returns the expanded pool (original + generated). If BRICS recombination
    produces no novel molecules (small or homologous pool), returns the
    original pool unchanged.
    """
    if len(candidates) < 2:
        return list(candidates)

    scored = _quick_score_candidates(candidates)
    top_candidates = [smi for smi, _ in scored[:top_n]]

    labeled_frags = _harvest_labeled_fragments(top_candidates)
    products: list[str] = []
    if len(labeled_frags) >= 2:
        products = _recombine_fragments(labeled_frags, max_products=max_products)

    # Filter to genuinely novel products
    original_canonical: set[str] = set()
    for s in candidates:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            original_canonical.add(Chem.MolToSmiles(m))
    novel_brics = [p for p in products if p not in original_canonical]

    if novel_brics:
        return _dedup_pool(candidates, novel_brics)

    return list(candidates)


def expand_candidate_pool(candidates: list[str], target_size: int = MIN_POOL_SIZE) -> list[str]:
    """Grow a small candidate pool via BRICS harvesting + recombination.

    If the pool is already at or above target_size, returns the input unchanged.
    Otherwise harvests fragments from top-scoring candidates, recombines them,
    and merges the products back into the pool. Deduplicates by canonical SMILES.

    Returns the expanded pool (original + generated), or the original pool if
    expansion fails to produce any genuinely new molecules.
    """
    if len(candidates) >= target_size:
        return list(candidates)

    expanded = _harvest_from_top_candidates(candidates, top_n=10, max_products=150)

    # Count genuinely new molecules — the input pool may contain duplicates
    # that _dedup_pool collapses, so comparing raw lengths would reject a
    # valid expansion that added novel structures.
    original_canonical: set[str] = set()
    for s in candidates:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            original_canonical.add(Chem.MolToSmiles(m))
    novel = [s for s in expanded if s not in original_canonical]
    if not novel:
        return list(candidates)

    return expanded


# ---------------------------------------------------------------------------
# Expected Improvement (EI) acquisition
# ---------------------------------------------------------------------------
# EI is the single-objective special case of EHVI. For the closed-loop
# benchmark the target is HOMO; EI measures how much better a measurement
# would be over the current best, under the conformal interval as the
# predictive distribution. This replaces the linear weighted sum with a
# quantity that has a formal decision-theoretic justification.


def expected_improvement(
    point: float,
    interval: tuple[float, float],
    current_best: float,
    minimise: bool = True,
) -> float:
    """Expected Improvement over current_best under a Gaussian approximation.

    The conformal interval is treated as a 90% predictive interval and
    approximated by a Gaussian with matching coverage (half-width = 1.645σ).

    Args:
        point: Predicted value (mean of the Gaussian).
        interval: (lower, upper) conformal prediction interval.
        current_best: Best observed value so far (the incumbent).
        minimise: True if lower is better (HOMO, LUMO), False if higher
            is better (dielectric).

    Returns:
        Expected improvement ≥ 0. Zero when the interval has zero width.
    """
    half_width = (interval[1] - interval[0]) / 2.0
    if half_width <= 0:
        return 0.0

    sigma = half_width / 1.645
    from math import erf, sqrt

    improvement = current_best - point if minimise else point - current_best

    if sigma <= 0:
        return max(0.0, improvement)

    z = improvement / (sigma * sqrt(2.0))
    # EI = improvement * CDF(z) + sigma * PDF(z)
    # Using: CDF(z) = 0.5 * (1 + erf(z)), PDF(z) = exp(-z^2) / sqrt(2*pi)
    cdf = 0.5 * (1.0 + erf(z))
    pdf = _standard_normal_pdf(z)
    ei = improvement * cdf + sigma * pdf
    return max(0.0, ei)


def _standard_normal_pdf(z: float) -> float:
    """Standard normal PDF at z."""
    from math import exp, pi, sqrt
    return exp(-0.5 * z * z) / sqrt(2.0 * pi)


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
        "expected_impact": (
            "the prediction sits close to the shortlist cut-off, so the "
            "measurement is likely to decide whether this molecule is pursued"
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
        "bald": (
            "this molecule has high epistemic uncertainty in the GPR posterior, "
            "so measuring it would most reduce model uncertainty"
        ),
        "pareto_ucb": (
            "the molecule sits on the optimistic Pareto frontier across "
            "multiple objectives, so the measurement could shift the frontier"
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
    threshold: float | None = None,
    bald_score: float = 0.0,
    pareto_score: float = 0.0,
) -> tuple[float, dict[str, float], tuple[float, float]]:
    uncertainty, interval = _normalised_interval_width(predictor, prop, point, mol)
    components = {
        "uncertainty": round(uncertainty, 4),
        "expected_impact": _expected_impact_score(point, interval, threshold),
        "novelty": round(novelty, 4),
        "doa_proximity": round(doa, 4),
        "bias": round(biases.get(prop, 0.0), 4),
        "bald": round(bald_score, 4),
        "pareto_ucb": round(pareto_score, 4),
    }
    score = sum(weights.get(name, 0.0) * value for name, value in components.items())
    return round(score, 6), components, interval


_Evaluated = tuple["MoleculeContext", dict[str, float], float, float, str]
_RawSuggestion = tuple[
    float, dict[str, float], tuple[float, float], str, float, str, str
]


def _evaluate_candidates(
    candidates: list[str],
    calibration_fps: list[Any],
    max_sa_score: float,
) -> tuple[list[_Evaluated], dict[str, Any]]:
    """Score every candidate once: predictions, novelty, DoA and fingerprint.

    Split out of :func:`suggest_experiments` so the expected-impact term can be
    computed against pool-wide decision thresholds without pushing the caller
    over the project's cyclomatic-complexity budget.

    Returns ``(evaluated, fingerprints)`` where each evaluated entry is
    ``(ctx, predictions, novelty, doa_score, doa_reason)``.
    """
    from aurelius.scoring.oracle.quantum import compute_quantum_domain_penalty
    from aurelius.utils.chem_utils import electrolyte_synthetic_accessibility

    evaluated: list[_Evaluated] = []
    fingerprints: dict[str, Any] = {}

    for smiles in candidates:
        ctx = MoleculeContext.from_smiles(smiles)
        if ctx is None:
            logger.debug("Skipping unparseable SMILES: %s", smiles)
            continue
        if electrolyte_synthetic_accessibility(ctx) > max_sa_score:
            continue

        try:
            predictions = _predicted_values(ctx)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Oracle failed for %s (%s); skipping.", smiles, exc)
            continue

        fingerprints[Chem.MolToSmiles(ctx.mol)] = ctx.get_ecfp4()
        novelty = 1.0 - _max_tanimoto_to_calibration(ctx.get_ecfp4(), calibration_fps)
        doa_penalty, doa_reason = compute_quantum_domain_penalty(ctx)
        evaluated.append(
            (ctx, predictions, novelty, _doa_proximity_score(doa_penalty), doa_reason)
        )

    return evaluated, fingerprints


def _collect_raw_suggestions(
    evaluated: list[_Evaluated],
    all_predictions: list[dict[str, float]],
    wanted: list[str],
    predictor: Any,
    bald_scores: dict[str, float],
    per_prop_uncertainty: dict[str, dict[str, float]],
    biases: dict[str, float],
    weights: dict[str, float],
    thresholds: dict[str, float],
) -> tuple[list[_RawSuggestion], list[float]]:
    """Pass 2 of suggestion: score each (molecule, property) pair.

    For every evaluated molecule and every requested property, computes the
    composite priority score and the raw expected-improvement value. EI is left
    unnormalised so the caller can rescale it across the pool before blending.

    Returns ``(raw_suggestions, ei_values)``.
    """
    raw_suggestions: list[_RawSuggestion] = []
    ei_values: list[float] = []
    for ctx, predictions, novelty, doa, doa_reason in evaluated:
        smiles_key = Chem.MolToSmiles(ctx.mol)
        bald = bald_scores.get(smiles_key, 0.0)
        for prop in wanted:
            point = predictions[prop]
            pareto = pareto_ucb_score(
                predictions, per_prop_uncertainty.get(smiles_key, {}),
                all_predictions,
            )
            score, components, interval = _score_property(
                prop, point, predictor, ctx.mol, novelty, doa, biases,
                weights, thresholds.get(prop),
                bald_score=bald, pareto_score=pareto,
            )
            prop_values = [p[prop] for _, p, *_ in evaluated]
            minimise = prop in ("homo", "lump")
            incumbent = min(prop_values) if minimise else max(prop_values)
            ei = expected_improvement(point, interval, incumbent, minimise=minimise)
            ei_values.append(ei)
            raw_suggestions.append(
                (score, components, interval, smiles_key, point, prop, doa_reason)
            )
    return raw_suggestions, ei_values


def _build_suggestions_from_raw(
    raw_suggestions: list[_RawSuggestion],
    ei_values: list[float],
    ei_max: float,
) -> list[ExperimentSuggestion]:
    """Pass 3: turn scored pairs into ExperimentSuggestion objects with EI blending."""
    suggestions: list[ExperimentSuggestion] = []
    for (score, components, interval, smiles, point, prop, doa_reason), ei in zip(
        raw_suggestions, ei_values, strict=True
    ):
        ei_norm = ei / ei_max if ei_max > 0 else 0.0
        components["expected_improvement"] = round(ei, 6)
        blended = 0.8 * score + 0.2 * ei_norm
        canonical_property = MEASURABLE_PROPERTIES[prop]
        units = _UNITS[canonical_property]
        rationale = _build_rationale(prop, components, interval, units)
        if doa_reason != "within domain":
            rationale += f" Domain note: {doa_reason}."
        suggestions.append(
            ExperimentSuggestion(
                smiles=smiles,
                property_to_measure=canonical_property,
                priority_score=round(blended, 6),
                rationale=rationale,
                predicted_value=round(point, 4),
                prediction_interval=(round(interval[0], 4), round(interval[1], 4)),
                components=components,
                units=units,
            )
        )
    return suggestions


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

    active_weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    wanted = properties or list(MEASURABLE_PROPERTIES)
    unknown = set(wanted) - set(MEASURABLE_PROPERTIES)
    if unknown:
        raise ValueError(f"Unknown properties: {sorted(unknown)}")

    predictor = get_conformal_predictor()
    calibration_fps = _load_calibration_fingerprints()
    biases = _bias_magnitudes(controller)

    # Pool expansion: a small pool cannot support diversity-based acquisition.
    # BRICS harvesting grows it to >= target_size before scoring. Very small
    # pools (< _MIN_EXPAND_POOL) skip expansion — there is not enough material
    # for meaningful BRICS fragment harvesting, and the caller's candidates
    # should be scored directly.
    if _MIN_EXPAND_POOL <= len(candidates) < MIN_POOL_SIZE:
        expanded = expand_candidate_pool(candidates, target_size=MIN_POOL_SIZE)
        if len(expanded) > len(candidates):
            logger.info(
                "Pool expanded %d -> %d via BRICS harvesting",
                len(candidates), len(expanded),
            )
            candidates = expanded

    # Pass 1: evaluate every candidate once, so the decision thresholds used by
    # the expected-impact term reflect the whole pool rather than one molecule.
    evaluated, fingerprints = _evaluate_candidates(
        candidates, calibration_fps, max_sa_score
    )

    # Pre-compute BALD acquisition scores (Phase 2): epistemic posterior
    # variance per candidate molecule. This is a single batch call to the
    # MLX GPR surrogate, reusing the variance already computed there.
    all_predictions = [pred for _c, pred, *_ in evaluated]
    gpr_model = _gpr_model_from_predictor(predictor)
    bald_scores = _compute_bald_scores(evaluated, gpr_model)

    # Pre-compute per-property uncertainty for Pareto UCB (Phase 2).
    # We use the conformal interval half-width as a proxy for sigma on each
    # objective, since the GPR variance is molecule-level (not per-property).
    per_prop_uncertainty = _compute_per_property_uncertainty(
        evaluated, predictor, wanted,
    )

    thresholds = _decision_thresholds(
        {prop: [pred[prop] for _c, pred, *_ in evaluated] for prop in wanted}
    )

    # Pass 2: score each (molecule, property) pair against those thresholds.
    # EI is computed per-property using the incumbent (best calibrated value).
    # First pass collects raw EI values so they can be normalised to [0, 1]
    # across the pool — EI is in physical units (eV) and would otherwise
    # dominate the weighted sum which is in [0, 1].
    raw_suggestions, ei_values = _collect_raw_suggestions(
        evaluated, all_predictions, wanted, predictor,
        bald_scores, per_prop_uncertainty, biases,
        active_weights, thresholds,
    )

    ei_max = max(ei_values) if ei_values else 1.0
    suggestions = _build_suggestions_from_raw(raw_suggestions, ei_values, ei_max)

    suggestions.sort(key=lambda s: (-s.priority_score, s.smiles, s.property_to_measure))
    return _diversify(suggestions, top_n, fingerprints=fingerprints)


# ---------------------------------------------------------------------------
# BALD + Pareto UCB acquisition (Phase 2)
# ---------------------------------------------------------------------------
# BALD (Bayesian Active Learning by Disagreement) targets the expected reduction
# in GPR posterior variance — the measurement that most reduces epistemic
# uncertainty on the Pareto frontier. For a Gaussian posterior this reduces to
# the posterior variance sigma*^2(x) from the GPR: higher variance = more to
# learn from measuring this molecule.
#
# Pareto UCB (Upper Confidence Bound) scores each candidate by what fraction of
# its objectives are on the optimistic Pareto front (mu + beta*sigma). This
# naturally balances exploration (high sigma) and exploitation (good mu) across
# the multi-objective electrolyte design space.


def _gpr_model_from_predictor(predictor: Any) -> Any | None:
    """Extract a fitted GPR model from the conformal predictor's delta layer.

    The conformal predictor wraps a DeltaCorrection object that holds the
    sklearn GPR used for residual correction. We extract it here so the BALD
    acquisition can query posterior variance without retraining.

    Returns None if no GPR model is available.
    """
    try:
        delta = getattr(predictor, "_delta_correction", None)
        if delta is None:
            return None
        gpr = getattr(delta, "_gpr_model", None)
        if gpr is None:
            gpr = getattr(delta, "gpr_model", None)
        if gpr is None:
            return None
        if getattr(gpr, "X_train_", None) is None:
            return None
        return gpr
    except Exception:
        return None


def bald_acquisition_score(
    mol: Any,
    gpr_model: Any | None,
) -> float:
    """Compute BALD score for a single candidate molecule.

    For a Gaussian posterior, BALD = H[p(y|x,D)] - E[H[p(y|x,theta)]]
    reduces monotonically to the epistemic variance sigma*^2(x) / sigma_noise^2.
    We return the normalised posterior std so the score is in [0, 1].

    Uses MLX GPR surrogate if available (fast GPU path), else sklearn.
    Returns 0.0 when no GPR model is available.
    """
    if gpr_model is None:
        return 0.0

    from rdkit import Chem

    from aurelius.scoring.oracle.mlx_surrogate import predict_deltas_batch_mlx

    smiles = Chem.MolToSmiles(mol)
    mol_obj = Chem.MolFromSmiles(smiles)
    if mol_obj is None:
        return 0.0

    try:
        _, stds = predict_deltas_batch_mlx(gpr_model, [mol_obj], return_std=True)
    except Exception:
        return 0.0

    if stds is None or len(stds) == 0:
        return 0.0

    raw_score = float(stds[0])
    return raw_score / (1.0 + raw_score)


_PARETO_ORIENTATION: dict[str, float] = {
    "homo": -1.0,
    "lumo": -1.0,
    "dielectric": 1.0,
    "viscosity": -1.0,
}


def _valid_pareto_props(
    predictions: dict[str, float], uncertainties: dict[str, float]
) -> list[str]:
    """Properties present in both predictions and uncertainties."""
    return [p for p in _PARETO_ORIENTATION if p in predictions and p in uncertainties]


def _ucb_values(
    predictions: dict[str, float],
    uncertainties: dict[str, float],
    props: list[str],
    beta: float,
) -> list[float]:
    """Compute UCB (mu + beta*sigma) per oriented objective."""
    return [
        _PARETO_ORIENTATION[p] * predictions.get(p, 0.0)
        + beta * abs(uncertainties.get(p, 0.0))
        for p in props
    ]


def _is_dominated(this_ucb: list[float], other_ucb: list[float]) -> bool:
    """True if *other_ucb* dominates *this_ucb*: >= on all objectives, > on one."""
    return all(o >= t for o, t in zip(other_ucb, this_ucb, strict=True)) and any(
        o > t for o, t in zip(other_ucb, this_ucb, strict=True)
    )


def pareto_ucb_score(
    predictions: dict[str, float],
    uncertainties: dict[str, float],
    all_predictions: list[dict[str, float]],
    beta: float = _PARETO_UCB_BETA,
) -> float:
    """Pareto-front score under UCB orientation.

    For each objective, compute the UCB: mu + beta * sigma, oriented so
    higher is better. A candidate is on the Pareto front if no other candidate
    dominates it across all UCB values. The score is 1.0 if Pareto-optimal,
    0.0 otherwise.

    Objectives are oriented so that higher is always better:
      - HOMO: negated (high |homo| = stable, hard to oxidise)
      - LUMO: negated (low LUMO = easily reduced; we want high LUMO = stable)
      - dielectric: as-is (higher = better solvation)
      - viscosity: negated (lower = better ion transport)

    Returns a value in [0, 1].
    """
    valid_props = _valid_pareto_props(predictions, uncertainties)
    if not valid_props or not all_predictions:
        return 0.0

    if not uncertainties:
        uncertainties = {p: 0.0 for p in valid_props}

    this_ucb = _ucb_values(predictions, uncertainties, valid_props, beta)

    for other in all_predictions:
        if other is predictions:
            continue
        other_ucb = _ucb_values(other, uncertainties, valid_props, beta)
        if _is_dominated(this_ucb, other_ucb):
            return 0.0

    return 1.0


def _compute_bald_scores(
    evaluated: list[tuple[Any, dict[str, float], float, float, str]],
    gpr_model: Any | None,
) -> dict[str, float]:
    """Compute BALD scores for all evaluated candidates.

    Uses the MLX GPR surrogate to batch-predict posterior variance.
    Returns a dict mapping canonical SMILES to BALD score.
    """
    from rdkit import Chem

    if gpr_model is None or not evaluated:
        return {}

    mols = [ctx.mol for ctx, _, *_ in evaluated]
    smiles_keys = [Chem.MolToSmiles(ctx.mol) for ctx, _, *_ in evaluated]

    try:
        from aurelius.scoring.oracle.mlx_surrogate import MLXGPRSurrogate

        surrogate = MLXGPRSurrogate(gpr_model)
        if surrogate.is_available:
            _, stds = surrogate.predict_batch(mols, return_std=True)
            if stds is not None:
                return {
                    smi: std / (1.0 + std) for smi, std in zip(smiles_keys, stds, strict=False)
                }
    except Exception:
        pass

    try:


        # Fallback: approximate BALD via the uncertainty term from the conformal
        # interval half-width, normalised to [0, 1].
        from aurelius.scoring.oracle.conformal import get_conformal_predictor

        predictor = get_conformal_predictor()
        result = {}
        for ctx, preds, _, _, _ in evaluated:
            total_width = 0.0
            n_props = 0
            for prop in MEASURABLE_PROPERTIES:
                _, interval = predictor.predict_interval(prop, preds.get(prop, 0.0), mol=ctx.mol)
                total_width += interval[1] - interval[0]
                n_props += 1
            avg_width = total_width / max(n_props, 1)
            result[Chem.MolToSmiles(ctx.mol)] = avg_width / (1.0 + avg_width)
        return result
    except Exception:
        return {}


def _compute_per_property_uncertainty(
    evaluated: list[tuple[Any, dict[str, float], float, float, str]],
    predictor: Any,
    properties: list[str],
) -> dict[str, dict[str, float]]:
    """Compute per-property uncertainty (interval half-width) for Pareto UCB.

    Returns a dict mapping canonical SMILES to {prop: uncertainty} dicts.
    """
    from rdkit import Chem

    result: dict[str, dict[str, float]] = {}
    for ctx, preds, _, _, _ in evaluated:
        smi = Chem.MolToSmiles(ctx.mol)
        unc: dict[str, float] = {}
        for prop in properties:
            try:
                _, interval = predictor.predict_interval(
                    prop, preds.get(prop, 0.0), mol=ctx.mol
                )
                unc[prop] = (interval[1] - interval[0]) / 2.0
            except Exception:
                unc[prop] = 0.0
        result[smi] = unc
    return result


# Each additional suggestion for an already-chosen molecule or property is
# discounted by this factor, compounding. 0.6 is strong enough to interleave
# properties without ever letting a weak suggestion outrank a much stronger one.
_REDUNDANCY_DISCOUNT = 0.6

# Weight of the structural-redundancy penalty (ADR-2026-08-08-06). A candidate
# whose scaffold is identical to one already in the batch keeps only
# (1 - _SIMILARITY_LAMBDA) of its score. Matches the diversity_lambda
# convention used by ``selection.tournament_select``.
_SIMILARITY_LAMBDA = 0.7


def _structural_penalty(fp: Any, chosen_fps: list[Any]) -> float:
    """Discount a candidate by its similarity to molecules already in the batch.

    Returns ``1 - _SIMILARITY_LAMBDA * max_tanimoto`` against the batch, so an
    exact structural duplicate keeps 0.3 of its score and a wholly unrelated
    scaffold keeps all of it.
    """
    if not chosen_fps or fp is None:
        return 1.0
    from rdkit import DataStructs

    max_sim = float(max(DataStructs.BulkTanimotoSimilarity(fp, chosen_fps)))
    return 1.0 - _SIMILARITY_LAMBDA * max_sim


def _diversify(
    ranked: list[ExperimentSuggestion],
    top_n: int,
    fingerprints: dict[str, Any] | None = None,
) -> list[ExperimentSuggestion]:
    """Greedily pick a worklist that is not four copies of the same experiment.

    Ranking each (molecule, property) pair independently produces a list whose
    top entries are near-duplicates: the same property on similar scaffolds,
    because the terms that drive the score vary smoothly across chemistry.
    A batch of measurements is only worth its *joint* information, and
    near-duplicate measurements have highly correlated residuals, so the
    marginal value of the second one is much lower than its standalone score
    suggests.

    Two redundancy signals are combined, both standard greedy approximations
    to batch-mode active learning:

    1. **Exact repetition** — a compounding discount for molecules and
       properties already represented in the batch.
    2. **Structural redundancy** (ADR-2026-08-08-06) — a Tanimoto penalty
       against the scaffolds already chosen, when ``fingerprints`` is
       supplied.

    Signal 1 alone is insufficient and was the measured defect: it treats two
    distinct-but-near-identical homologues as fully independent, so the batch
    filled up with one scaffold family. The suggester's ``novelty`` term does
    not help, because it measures distance to the *calibration set*, which is
    identical for every member of such a family and therefore cannot separate
    them. Empirically the batch was more redundant than random sampling
    (mean pairwise Tanimoto 0.132 vs 0.077 at k=10).

    This is a re-ordering only: nothing new enters the list, and the adjusted
    score is recorded on each suggestion so the effect is auditable.
    """
    selected: list[ExperimentSuggestion] = []
    remaining = list(ranked)
    seen_molecules: dict[str, int] = {}
    seen_properties: dict[str, int] = {}
    chosen_fps: list[Any] = []

    while remaining and len(selected) < top_n:
        best_index, best_value = 0, -1.0
        for i, candidate in enumerate(remaining):
            penalty = _REDUNDANCY_DISCOUNT ** (
                seen_molecules.get(candidate.smiles, 0)
                + seen_properties.get(candidate.property_to_measure, 0)
            )
            value = candidate.priority_score * penalty
            if fingerprints is not None:
                value *= _structural_penalty(
                    fingerprints.get(candidate.smiles), chosen_fps
                )
            if value > best_value:
                best_index, best_value = i, value
        chosen = remaining.pop(best_index)
        seen_molecules[chosen.smiles] = seen_molecules.get(chosen.smiles, 0) + 1
        seen_properties[chosen.property_to_measure] = (
            seen_properties.get(chosen.property_to_measure, 0) + 1
        )
        if fingerprints is not None and chosen.smiles not in {
            s.smiles for s in selected
        }:
            fp = fingerprints.get(chosen.smiles)
            if fp is not None:
                chosen_fps.append(fp)
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
