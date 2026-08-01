"""Multi-objective scoring functions for the Aurelius pipeline.

Extracted from pipeline.py to improve modularity.  Contains the
declarative objectives list, the composite scoring logic, penalty
functions, and human-readable rejection-reason builder.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rdkit import Chem

from aurelius.constants import (
    AL_CORROSION_LUMO_THRESHOLD,
    AL_CORROSION_MIN_FLUORINE,
    AL_CORROSION_PENALTY_FACTOR,
    CED_SIGMOID_STEEPNESS,
    CED_TARGET,
    DIELECTRIC_TARGET,
    DISCOVERY_THRESHOLD,
    HOMO_THRESHOLD,
    HYDROLYSIS_RISK_THRESHOLD,
    LI_BINDING_ENERGY_SIGMA,
    LI_BINDING_ENERGY_TARGET,
    LI_SOLVATION_TARGET,
    LUMO_TARGET,
    SA_THRESHOLD,
    SCORE_WEIGHT_CED,
    SCORE_WEIGHT_DIELECTRIC,
    SCORE_WEIGHT_GAS_EVOLUTION,
    SCORE_WEIGHT_HOMO,
    SCORE_WEIGHT_HYDROLYSIS,
    SCORE_WEIGHT_LI_BINDING,
    SCORE_WEIGHT_LI_SOLVATION,
    SCORE_WEIGHT_LUMO,
    SCORE_WEIGHT_SA,
    SCORE_WEIGHT_SEI_FRACTURE,
    SCORE_WEIGHT_VISCOSITY,
    SEI_FRACTURE_SIGMOID_STEEPNESS,
    SEI_FRACTURE_TARGET,
    VIABILITY_THRESHOLD,
    VISCOSITY_THRESHOLD,
)
from aurelius.constants import (
    CARBONYL_F_PATTERN as _CARBONYL_F_PATTERN,
)
from aurelius.constants import (
    CF3_PATTERN as _CF3_PATTERN,
)
from aurelius.constants import (
    HYDROLYTICALLY_UNSTABLE_PATTERNS as _HYDRO_PATTERNS,
)
from aurelius.constants import (
    HYPOFLUORITE_PATTERN as _HYPOFLUORITE_PATTERN,
)
from aurelius.constants import (
    HYPOFLUORITE_PENALTY_FACTOR as _HYPOFLUORITE_PENALTY,
)
from aurelius.constants import (
    SULFONYL_F_PATTERN as _SULFONYL_F_PATTERN,
)
from aurelius.scoring.oracle import (
    _GC_FRAGMENTS,
    _count_fragments,
    _saturate_contrib,
)
from aurelius.types import MoleculeContext
from aurelius.utils.chem_utils import electrolyte_synthetic_accessibility

logger = logging.getLogger(__name__)


def _gaussian(value: float, target: float, sigma: float) -> float:
    return math.exp(-0.5 * ((value - target) / sigma) ** 2)


def _sigmoid(value: float, target: float, steepness: float, higher_is_better: bool = True) -> float:
    if higher_is_better:
        return 1.0 / (1.0 + math.exp(-steepness * (value - target)))
    return 1.0 / (1.0 + math.exp(steepness * (value - target)))


@dataclass
class Objective:
    """A single scoring objective with a direct callable function.

    Each objective defines how a raw property value is converted to a
    sub-score via a callable function, then weighted in the final composite.
    """

    name: str
    property_key: str
    weight: float
    function: Callable[[float], float]
    failure_reason_template: str = "{name}={value:.3f} (below threshold)"

    def __call__(self, value: float) -> float:
        return self.function(value)


_OBJECTIVES: list[Objective] = [
    Objective("lumo_reward", "lumo_eV", SCORE_WEIGHT_LUMO,
              lambda v: _gaussian(v, LUMO_TARGET, 0.75),
              failure_reason_template="LUMO={value:.3f}eV (poor SEI formation)"),
    Objective("homo_penalty", "homo_eV", SCORE_WEIGHT_HOMO,
              lambda v: _sigmoid(v, HOMO_THRESHOLD, 5.0, False),
              failure_reason_template="HOMO={value:.3f}eV (oxidative instability)"),
    Objective("dielectric_reward", "dielectric_proxy", SCORE_WEIGHT_DIELECTRIC,
              lambda v: _sigmoid(v, DIELECTRIC_TARGET, 1.5),
              failure_reason_template="dielectric_proxy={value:.3f} (poor salt dissolution)"),
    Objective("viscosity_penalty", "viscosity_proxy", SCORE_WEIGHT_VISCOSITY,
              lambda v: _sigmoid(v, VISCOSITY_THRESHOLD, 2.0, False),
              failure_reason_template="viscosity_proxy={value:.3f} (poor ion mobility)"),
    Objective("li_solvation_reward", "li_solvation_proxy", SCORE_WEIGHT_LI_SOLVATION,
              lambda v: _gaussian(v, LI_SOLVATION_TARGET, 1.0),
              failure_reason_template="li_solvation_proxy={value:.3f} (poor Li+ binding)"),
    Objective("li_binding_reward", "li_binding_energy_kcal", SCORE_WEIGHT_LI_BINDING,
              lambda v: _gaussian(v, LI_BINDING_ENERGY_TARGET, LI_BINDING_ENERGY_SIGMA),
              failure_reason_template="li_binding_energy_kcal={value:.3f} (unstable Li+ coordination)"),
    Objective("ced_reward", "ced_proxy", SCORE_WEIGHT_CED,
              lambda v: _sigmoid(v, CED_TARGET, CED_SIGMOID_STEEPNESS),
              failure_reason_template="CED proxy={value:.3f} (poor SEI mechanical robustness)"),
    Objective("sei_fracture_reward", "sei_fracture_toughness_proxy", SCORE_WEIGHT_SEI_FRACTURE,
              lambda v: _sigmoid(v, SEI_FRACTURE_TARGET, SEI_FRACTURE_SIGMOID_STEEPNESS),
              failure_reason_template="SEI fracture proxy={value:.3f} (poor SEI mechanical robustness)"),
    Objective("sa_penalty", "sa_score", SCORE_WEIGHT_SA,
              lambda v: _sigmoid(v, SA_THRESHOLD, 2.0, False),
              failure_reason_template="SA score={value:.2f} (hard to synthesize)"),
    Objective("gas_evolution_penalty", "gas_evolution_proxy", SCORE_WEIGHT_GAS_EVOLUTION,
              lambda v: _sigmoid(v, 0.5, 2.0, False),
              failure_reason_template="gas_evolution_proxy={value:.3f} (high degradation risk)"),
    Objective("hydrolysis_penalty", "hydrolysis_risk_proxy", SCORE_WEIGHT_HYDROLYSIS,
              lambda v: _sigmoid(v, HYDROLYSIS_RISK_THRESHOLD, 3.0, False),
              failure_reason_template="hydrolysis_risk_proxy={value:.3f} (high hydrolysis risk)"),
]


def _check_hydrolytic_instability(mol: Chem.Mol) -> float:
    """Check for hydrolytically unstable motifs.

    Returns a multiplier in [0.5, 1.0].
    """
    penalty = 1.0
    for pattern, name, severity in _HYDRO_PATTERNS:
        if pattern is not None and mol.HasSubstructMatch(pattern):
            penalty *= (1.0 - severity)
            logger.debug("Hydrolytic instability detected: %s (penalty %.2f)", name, severity)
    return max(penalty, 0.5)


def _check_hypofluorite_instability(mol: Chem.Mol) -> float:
    """Penalise molecules with O-F (hypofluorite) bonds.

    Hypofluorites are violently reactive oxidisers — they decompose
    exothermically at room temperature and cannot be used as battery
    electrolyte solvents.

    Returns a multiplier in [0.50, 1.0].
    """
    if _HYPOFLUORITE_PATTERN is not None and mol.HasSubstructMatch(_HYPOFLUORITE_PATTERN):
        return _HYPOFLUORITE_PENALTY
    return 1.0


def _check_al_corrosion_risk(mol: Chem.Mol) -> float:
    """Check for Al corrosion risk in high-LUMO fluorinated molecules.

    Returns a penalty multiplier in [0.7, 1.0].
    """
    n_f = sum(a.GetAtomicNum() == 9 for a in mol.GetAtoms())
    n_cf3 = len(mol.GetSubstructMatches(_CF3_PATTERN))
    n_f_ewg = len(mol.GetSubstructMatches(_CARBONYL_F_PATTERN))
    n_f_ewg += len(mol.GetSubstructMatches(_SULFONYL_F_PATTERN))
    if n_f >= AL_CORROSION_MIN_FLUORINE and (n_cf3 >= 1 or n_f_ewg >= 1):
        return AL_CORROSION_PENALTY_FACTOR
    return 1.0


def _check_building_block_grounding(mol: Chem.Mol) -> float:
    """Penalty for molecules with BRICS fragments not matching commercial precursors.

    Returns a multiplier in [0.7, 1.0].
    """
    from aurelius.agent.mutation.brics import combined_grounding_score
    coverage = combined_grounding_score(mol)
    return 0.7 + 0.3 * coverage


def _concept_grounding_score(mol: Chem.Mol) -> int:
    """Count how many distinct concepts from the concept library a molecule matches.

    A concept is considered matched if the molecule contains the concept's SMARTS
    pattern as a substructure. Returns the number of distinct concepts matched
    (0-based).

    Physical justification: Electrolyte molecules that preserve multiple known
    functional motifs (cyclic carbonates, fluorinated ethers, sulfones, etc.)
    are more likely to be synthetically accessible and electrochemically viable.
    This coarse-grained grounding score supplements the BRICS-based grounding
    check by rewarding concept retention during mutation.
    """
    from importlib import resources

    package_dir = resources.files("aurelius.data")
    concept_path = package_dir / "concept_library.json"

    try:
        with concept_path.open("r") as fh:
            import json
            library = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return 0

    concepts = library.get("concepts", [])
    matches = 0
    for concept in concepts:
        pattern = Chem.MolFromSmarts(concept["smarts"])
        if pattern is not None and mol.HasSubstructMatch(pattern):
            matches += 1
    return matches


def _apply_penalties(
    total_score: float, lumo_eV: float, ctx: MoleculeContext | None
) -> float:
    """Apply substructure-based penalty multipliers to a raw total score.

    Includes hydrolytic instability, hypofluorite, Al corrosion, and
    building-block grounding penalties.
    """
    al_corrosion_penalty = 1.0
    if ctx is None:
        return total_score
    total_score *= _check_hydrolytic_instability(ctx.mol)
    total_score *= _check_hypofluorite_instability(ctx.mol)
    if lumo_eV > AL_CORROSION_LUMO_THRESHOLD:
        al_corrosion_penalty = _check_al_corrosion_risk(ctx.mol)
    total_score *= al_corrosion_penalty
    total_score *= _check_building_block_grounding(ctx.mol)

    if total_score >= DISCOVERY_THRESHOLD:
        from aurelius.agent.mutation.brics import combined_grounding_score
        grounding = combined_grounding_score(ctx.mol)
        if grounding < 0.6:
            total_score *= 0.8

    return total_score


def _apply_domain_penalty(score: dict[str, Any], t2_result: dict[str, Any] | None) -> dict[str, Any]:
    """Apply domain-of-applicability penalty to the score dict."""
    if t2_result is not None:
        domain_penalty = t2_result.get("domain_penalty", 1.0)
        if domain_penalty < 1.0:
            score["total_score"] *= domain_penalty
            score["domain_penalty_applied"] = domain_penalty
            reason = t2_result.get("domain_reason", "")
            if reason:
                score.setdefault("rejection_reasons", []).append(
                    f"Domain penalty {domain_penalty:.2f}: {reason}"
                )
    score["total_score"] = max(0.0, min(100.0, score["total_score"]))
    score["is_viable"] = score["total_score"] >= VIABILITY_THRESHOLD
    return score


def _build_contributor_suffix(
    obj: Objective,
    counts: dict[str, int] | None,
    prop_idx_map: dict[str, int],
) -> str:
    """Append top-3 GC fragment contributors for a low-scoring objective."""
    contrib_idx = prop_idx_map[obj.property_key]
    contribs: list[tuple[str, float]] = []
    for _, name, *vals in _GC_FRAGMENTS:
        raw_c = vals[contrib_idx - 2]
        n = counts.get(name, 0)
        if n > 0 and abs(raw_c) > 1e-6:
            total = _saturate_contrib(n, raw_c * 2.0)
            contribs.append((name, total))
    contribs.sort(key=lambda x: abs(x[1]), reverse=True)
    top = contribs[:3]
    if not top:
        return ""
    parts = []
    for name, total in top:
        sign = "+" if total >= 0 else ""
        parts.append(f"{name} ({sign}{total:.1f})")
    return " Top contributors: " + ", ".join(parts) + "."


def _build_rejection_reasons(
    total_score: float,
    sub_scores: dict[str, float],
    raw_values: dict[str, float],
    is_viable: bool,
    ctx: MoleculeContext | None,
) -> list[str]:
    """Build human-readable rejection reasons for a failed screening.

    Returns an empty list if the molecule is viable.
    """
    if is_viable:
        return []

    _PROP_GC_IDX: dict[str, int] = {
        "dielectric_proxy": 2,
        "viscosity_proxy": 3,
        "li_solvation_proxy": 4,
        "ced_proxy": 5,
    }

    counts = None
    if ctx is not None:
        counts = _count_fragments(ctx.mol)

    reasons = []
    for obj in _OBJECTIVES:
        s = sub_scores.get(obj.name, 0.0)
        if s < 0.3:
            value = raw_values.get(obj.property_key, 0.0)
            reason = obj.failure_reason_template.format(value=value)
            if counts is not None and obj.property_key in _PROP_GC_IDX:
                reason += _build_contributor_suffix(obj, counts, _PROP_GC_IDX)
            reasons.append(reason)

    if ctx is not None:
        if _check_al_corrosion_risk(ctx.mol) < 1.0:
            reasons.append("Al corrosion risk (high-LUMO fluorinated molecule)")
        if _check_hypofluorite_instability(ctx.mol) < 1.0:
            reasons.append("hypofluorite (O-F) bond — violently reactive")
    return [
        f"Aurelius Score {total_score:.1f} below threshold: {'; '.join(reasons)}"
    ]


def compute_score(
    homo_eV: float = -99.0,
    lumo_eV: float = -99.0,
    dielectric_proxy: float = 0.0,
    viscosity_proxy: float = 99.0,
    li_solvation_proxy: float = 0.0,
    li_binding_energy_kcal: float = 0.0,
    ced_proxy: float = 0.0,
    sei_fracture_toughness_proxy: float = 0.0,
    gas_evolution_proxy: float = 0.0,
    hydrolysis_risk_proxy: float = 0.0,
    ctx: MoleculeContext | None = None,
    quantum_confidence: str = "unknown",
) -> dict[str, Any]:
    """Compute the multi-objective composite Aurelius Score.

    Args:
        See _OBJECTIVES for the full list of weighted objectives.
        quantum_confidence: Confidence level from quantum backend
            ("xtb", "tom_high", or "tom_low").

    Returns:
        Dict with total_score, is_viable, sub_scores, rejection_reasons.
    """
    raw_values: dict[str, float] = {
        "lumo_eV": lumo_eV,
        "homo_eV": homo_eV,
        "dielectric_proxy": dielectric_proxy,
        "viscosity_proxy": viscosity_proxy,
        "li_solvation_proxy": li_solvation_proxy,
        "li_binding_energy_kcal": li_binding_energy_kcal,
        "ced_proxy": ced_proxy,
        "sei_fracture_toughness_proxy": sei_fracture_toughness_proxy,
        "gas_evolution_proxy": gas_evolution_proxy,
        "hydrolysis_risk_proxy": hydrolysis_risk_proxy,
    }

    sub_scores: dict[str, float] = {}
    total_score = 0.0

    sa_score: float = 5.0
    if ctx is not None:
        try:
            sa_score = electrolyte_synthetic_accessibility(ctx)
        except Exception:
            sa_score = 5.0
    raw_values["sa_score"] = sa_score

    for obj in _OBJECTIVES:
        score = obj(raw_values[obj.property_key])
        sub_scores[obj.name] = round(score, 4)
        total_score += obj.weight * score

    total_score *= 100.0

    total_score = _apply_penalties(total_score, lumo_eV, ctx)
    total_score = max(0.0, min(100.0, total_score))

    if quantum_confidence == "tom_low":
        total_score *= 0.85
        total_score = max(0.0, min(100.0, total_score))

    is_viable = total_score >= VIABILITY_THRESHOLD

    rejection_reasons = _build_rejection_reasons(
        total_score, sub_scores, raw_values, is_viable, ctx
    )

    return {
        "total_score": total_score,
        "is_viable": is_viable,
        "sub_scores": sub_scores,
        "sa_score": round(sa_score, 4),
        "rejection_reasons": rejection_reasons,
    }


def format_score(score: dict[str, Any]) -> str:
    """Format a score dict into a short human-readable string."""
    total = score.get("total_score", 0.0)
    viable = score.get("is_viable", False)
    return f"Score: {total:.1f}/100 {'VIABLE' if viable else 'REJECTED'}"
