"""Standalone public API for the Aurelius oracle.

Decouples the physics-grounded property oracle from the evolutionary
algorithm so external researchers can use it as a library without
running the discovery loop.

Public interface:
    predict_properties(smiles) -> dict
        Full hybrid quantum/GC property prediction + Aurelius score.

    predict_mixture(components, fractions) -> dict
        N-component mixture prediction using ideal mixing rules with a
        generalized Margules synergy bonus.

    get_domain_applicability(smiles) -> tuple[float, str]
        Domain-of-applicability penalty multiplier and reason string.

Design notes:
  - No circular imports: this module only consumes ``aurelius.pipeline``
    and ``aurelius.scoring.oracle``; nothing imports it back except the CLI.
  - Results match ``AureliusPipeline.screen_molecule()`` exactly because
    ``predict_properties`` routes through the same code path.
  - A single lazily-initialised pipeline instance is shared across calls
    so repeated use reuses the in-memory oracle cache.

Usage:
    >>> from aurelius.oracle_api import predict_properties, predict_mixture
    >>> predict_properties("C1COC(=O)O1")["dielectric_proxy"]
    13.7
"""

from __future__ import annotations

import logging
from typing import Any

from aurelius.pipeline import AureliusPipeline
from aurelius.scoring.oracle import (
    mixture_synergy_bonus_n,
    predict_mixture_dielectric_n,
    predict_mixture_li_solvation_n,
    predict_mixture_viscosity_n,
)
from aurelius.types import MoleculeContext

logger = logging.getLogger(__name__)

_PIPELINE: AureliusPipeline | None = None


def _get_pipeline() -> AureliusPipeline:
    """Return a lazily-initialised shared pipeline instance."""
    global _PIPELINE
    if _PIPELINE is None:
        pipeline = AureliusPipeline()
        pipeline.initialize()
        _PIPELINE = pipeline
    return _PIPELINE


def reset_pipeline() -> None:
    """Drop the cached pipeline (primarily for tests)."""
    global _PIPELINE
    _PIPELINE = None


def predict_properties(smiles: str) -> dict[str, Any]:
    """Predict all properties for a single molecule via the full pipeline.

    Args:
        smiles: Valid SMILES string.

    Returns:
        Dict with the oracle tier-2 properties (HOMO, LUMO, dielectric,
        viscosity, Li+ solvation, domain of applicability) plus the
        composite Aurelius score.

    Raises:
        ValueError: If the SMILES cannot be parsed.
    """
    ctx = MoleculeContext.from_smiles(smiles)
    if ctx is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    result = _get_pipeline().screen_molecule(ctx)
    tier2 = result.get("tier2") or {}
    score = result.get("score") or {}

    out: dict[str, Any] = dict(tier2)
    out["smiles"] = ctx.smiles
    out["total_score"] = score.get("total_score", 0.0)
    out["is_viable"] = score.get("is_viable", False)
    out["rejection_reasons"] = score.get("rejection_reasons", [])
    return out


def get_domain_applicability(smiles: str) -> tuple[float, str]:
    """Return the domain-of-applicability multiplier and reason for a molecule.

    The multiplier lies in [0.70, 1.0] where 1.0 means the prediction is
    fully inside the calibrated domain. Penalties accumulate from the
    quantum (TOM) and GC axes via continuous sigmoids.

    Args:
        smiles: Valid SMILES string.

    Returns:
        (penalty_multiplier, reason_string)
    """
    props = predict_properties(smiles)
    return (
        float(props.get("domain_penalty", 1.0)),
        str(props.get("domain_reason", "within domain")),
    )


def _validate_mixture_inputs(
    components: list[str], fractions: list[float]
) -> None:
    """Validate mixture inputs, raising ValueError on malformed input."""
    if len(components) < 2:
        raise ValueError("predict_mixture requires at least 2 components")
    if len(components) != len(fractions):
        raise ValueError(
            f"{len(components)} components but {len(fractions)} fractions"
        )
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {sum(fractions):.6f}")


def _mix_properties(component_results: list[dict]) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float]]:
    """Extract per-component property arrays for mixing."""
    ds = [r.get("dielectric_proxy", 0.0) for r in component_results]
    vs = [r.get("viscosity_proxy", 99.0) for r in component_results]
    ls = [r.get("li_solvation_proxy", 0.0) for r in component_results]
    hs = [r.get("homo_eV", -99.0) for r in component_results]
    lums = [r.get("lumo_eV", -99.0) for r in component_results]
    scores = [r.get("total_score", 0.0) for r in component_results]
    return ds, vs, ls, hs, lums, scores


def predict_mixture(
    components: list[str],
    fractions: list[float],
) -> dict[str, Any]:
    """Predict properties for an N-component mixture (binary or ternary).

    Each component is validated and evaluated individually through the
    same pipeline as a single molecule (anti-gaming: every component must
    parse and pass the tier-1 filter). Mixture bulk properties use ideal
    mixing rules (volume-fraction mean for dielectric/Li+ solvation,
    Grunberg-Nissan log-linear mixing for viscosity) with a generalized
    Margules synergy bonus summed over all complementary component pairs.

    Args:
        components: List of N component SMILES (N >= 2).
        fractions: List of N volume fractions summing to 1.0.

    Returns:
        Dict with per-component results, mixture properties, and score.

    Raises:
        ValueError: If components/fractions lengths mismatch, fractions do
            not sum to 1.0, or any component is an invalid SMILES.
    """
    _validate_mixture_inputs(components, fractions)
    component_results = [predict_properties(s) for s in components]

    ds, vs, ls, hs, lums, scores = _mix_properties(component_results)

    d_mix = predict_mixture_dielectric_n(ds, fractions)
    v_mix = predict_mixture_viscosity_n(vs, fractions)
    ls_mix = predict_mixture_li_solvation_n(ls, fractions)
    h_mix = sum(h * f for h, f in zip(hs, fractions))
    l_mix = sum(l * f for l, f in zip(lums, fractions))
    synergy = mixture_synergy_bonus_n(ds, vs, fractions)

    weighted_base = sum(s * f for s, f in zip(scores, fractions))
    total = min(100.0, weighted_base + synergy)
    is_viable = total >= 50.0

    return {
        "components": list(components),
        "fractions": [round(f, 4) for f in fractions],
        "component_results": component_results,
        "mixture_properties": {
            "dielectric_proxy": round(d_mix, 4),
            "viscosity_proxy": round(v_mix, 4),
            "li_solvation_proxy": round(ls_mix, 4),
            "homo_eV": round(h_mix, 4),
            "lumo_eV": round(l_mix, 4),
            "synergy_bonus": round(synergy, 4),
        },
        "score": {
            "total_score": round(total, 4),
            "is_viable": is_viable,
            "synergy_bonus": round(synergy, 4),
            "weighted_base": round(weighted_base, 4),
            "component_scores": [round(s, 4) for s in scores],
            "rejection_reasons": [],
        },
    }


__all__ = [
    "predict_properties",
    "predict_mixture",
    "get_domain_applicability",
    "reset_pipeline",
]
