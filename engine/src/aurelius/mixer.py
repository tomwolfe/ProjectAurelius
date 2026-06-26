"""Mixture screening logic for binary and ternary electrolyte blends.

Extracted from pipeline.py to improve modularity.  Provides the
mixing-rule and synergy-bonus logic for screening multi-component
electrolyte formulations.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

from aurelius.constants import VIABILITY_THRESHOLD
from aurelius.scoring.oracle import (
    mixture_synergy_bonus,
    mixture_synergy_bonus_ternary,
    predict_mixture_dielectric,
    predict_mixture_li_solvation,
    predict_mixture_viscosity,
)
from aurelius.types import MoleculeContext

logger = logging.getLogger(__name__)


def screen_mixture(
    screen_fn: Callable[[MoleculeContext], dict[str, Any]],
    ctx1: MoleculeContext,
    ctx2: MoleculeContext,
    frac1: float = 0.5,
    ctx3: MoleculeContext | None = None,
    frac2: float | None = None,
) -> dict[str, Any]:
    """Score a binary or ternary mixture using thermodynamic mixing rules with synergy bonus.

    Evaluates each component individually via *screen_fn*, then
    computes mixture dielectric/viscosity using thermodynamic mixing
    rules and applies a non-linear synergy bonus for complementary pairs.

    For ternary blends, evaluates all three binary pairs within the
    mixture and applies the dominant synergy bonus via the extended
    Margules-inspired non-ideal mixing term.

    Args:
        screen_fn: Callable that screens a single MoleculeContext.
        ctx1: First component MoleculeContext.
        ctx2: Second component MoleculeContext.
        frac1: Volume fraction of first component.
        ctx3: Optional third component for ternary mixtures.
        frac2: Volume fraction of second component (only for ternary).

    Returns:
        Dict with component results, mixture properties, and score.
    """
    if ctx3 is not None and frac2 is not None:
        return _screen_ternary_mixture(screen_fn, ctx1, ctx2, ctx3, frac1, frac2)

    res1 = screen_fn(ctx1)
    res2 = screen_fn(ctx2)

    p1 = res1.get("tier2", {}) or {}
    p2 = res2.get("tier2", {}) or {}
    f2 = 1.0 - frac1

    d1 = p1.get("dielectric_proxy", 0.0)
    d2 = p2.get("dielectric_proxy", 0.0)
    v1 = p1.get("viscosity_proxy", 99.0)
    v2 = p2.get("viscosity_proxy", 99.0)

    d_mix = predict_mixture_dielectric(d1, d2, frac1)
    v_mix = predict_mixture_viscosity(v1, v2, frac1)
    ls_mix = predict_mixture_li_solvation(
        p1.get("li_solvation_proxy", 0.0),
        p2.get("li_solvation_proxy", 0.0),
        frac1,
    )
    h_mix = frac1 * p1.get("homo_eV", -99.0) + f2 * p2.get("homo_eV", -99.0)
    l_mix = frac1 * p1.get("lumo_eV", -99.0) + f2 * p2.get("lumo_eV", -99.0)

    synergy = mixture_synergy_bonus(d1, d2, v1, v2, frac1)

    s1 = res1.get("score", {}).get("total_score", 0.0)
    s2 = res2.get("score", {}).get("total_score", 0.0)
    weighted_base = frac1 * s1 + f2 * s2

    total = min(100.0, weighted_base + synergy)
    is_viable = total >= VIABILITY_THRESHOLD

    score: dict[str, Any] = {
        "total_score": total,
        "is_viable": is_viable,
        "synergy_bonus": round(synergy, 4),
        "weighted_base": round(weighted_base, 4),
        "sub_scores": {
            "component1_score": round(s1, 4),
            "component2_score": round(s2, 4),
        },
        "rejection_reasons": [],
    }

    mixture_props: dict[str, float] = {
        "dielectric_proxy": round(d_mix, 4),
        "viscosity_proxy": round(v_mix, 4),
        "li_solvation_proxy": round(ls_mix, 4),
        "homo_eV": round(h_mix, 4),
        "lumo_eV": round(l_mix, 4),
        "synergy_bonus": round(synergy, 4),
    }

    return {
        "component1": res1,
        "component2": res2,
        "mixture_properties": mixture_props,
        "score": score,
    }


def _screen_ternary_mixture(
    screen_fn: Callable[[MoleculeContext], dict[str, Any]],
    ctx1: MoleculeContext,
    ctx2: MoleculeContext,
    ctx3: MoleculeContext,
    frac1: float,
    frac2: float,
) -> dict[str, Any]:
    """Score a ternary mixture with full three-component synergy."""
    res1 = screen_fn(ctx1)
    res2 = screen_fn(ctx2)
    res3 = screen_fn(ctx3)

    p1 = res1.get("tier2", {}) or {}
    p2 = res2.get("tier2", {}) or {}
    p3 = res3.get("tier2", {}) or {}
    frac3 = max(0.0, 1.0 - frac1 - frac2)

    d1 = p1.get("dielectric_proxy", 0.0)
    d2 = p2.get("dielectric_proxy", 0.0)
    d3 = p3.get("dielectric_proxy", 0.0)
    v1 = p1.get("viscosity_proxy", 99.0)
    v2 = p2.get("viscosity_proxy", 99.0)
    v3 = p3.get("viscosity_proxy", 99.0)

    d_mix = frac1 * d1 + frac2 * d2 + frac3 * d3
    ln_v = (
        frac1 * math.log(max(v1, 0.001))
        + frac2 * math.log(max(v2, 0.001))
        + frac3 * math.log(max(v3, 0.001))
    )
    v_mix = math.exp(ln_v)
    ls_mix = (
        frac1 * p1.get("li_solvation_proxy", 0.0)
        + frac2 * p2.get("li_solvation_proxy", 0.0)
        + frac3 * p3.get("li_solvation_proxy", 0.0)
    )
    h_mix = (
        frac1 * p1.get("homo_eV", -99.0)
        + frac2 * p2.get("homo_eV", -99.0)
        + frac3 * p3.get("homo_eV", -99.0)
    )
    l_mix = (
        frac1 * p1.get("lumo_eV", -99.0)
        + frac2 * p2.get("lumo_eV", -99.0)
        + frac3 * p3.get("lumo_eV", -99.0)
    )

    synergy = mixture_synergy_bonus_ternary(d1, d2, d3, v1, v2, v3, frac1, frac2)

    s1 = res1.get("score", {}).get("total_score", 0.0)
    s2 = res2.get("score", {}).get("total_score", 0.0)
    s3 = res3.get("score", {}).get("total_score", 0.0)
    weighted_base = frac1 * s1 + frac2 * s2 + frac3 * s3

    total = min(100.0, weighted_base + synergy)
    is_viable = total >= VIABILITY_THRESHOLD

    score: dict[str, Any] = {
        "total_score": total,
        "is_viable": is_viable,
        "synergy_bonus": round(synergy, 4),
        "weighted_base": round(weighted_base, 4),
        "sub_scores": {
            "component1_score": round(s1, 4),
            "component2_score": round(s2, 4),
            "component3_score": round(s3, 4),
        },
        "rejection_reasons": [],
    }

    mixture_props: dict[str, float] = {
        "dielectric_proxy": round(d_mix, 4),
        "viscosity_proxy": round(v_mix, 4),
        "li_solvation_proxy": round(ls_mix, 4),
        "homo_eV": round(h_mix, 4),
        "lumo_eV": round(l_mix, 4),
        "synergy_bonus": round(synergy, 4),
    }

    return {
        "component1": res1,
        "component2": res2,
        "component3": res3,
        "mixture_properties": mixture_props,
        "score": score,
    }
