"""Post-loop mixture synergy analysis.

Extracted from ``DiscoveryLoop`` into a focused module with a single
responsibility: analysing pairwise synergy of top-scoring discoveries
after the main evolutionary loop has completed.
"""

from __future__ import annotations

import logging
from typing import Any

from aurelius.types import MoleculeContext, ScreeningResult

log = logging.getLogger(__name__)


def analyze_top_mixtures(
    pipeline: Any,
    discoveries: list[ScreeningResult],
) -> list[dict[str, Any]] | None:
    """Analyze pairwise synergy of top 10 discoveries.

    Pairs the highest-scoring molecules and evaluates them as binary
    mixtures to discover synergistic electrolyte blends. Returns the
    top 3 most synergistic mixtures.

    Args:
        pipeline: An ``AureliusPipeline`` instance with ``screen_mixture``.
        discoveries: List of ``ScreeningResult`` from the completed loop.

    Returns:
        Top 3 mixture results sorted by mixture score desc, or None if
        fewer than 2 discoveries exist.
    """
    top_10 = sorted(discoveries, key=lambda r: -r.total_score)[:10]
    if len(top_10) < 2:
        return None

    mixture_results = []
    for i in range(len(top_10)):
        for j in range(i + 1, len(top_10)):
            ctx_i = MoleculeContext.from_smiles(top_10[i].smiles)
            ctx_j = MoleculeContext.from_smiles(top_10[j].smiles)
            if ctx_i is None or ctx_j is None:
                continue
            mix = pipeline.screen_mixture(ctx_i, ctx_j, 0.5)
            mix_score = mix.get("score", {}).get("total_score", 0.0)
            synergy = mix.get("mixture_properties", {}).get("synergy_bonus", 0.0)
            mixture_results.append({
                "component1_smiles": top_10[i].smiles,
                "component2_smiles": top_10[j].smiles,
                "component1_score": top_10[i].total_score,
                "component2_score": top_10[j].total_score,
                "mixture_score": round(mix_score, 4),
                "synergy_bonus": round(synergy, 4),
            })
    mixture_results.sort(key=lambda m: -m["mixture_score"])
    return mixture_results[:3]
