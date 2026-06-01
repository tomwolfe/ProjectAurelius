"""Evolutionary Tournament Selection with Tanimoto diversity penalty.

Replaces the over-engineered Random Forest + Expected Improvement surrogate
with a direct, interpretable selection strategy:

1. Evaluate candidates using the actual Oracle (no surrogate needed).
2. Select top performers via tournament selection.
3. Apply a Tanimoto-based diversity penalty to prevent batch collapse.

This approach is simpler, faster, and works naturally for both cheap (TOM+GC)
and expensive (xTB) oracles — just adjust ``batch_size``.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
from rdkit.DataStructs import BulkTanimotoSimilarity, TanimotoSimilarity

from aurelius.types import MoleculeContext

log = logging.getLogger(__name__)


def tournament_select(
    contexts: list[MoleculeContext],
    scores: list[float],
    batch_size: int = 10,
    tournament_size: int = 3,
    diversity_lambda: float = 0.3,
    rng_seed: int = 42,
) -> list[MoleculeContext]:
    """Select a diverse batch of candidates using tournament selection.

    Each round picks ``tournament_size`` random candidates and selects the
    best-scoring one, then applies a Tanimoto diversity penalty to avoid
    selecting near-identical molecules.

    Args:
        contexts: List of candidate MoleculeContext objects.
        scores: Corresponding Oracle scores (same length as contexts).
        batch_size: Number of candidates to select.
        tournament_size: Number of candidates in each tournament.
        diversity_lambda: Strength of the diversity penalty (0 = none, 1 = max).
        rng_seed: Random seed for reproducibility.

    Returns:
        List of selected MoleculeContext objects (length <= batch_size).
    """
    n = len(contexts)
    if n == 0:
        return []
    if n <= batch_size:
        return list(contexts)

    rng = random.Random(rng_seed)
    selected: list[MoleculeContext] = []
    selected_fps: list[Any] = []
    used_indices: set[int] = set()

    for _ in range(min(batch_size, n)):
        pool = [i for i in range(n) if i not in used_indices]
        if not pool:
            break

        tournament = rng.sample(pool, min(tournament_size, len(pool)))
        best_idx = max(tournament, key=lambda i: scores[i])
        best_adj = scores[best_idx]

        if selected_fps:
            fp_best = contexts[best_idx].get_ecfp4()
            max_sim_best = max(
                TanimotoSimilarity(fp_best, sfp) for sfp in selected_fps
            )
            best_adj = scores[best_idx] * (1.0 - diversity_lambda * max_sim_best)

            for i in tournament:
                if i in used_indices:
                    continue
                fp_i = contexts[i].get_ecfp4()
                max_sim_i = max(
                    TanimotoSimilarity(fp_i, sfp) for sfp in selected_fps
                )
                adj = scores[i] * (1.0 - diversity_lambda * max_sim_i)
                if adj > best_adj:
                    best_adj = adj
                    best_idx = i

        used_indices.add(best_idx)
        ctx = contexts[best_idx]
        selected.append(ctx)
        selected_fps.append(ctx.get_ecfp4())

    return selected


def compute_pairwise_diversity(contexts: list[MoleculeContext]) -> float:
    """Compute mean Tanimoto dissimilarity (1 - similarity) across contexts."""
    if len(contexts) < 2:
        return 0.0

    fps = [ctx.get_ecfp4() for ctx in contexts]
    similarities: list[float] = []
    for i, fp_i in enumerate(fps):
        sims = BulkTanimotoSimilarity(fp_i, fps[i + 1:])
        similarities.extend(sims)

    if not similarities:
        return 0.0
    return float(1.0 - np.mean(similarities))
