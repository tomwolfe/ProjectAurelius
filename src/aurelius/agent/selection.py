"""Evolutionary Tournament Selection with Tanimoto diversity penalty.

Direct, interpretable selection strategy:

1. Evaluate candidates directly using the Oracle.
2. Select top performers via tournament selection.
3. Apply a Tanimoto-based diversity penalty to prevent batch collapse.

This approach is simple, fast, and works naturally for both cheap (TOM+GC)
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


def _adjusted_score(idx: int, scores: list[float], fps_list: list[Any], selected_fps: list[Any], diversity_lambda: float) -> float:
    """Compute diversity-penalised score for a candidate."""
    if not selected_fps:
        return scores[idx]
    fp = fps_list[idx]
    max_sim = max(TanimotoSimilarity(fp, sfp) for sfp in selected_fps)
    return scores[idx] * (1.0 - diversity_lambda * max_sim)


def _best_in_tournament(
    tournament: list[int],
    scores: list[float],
    fps_list: list[Any],
    selected_fps: list[Any],
    diversity_lambda: float,
) -> tuple[int, float]:
    """Find the best candidate in a tournament, adjusted for diversity."""
    best_idx = max(tournament, key=lambda i: scores[i])
    best_adj = _adjusted_score(best_idx, scores, fps_list, selected_fps, diversity_lambda)

    for i in tournament:
        adj = _adjusted_score(i, scores, fps_list, selected_fps, diversity_lambda)
        if adj > best_adj:
            best_adj = adj
            best_idx = i
    return best_idx, best_adj


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
    """
    n = len(contexts)
    if n == 0:
        return []
    if n <= batch_size:
        return list(contexts)

    rng = random.Random(rng_seed)
    fps_list = [ctx.get_ecfp4() for ctx in contexts]
    selected: list[MoleculeContext] = []
    selected_fps: list[Any] = []
    used_indices: set[int] = set()

    for _ in range(min(batch_size, n)):
        pool = [i for i in range(n) if i not in used_indices]
        if not pool:
            break

        tournament = rng.sample(pool, min(tournament_size, len(pool)))
        best_idx, _ = _best_in_tournament(tournament, scores, fps_list, selected_fps, diversity_lambda)

        used_indices.add(best_idx)
        selected.append(contexts[best_idx])
        selected_fps.append(fps_list[best_idx])

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
