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


def _ucb_score(
    idx: int,
    scores: list[float],
    uncertainties: list[float] | None,
    exploration_beta: float,
) -> float:
    """Upper Confidence Bound acquisition score.

    UCB = mean_score + beta * uncertainty_std

    When exploration mode is active (exploration_beta > 0), candidates with
    high epistemic uncertainty receive a boost relative to their raw score,
    biasing selection toward molecules the model knows least about. This
    maximises information gain for wet-lab validation.

    When uncertainties are unavailable, falls back to raw score.
    """
    if uncertainties is None or idx >= len(uncertainties):
        return scores[idx]
    uncertainty = uncertainties[idx]
    if uncertainty <= 0.0:
        return scores[idx]
    return scores[idx] + exploration_beta * uncertainty


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
    exploration_mode: bool = False,
    uncertainties: list[float] | None = None,
    exploration_beta: float = 0.5,
) -> list[MoleculeContext]:
    """Select a diverse batch of candidates using tournament selection.

    Each round picks ``tournament_size`` random candidates and selects the
    best-scoring one, then applies a Tanimoto diversity penalty to avoid
    selecting near-identical molecules.

    When ``exploration_mode`` is True, uses an Upper Confidence Bound (UCB)
    acquisition function (score + beta * uncertainty) instead of raw scores,
    prioritising candidates with high epistemic uncertainty to maximise
    information gain for wet-lab validation. The ``uncertainties`` list
    should contain combined UQ standard deviations per candidate (e.g.,
    the mean of normalised dielectic_std and viscosity_std from
    GcUqEnsemble).
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

    # Use UCB acquisition when in exploration mode
    effective_scores: list[float] = (
        [_ucb_score(i, scores, uncertainties, exploration_beta) for i in range(len(scores))]
        if exploration_mode and uncertainties is not None
        else scores
    )

    for _ in range(min(batch_size, n)):
        pool = [i for i in range(n) if i not in used_indices]
        if not pool:
            break

        tournament = rng.sample(pool, min(tournament_size, len(pool)))
        best_idx, _ = _best_in_tournament(
            tournament, effective_scores, fps_list, selected_fps, diversity_lambda,
        )

        used_indices.add(best_idx)
        selected.append(contexts[best_idx])
        selected_fps.append(fps_list[best_idx])

    return selected


def _extract_objectives(r: Any) -> tuple[float, float, float] | None:
    """Extract (lumo, dielectric, -viscosity) tuple from a result object or dict."""
    try:
        if hasattr(r, 'lumo_eV') and hasattr(r, 'dielectric_proxy') and hasattr(r, 'viscosity_proxy'):
            return (float(r.lumo_eV), float(r.dielectric_proxy), -float(r.viscosity_proxy))
    except (AttributeError, TypeError):
        pass
    try:
        lumo = r.get('lumo_eV', -99.0)
        diel = r.get('dielectric_proxy', 0.0)
        visc = r.get('viscosity_proxy', 99.0)
        if lumo is not None and diel is not None and visc is not None:
            return (float(lumo), float(diel), -float(visc))
    except (AttributeError, TypeError):
        pass
    return None


def _dominates(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
    """Returns True if a dominates b (a >= b in all objectives, > in at least one)."""
    return all(ai >= bi for ai, bi in zip(a, b, strict=False)) and any(
        ai > bi for ai, bi in zip(a, b, strict=False)
    )


def _non_dominated_indices(objectives: list[tuple[float, float, float]]) -> set[int]:
    """Return indices of non-dominated solutions."""
    pareto: set[int] = set()
    for i in range(len(objectives)):
        dominated = False
        for j in range(len(objectives)):
            if i != j and _dominates(objectives[j], objectives[i]):
                dominated = True
                break
        if not dominated:
            pareto.add(i)
    return pareto


def extract_pareto_front(results: list[Any]) -> list[Any]:
    """Identify Pareto-optimal solutions from screening results.

    Objectives (all to be maximized — viscosity is negated):
      1. Maximise lumo_eV (SEI formation)
      2. Maximise dielectric_proxy (salt dissociation)
      3. Maximise -viscosity_proxy (low viscosity = high ion mobility)

    Uses a simple non-dominated sorting algorithm (pure Python, O(n²)).
    Returns the subset of results that are Pareto-optimal (non-dominated).
    """
    if not results:
        return []

    obj_list: list[tuple[float, float, float]] = []
    valid_indices: list[int] = []
    for i, r in enumerate(results):
        obj = _extract_objectives(r)
        if obj is not None:
            obj_list.append(obj)
            valid_indices.append(i)

    pareto_indices = _non_dominated_indices(obj_list)
    return [results[valid_indices[i]] for i in sorted(pareto_indices)]


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
