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
from rdkit.DataStructs import BulkTanimotoSimilarity

from aurelius.types import MoleculeContext

log = logging.getLogger(__name__)


def _fp_to_array(fp: Any, n_bits: int = 2048) -> np.ndarray:
    """Convert an RDKit BitVect to a fixed-size numpy uint8 array."""
    arr = np.zeros(n_bits, dtype=np.uint8)
    for idx in fp.GetOnBits():
        arr[idx] = 1
    return arr


def _batch_max_tanimoto(
    fps_arr: np.ndarray,
    selected_arr: np.ndarray,
) -> np.ndarray:
    """Vectorised max-Tanimoto of each row in fps_arr against rows in selected_arr.

    Both arrays have shape (N, n_bits). Returns an (N,) array where
    entry i is max_j Tanimoto(fps_arr[i], selected_arr[j]).
    """
    if selected_arr.shape[0] == 0:
        return np.zeros(fps_arr.shape[0], dtype=np.float64)
    intersection = np.bitwise_and(fps_arr[:, None, :], selected_arr[None, :, :]).sum(axis=2)
    union = np.bitwise_or(fps_arr[:, None, :], selected_arr[None, :, :]).sum(axis=2)
    sims = np.divide(
        intersection.astype(np.float64),
        union.astype(np.float64),
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > 0,
    )
    return sims.max(axis=1)


def _adjusted_score(
    idx: int,
    scores: list[float],
    max_sims: list[float] | np.ndarray,
    diversity_lambda: float,
) -> float:
    """Compute diversity-penalised score for a candidate.

    Uses pre-computed max-similarity values so the caller can vectorise.
    """
    if len(max_sims) == 0:
        return scores[idx]
    return scores[idx] * (1.0 - diversity_lambda * float(max_sims[idx]))


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
    max_sims: np.ndarray,
    diversity_lambda: float,
) -> tuple[int, float]:
    """Find the best candidate in a tournament, adjusted for diversity.

    ``max_sims`` is a pre-computed (n_candidates,) array with the max
    Tanimoto similarity of each candidate against the currently selected set.
    """
    best_idx = max(tournament, key=lambda i: scores[i])
    best_adj = _adjusted_score(best_idx, scores, max_sims, diversity_lambda)

    for i in tournament:
        adj = _adjusted_score(i, scores, max_sims, diversity_lambda)
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
    # Pre-convert all fingerprints to a single (n, 2048) uint8 array
    fps_arr = np.array([_fp_to_array(ctx.get_ecfp4()) for ctx in contexts], dtype=np.uint8)
    selected: list[MoleculeContext] = []
    selected_rows: list[int] = []
    used_indices: set[int] = set()

    # Use UCB acquisition when in exploration mode
    effective_scores: list[float] = (
        [_ucb_score(i, scores, uncertainties, exploration_beta) for i in range(len(scores))]
        if exploration_mode and uncertainties is not None
        else scores
    )

    # max_sims will be recomputed each round as selected_rows grows
    for _ in range(min(batch_size, n)):
        pool = [i for i in range(n) if i not in used_indices]
        if not pool:
            break

        # Compute max Tanimoto for every pool candidate vs. selected set
        if selected_rows:
            selected_arr = fps_arr[selected_rows]
            pool_arr = fps_arr[pool]
            pool_max_sims = _batch_max_tanimoto(pool_arr, selected_arr)
            # Build full-size max_sims array (default 0 for unselected indices)
            max_sims = np.zeros(n, dtype=np.float64)
            for pi, psi in zip(pool, pool_max_sims):
                max_sims[pi] = psi
        else:
            max_sims = np.zeros(n, dtype=np.float64)

        tournament = rng.sample(pool, min(tournament_size, len(pool)))
        best_idx, _ = _best_in_tournament(
            tournament, effective_scores, max_sims, diversity_lambda,
        )

        used_indices.add(best_idx)
        selected.append(contexts[best_idx])
        selected_rows.append(best_idx)

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


def _normalize_uncertainties(uncertainties: list[float]) -> list[float]:
    """Min-max normalize uncertainties to [0, 1]."""
    if not uncertainties:
        return []
    min_u = min(uncertainties)
    max_u = max(uncertainties)
    if max_u - min_u < 1e-12:
        return [0.5] * len(uncertainties)
    return [(u - min_u) / (max_u - min_u) for u in uncertainties]


def select_for_active_learning(
    contexts: list[MoleculeContext],
    scores: list[float],
    uncertainties: list[float],
    batch_size: int = 10,
    beta: float = 0.5,
) -> list[MoleculeContext]:
    """Select candidates for active learning using UCB acquisition.

    Normalises uncertainties to [0, 1], then computes
    UCB = score + beta * normalised_uncertainty, and returns the top
    ``batch_size`` unique candidates. Prioritises molecules the model
    is most uncertain about (high epistemic uncertainty) for wet-lab
    validation, maximising information gain per experiment.

    Args:
        contexts: Candidate molecules.
        scores: Raw total scores for each candidate.
        uncertainties: Epistemic uncertainty (e.g., combined diel+visc std).
        batch_size: Number of candidates to select.
        beta: Exploration weight. Higher values favour high-uncertainty
            molecules over high-scoring ones.

    Returns:
        Top ``batch_size`` unique candidates ordered by UCB score descending.
    """
    if not contexts or not scores:
        return []

    norm_uncertainties = _normalize_uncertainties(uncertainties)
    ucb_scores = [s + beta * u for s, u in zip(scores, norm_uncertainties, strict=False)]

    # Deduplicate by SMILES while keeping highest UCB per molecule
    seen: dict[str, tuple[float, MoleculeContext]] = {}
    for ctx, ucb in zip(contexts, ucb_scores, strict=False):
        if ctx.smiles not in seen or ucb > seen[ctx.smiles][0]:
            seen[ctx.smiles] = (ucb, ctx)

    ranked = sorted(seen.values(), key=lambda x: -x[0])
    return [ctx for _, ctx in ranked[:batch_size]]


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
