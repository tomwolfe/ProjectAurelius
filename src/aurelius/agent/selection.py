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
    confidences: list[float] | None = None,
) -> tuple[int, float]:
    """Find the best candidate in a tournament, adjusted for diversity and optionally confidence."""
    # Compute adjusted scores with confidence if provided
    adjusted_scores = []
    for i, (score, fp) in enumerate(zip(scores, fps_list, strict=True)):
        conf = confidences[i] if confidences and i < len(confidences) else 1.0
        max_sim = max(TanimotoSimilarity(fp, sfp) for sfp in selected_fps) if selected_fps else 0.0
        adj = score * conf * (1.0 - diversity_lambda * max_sim)
        adjusted_scores.append(adj)

    best_idx = max(tournament, key=lambda i: adjusted_scores[i])
    return best_idx, adjusted_scores[best_idx]


def tournament_select(
    contexts: list[MoleculeContext],
    scores: list[float],
    batch_size: int = 10,
    tournament_size: int = 3,
    diversity_lambda: float = 0.3,
    rng_seed: int = 42,
    confidences: list[float] | None = None,
) -> list[MoleculeContext]:
    """Select a diverse batch of candidates using tournament selection.

    Each round picks ``tournament_size`` random candidates and selects the
    best-scoring one, then applies a Tanimoto diversity penalty to avoid
    selecting near-identical molecules.

    If ``confidences`` is provided, scores are adjusted by confidence:
    ``adjusted_score = score × confidence × (1.0 - diversity_lambda * max_sim)``
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
        best_idx, _ = _best_in_tournament(tournament, scores, fps_list, selected_fps, diversity_lambda, confidences)

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


# ---------------------------------------------------------------------------
# NSGA-II multi-objective selection
# ---------------------------------------------------------------------------

_NSGAIObjectiveSpec = list[tuple[str, str]]
"""List of (key, direction) where key names an entry in ``scores_dict`` and
direction is ``"max"`` or ``"min"``. NSGA-II treats all objectives as
minimisation internally; ``"max"`` objectives are negated before comparison.
"""


def _non_dominated_sort(
    objectives: np.ndarray,
    maximise: np.ndarray,
) -> list[list[int]]:
    """Fast non-dominated sorting (Algorithm 6.1 from Deb et al., 2002).

    Parameters
    ----------
    objectives
        2-D array of shape ``(n_pop, n_obj)``.  Each row is a candidate.
    maximise
        Boolean array of shape ``(n_obj,)``.  ``True`` means the
        corresponding column should be *maximised*.

    Returns
    -------
    list[list[int]]
        A list of fronts (lists of indices into the original population),
        from the first (Pareto-optimal) front down to the last.
    """
    n_pop, n_obj = objectives.shape

    # Convert to minimisation domain: negate columns marked for maximisation.
    adjusted = objectives.copy()
    for j in range(n_obj):
        if maximise[j]:
            adjusted[:, j] = -adjusted[:, j]

    # --- Compute dominance matrix ---
    # ``dominates[i, j]`` is True when solution *i* dominates solution *j*.
    # We only need upper triangle because of symmetry.
    dominated_count = np.zeros(n_pop, dtype=int)
    dominated_set: list[set[int]] = [set() for _ in range(n_pop)]
    dominates: list[set[int]] = [set() for _ in range(n_pop)]

    for i in range(n_pop):
        for j in range(i + 1, n_pop):
            # Compare i and j
            diff = adjusted[i] - adjusted[j]
            i_better = np.all(diff <= 0)  # i is no worse on any objective
            i_strictly = np.any(diff < 0)  # i is strictly better on at least one
            j_better = np.all(-diff <= 0)
            j_strictly = np.any(-diff < 0)

            if i_better and i_strictly:
                dominates[i].add(j)
                dominated_count[j] += 1
                dominated_set[j].add(i)
            elif j_better and j_strictly:
                dominates[j].add(i)
                dominated_count[i] += 1
                dominated_set[i].add(j)

    # --- Build fronts via the standard NSGA-II front extraction ---
    fronts: list[list[int]] = []
    current_front = [i for i in range(n_pop) if dominated_count[i] == 0]

    while current_front:
        fronts.append(current_front)
        next_front: list[int] = []
        for i in current_front:
            for j in dominates[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)
        current_front = next_front

    return fronts


def _crowding_distance(
    front_indices: list[int],
    objectives: np.ndarray,
    maximise: np.ndarray,
) -> np.ndarray:
    """Compute crowding distance for individuals in a single front (Algorithm 6.2).

    Returns a distance array of length ``len(front_indices)`` where ``inf``
    denotes a boundary individual.
    """
    n = len(front_indices)
    distances = np.zeros(n)
    if n <= 2:
        # Boundary individuals get infinite crowding distance.
        return np.full(n, float("inf"))

    # Work in the maximisation domain (larger is better for all objectives).
    adjusted = objectives[front_indices].copy()
    for j in range(objectives.shape[1]):
        if maximise[j]:
            adjusted[:, j] = -adjusted[:, j]

    n_obj = adjusted.shape[1]
    for j in range(n_obj):
        obj_vals = adjusted[:, j]
        sorted_local = np.argsort(obj_vals)
        distances[sorted_local[0]] = float("inf")
        distances[sorted_local[-1]] = float("inf")
        span = obj_vals[sorted_local[-1]] - obj_vals[sorted_local[0]]
        if span == 0:
            continue
        for k in range(1, n - 1):
            distances[sorted_local[k]] += (
                obj_vals[sorted_local[k + 1]] - obj_vals[sorted_local[k - 1]]
            ) / span

    return distances


def nsga2_select(
    contexts: list[MoleculeContext],
    scores_dict: dict[str, list[float]],
    objectives: _NSGAIObjectiveSpec,
    batch_size: int,
    rng_seed: int = 42,
    confidences: list[float] | None = None,
    top_fraction: float = 0.5,
) -> list[MoleculeContext]:
    """Select a diverse, Pareto-optimal batch using NSGA-II.

    Unlike single-objective tournament selection, this method finds
    non-dominated solutions across *multiple* properties simultaneously,
    preserving diversity via crowding distance.  It is useful when the
    weighted composite score may hide trade-offs that a human chemist
    would want to explore.

    Parameters
    ----------
    contexts
        Candidate molecules.
    scores_dict
        Mapping from objective key (e.g. ``"dielectric_proxy"``) to a
        list of values, one per candidate.  Must be aligned with
        ``contexts``.
    objectives
        List of ``(key, direction)`` tuples where *key* indexes into
        ``scores_dict`` and *direction* is ``"max"`` or ``"min"``.
    batch_size
        Number of candidates to select.
    rng_seed
        Seed for any tie-breaking randomness.
    confidences
        Optional conformal-confidence multipliers in ``[0, 1]``.
        Higher confidence → candidates more likely to be retained.
        Applied as an additional (maximise) objective.
    top_fraction
        When the first front exceeds remaining capacity, use this
        fraction of the front with the highest crowding distances
        (the rest are filled by subsequent fronts).  Values in ``(0, 1]``.

    Returns
    -------
    list[MoleculeContext]
        Selected candidates, ordered by front rank then crowding
        distance (descending).
    """
    n = len(contexts)
    if n == 0:
        return []
    if n <= batch_size:
        return list(contexts)

    # Assemble the objective matrix.
    obj_columns = []
    maximise = []
    for key, direction in objectives:
        col = np.asarray(scores_dict[key], dtype=float)
        obj_columns.append(col)
        maximise.append(direction == "max")

    # Optionally add conformal confidence as a final maximise objective.
    if confidences is not None:
        conf_arr = np.asarray(confidences, dtype=float)
        obj_columns.append(conf_arr)
        maximise.append(True)

    obj_matrix = np.column_stack(obj_columns)
    maximise_arr = np.array(maximise, dtype=bool)

    # --- Fast non-dominated sorting ---
    fronts = _non_dominated_sort(obj_matrix, maximise_arr)

    # --- Build ranked list: (front_index, crowding_dist, candidate_index) ---
    ranked: list[tuple[int, float, int]] = []
    for f_idx, front in enumerate(fronts):
        crowding = _crowding_distance(front, obj_matrix, maximise_arr)
        for local_idx, global_idx in enumerate(front):
            ranked.append((f_idx, crowding[local_idx], global_idx))

    # Sort: lower front index first; within front, higher crowding distance first.
    rng = np.random.default_rng(rng_seed)
    # Add tiny jitter to break ties deterministically.
    jitter = rng.uniform(-1e-9, 1e-9, size=len(ranked))
    ranked.sort(key=lambda r: (r[0], -r[1] - jitter[r[2]]))

    selected_indices = [r[2] for r in ranked[:batch_size]]
    return [contexts[i] for i in selected_indices]
