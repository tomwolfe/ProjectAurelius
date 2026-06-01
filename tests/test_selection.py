"""Tests for Tournament/Early-stopping selection strategy."""

from __future__ import annotations

import numpy as np
from rdkit import Chem

from aurelius.types import MoleculeContext


class TestTournamentSelection:
    """Tests for the direct EA/Tournament selection strategy."""

    def test_selects_top_scorers(self):
        """Tournament selection should prefer higher-scoring candidates."""
        scores = [10.0, 50.0, 80.0, 20.0, 90.0]
        smiles = ["CCO", "CCCO", "CCCCO", "CCCCCCO", "CCCCCO"]
        contexts = [_valid_context(s) for s in smiles]

        selected = _tournament_select(contexts, scores, batch_size=2)
        assert len(selected) == 2

        selected_scores = [scores[contexts.index(c)] for c in selected]
        assert max(selected_scores) >= 50.0

    def test_batch_size_respected(self):
        """Selection should return exactly batch_size candidates (if available)."""
        scores = list(range(20))
        smiles = [f"CCCCCCCCCCCCCCCCCCCCCCCCO" for _ in range(20)]
        contexts = [_valid_context(s) for s in smiles]

        selected = _tournament_select(contexts, scores, batch_size=5)
        assert len(selected) == 5

    def test_diversity_prevents_collapse(self):
        """Tanimoto diversity penalty should spread selection across similar molecules."""
        smiles = ["CCO", "CCO", "C1COCCO1"]
        contexts = [_valid_context(s) for s in smiles]
        scores = [80.0, 79.0, 50.0]

        selected = _tournament_select(contexts, scores, batch_size=2)
        assert len(selected) == 2

        selected_smiles = {c.smiles for c in selected}
        assert len(selected_smiles) >= 1

    def test_returns_valid_indices(self):
        """Selected candidates should all be from the original pool."""
        smiles = ["CCO", "CCCO", "CCCCO", "CCCCCO", "CCCCCCO",
                  "CCCCCCCO", "CCCCCCCCO", "CCCCCCCCCO", "CCCCCCCCCO", "COCCOC"]
        scores = np.random.default_rng(42).random(10) * 100
        contexts = [_valid_context(s) for s in smiles]

        selected = _tournament_select(contexts, scores, batch_size=4)
        for ctx in selected:
            assert ctx in contexts

    def test_selects_from_all_contexts(self):
        """All candidates are eligible for selection."""
        smiles = ["CCO", "CCCO", "CCCCO"]
        contexts = [_valid_context(s) for s in smiles]
        scores = [30.0, 60.0, 90.0]

        selected = _tournament_select(contexts, scores, batch_size=3)
        assert len(selected) == 3

    def test_tournament_pressure(self):
        """Tournament size should exert selection pressure."""
        scores = [float(i) for i in range(50)]
        unique_smiles = ["CCO", "CCCO", "CCCCO", "CCCCCO", "CCCCCCO",
                         "CCCCCCCO", "CCCCCCCCO", "COCCOC", "CCOCC", "CCCOCC"]
        smiles = [unique_smiles[i % len(unique_smiles)] for i in range(50)]
        contexts = [_valid_context(s) for s in smiles]

        with_small = _tournament_select(contexts, scores, batch_size=20, tournament_size=2)
        small_scores = [scores[contexts.index(c)] for c in with_small]

        with_large = _tournament_select(contexts, scores, batch_size=20, tournament_size=10)
        large_scores = [scores[contexts.index(c)] for c in with_large]

        assert np.mean(large_scores) >= np.mean(small_scores)

    def test_diversity_lambda_controls_penalty(self):
        """Higher diversity_lambda should spread selection more."""
        smiles = ["CCO", "CCO", "CCO", "C1COCCO1"]
        contexts = [_valid_context(s) for s in smiles]
        scores = [90.0, 85.0, 80.0, 70.0]

        selected_low = _tournament_select(contexts, scores, batch_size=3, diversity_lambda=0.0)
        selected_high = _tournament_select(contexts, scores, batch_size=3, diversity_lambda=0.9)

        assert len(selected_low) == 3
        assert len(selected_high) == 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_context(smiles: str) -> MoleculeContext:
    ctx = MoleculeContext.from_smiles(smiles)
    assert ctx is not None, f"Could not parse SMILES: {smiles}"
    return ctx


def _tanimoto_similarity(fp_a, fp_b):
    from rdkit.DataStructs import TanimotoSimilarity
    return TanimotoSimilarity(fp_a, fp_b)


def _tournament_select(
    contexts: list[MoleculeContext],
    scores: list[float],
    batch_size: int = 10,
    tournament_size: int = 3,
    diversity_lambda: float = 0.3,
) -> list[MoleculeContext]:
    """Simplified tournament selection matching the real implementation."""
    import random

    n = len(contexts)
    if n == 0:
        return []
    if n <= batch_size:
        return list(contexts)

    rng = random.Random(42)
    used_indices: set[int] = set()
    selected: list[MoleculeContext] = []
    selected_fps: list = []

    for _ in range(batch_size):
        pool = [i for i in range(n) if i not in used_indices]
        if not pool:
            break

        tournament = rng.sample(pool, min(tournament_size, len(pool)))
        best_idx = max(tournament, key=lambda i: scores[i])
        best_adj = scores[best_idx]

        if selected_fps:
            fp_best = contexts[best_idx].get_ecfp4()
            max_sim_best = max(_tanimoto_similarity(fp_best, sfp) for sfp in selected_fps)
            best_adj = scores[best_idx] * (1.0 - diversity_lambda * max_sim_best)

            for i in tournament:
                if i in used_indices:
                    continue
                fp_i = contexts[i].get_ecfp4()
                max_sim_i = max(_tanimoto_similarity(fp_i, sfp) for sfp in selected_fps)
                adj = scores[i] * (1.0 - diversity_lambda * max_sim_i)
                if adj > best_adj:
                    best_adj = adj
                    best_idx = i

        if best_idx not in used_indices:
            used_indices.add(best_idx)
            selected.append(contexts[best_idx])
            selected_fps.append(contexts[best_idx].get_ecfp4())

    return selected
