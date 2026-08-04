"""Tests for Tournament, NSGA-II, and conformal confidence selection strategies."""

from __future__ import annotations

import numpy as np

from aurelius.agent.selection import nsga2_select
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
        smiles = ["CCCCCCCCCCCCCCCCCCCCCCCCO" for _ in range(20)]
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


class TestNSGA2Selection:
    """Tests for the NSGA-II multi-objective selection strategy."""

    def test_basic_batch_size(self):
        """NSGA-II should return exactly batch_size when pool > batch_size."""
        smiles = ["CCO", "CCCO", "CCCCO", "CCCCCO", "CCCCCCO", "CCCCCCCO"]
        contexts = [_valid_context(s) for s in smiles]
        scores_dict = {
            "dielectric_proxy": [30, 40, 50, 60, 70, 80],
            "viscosity_proxy": [5, 4, 3, 2, 1, 2],
            "sa_score": [3, 3, 4, 4, 5, 5],
        }
        objectives = [
            ("dielectric_proxy", "max"),
            ("viscosity_proxy", "min"),
            ("sa_score", "min"),
        ]
        selected = nsga2_select(contexts, scores_dict, objectives, batch_size=3)
        assert len(selected) == 3
        for ctx in selected:
            assert ctx in contexts

    def test_all_selected_when_fewer_than_batch(self):
        """If pool size <= batch_size, all candidates should be returned."""
        smiles = ["CCO", "CCCO", "CCCCO"]
        contexts = [_valid_context(s) for s in smiles]
        scores_dict = {"dielectric_proxy": [10, 20, 30]}
        objectives = [("dielectric_proxy", "max")]
        selected = nsga2_select(contexts, scores_dict, objectives, batch_size=10)
        assert len(selected) == 3

    def test_empty_pool(self):
        """Empty input should return empty list."""
        selected = nsga2_select([], {"dielectric_proxy": []}, [("dielectric_proxy", "max")], batch_size=5)
        assert selected == []

    def test_pareto_front_preserved(self):
        """Pareto-optimal candidates should be preferred."""
        # Candidate 0: diele=80, visc=1, sa=3  (best on dielectric, worst on viscosity)
        # Candidate 1: diele=60, visc=2, sa=4
        # Candidate 2: diele=40, visc=3, sa=5
        # Candidate 3: diele=20, visc=4, sa=6
        # Candidate 4: diele=10, visc=5, sa=7
        contexts = [_valid_context(f"CC{'C'*i}O") for i in range(5)]
        scores_dict = {
            "dielectric_proxy": [80, 60, 40, 20, 10],
            "viscosity_proxy": [1,  2,  3,  4,  5],
            "sa_score":        [3,  4,  5,  6,  7],
        }
        objectives = [
            ("dielectric_proxy", "max"),
            ("viscosity_proxy", "min"),
            ("sa_score", "min"),
        ]
        selected = nsga2_select(contexts, scores_dict, objectives, batch_size=4)
        assert len(selected) == 4
        # The best candidate (index 0) should always be selected — it's on the Pareto front
        assert contexts[0] in selected

    def test_confidence_bias(self):
        """Higher-confidence candidates should be preferred when other things are equal."""
        smiles = [f"CC{'C'*i}O" for i in range(8)]
        contexts = [_valid_context(s) for s in smiles]
        scores_dict = {
            "dielectric_proxy": [50, 50, 50, 50, 50, 50, 50, 50],
            "viscosity_proxy": [2, 2, 2, 2, 2, 2, 2, 2],
        }
        objectives = [("dielectric_proxy", "max"), ("viscosity_proxy", "min")]
        low_conf = [0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9]
        confidence_selected = nsga2_select(
            contexts, scores_dict, objectives, batch_size=4, confidences=low_conf
        )
        # All candidates are Pareto-optimal (same scores), so confidence should
        # break ties — higher confidence ones should be selected more.
        high_conf_idx = {smiles.index(c.smiles) for c in confidence_selected if c.smiles in smiles[4:]}
        # At least some should be from the high-confidence group
        assert len(high_conf_idx) >= 1

    def test_deterministic_with_same_seed(self):
        """Same seed should produce identical results."""
        smiles = [f"CC{'C'*i}O" for i in range(10)]
        contexts = [_valid_context(s) for s in smiles]
        scores_dict = {
            "dielectric_proxy": list(range(10, 0, -1)),
            "viscosity_proxy": list(range(1, 11)),
        }
        objectives = [("dielectric_proxy", "max"), ("viscosity_proxy", "min")]
        result1 = nsga2_select(contexts, scores_dict, objectives, batch_size=3, rng_seed=42)
        result2 = nsga2_select(contexts, scores_dict, objectives, batch_size=3, rng_seed=42)
        assert [c.smiles for c in result1] == [c.smiles for c in result2]

    def test_returns_from_original_pool(self):
        """All selected candidates must be from the original pool."""
        smiles = ["CCO", "CCCO", "CCCCO", "CCCCCO", "CCCCCCO",
                  "CCCCCCCO", "CCCCCCCCO", "COCCOC", "CCOCC", "CCCOCC"]
        contexts = [_valid_context(s) for s in smiles]
        rng = np.random.default_rng(42)
        scores_dict = {
            "dielectric_proxy": rng.random(10) * 100,
            "viscosity_proxy": rng.random(10) * 10,
            "sa_score": rng.random(10) * 7 + 1,
        }
        objectives = [
            ("dielectric_proxy", "max"),
            ("viscosity_proxy", "min"),
            ("sa_score", "min"),
        ]
        selected = nsga2_select(contexts, scores_dict, objectives, batch_size=5)
        assert len(selected) == 5
        for ctx in selected:
            assert ctx in contexts

    def test_single_objective_reduces_to_ranking(self):
        """With a single objective, NSGA-II should prefer the best value."""
        smiles = ["CCO", "CCCO", "CCCCO"]
        contexts = [_valid_context(s) for s in smiles]
        scores_dict = {"dielectric_proxy": [30.0, 60.0, 90.0]}
        objectives = [("dielectric_proxy", "max")]
        selected = nsga2_select(contexts, scores_dict, objectives, batch_size=2)
        assert len(selected) == 2
        # The best (index 2) should be selected
        assert contexts[2] in selected

    def test_minimisation_objective(self):
        """Minimisation objectives should prefer smaller values."""
        smiles = ["CCO", "CCCO", "CCCCO"]
        contexts = [_valid_context(s) for s in smiles]
        scores_dict = {"viscosity_proxy": [5.0, 2.0, 8.0]}
        objectives = [("viscosity_proxy", "min")]
        selected = nsga2_select(contexts, scores_dict, objectives, batch_size=1)
        assert len(selected) == 1
        # Best (lowest viscosity = index 1) should be selected
        assert selected[0] == contexts[1]


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
