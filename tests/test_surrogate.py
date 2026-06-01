"""Tests for Random Forest surrogate and novelty-weighted acquisition."""

from __future__ import annotations

import numpy as np
import pytest

from aurelius.agent.surrogate import RandomForestSurrogate


class TestRandomForestSurrogate:
    """Tests for the Random Forest surrogate model."""

    def test_fit_raises_on_insufficient_data(self):
        """Surrogate requires at least 2 data points to fit."""
        surrogate = RandomForestSurrogate()
        X = np.array([[1.0, 0.0, 0.0]])
        y = np.array([0.5])
        with pytest.raises(ValueError, match="At least 2 samples"):
            surrogate.fit(X, y)

    def test_novelty_weighted_score_requires_fit(self):
        """Scoring candidates before fitting must raise RuntimeError."""
        surrogate = RandomForestSurrogate()
        candidates = np.array([[1.0, 0.0, 0.0]])
        with pytest.raises(RuntimeError, match="must be fitted"):
            surrogate.novelty_weighted_score(candidates)

    def test_score_candidates_requires_fit(self):
        """Selecting top candidates before fitting must raise RuntimeError."""
        surrogate = RandomForestSurrogate()
        candidates = np.array([[1.0, 0.0, 0.0]])
        with pytest.raises(RuntimeError, match="must be fitted"):
            surrogate.score_candidates(candidates, top_n=1)

    def test_novelty_weighted_favors_high_mean(self):
        """Score should be higher for candidates with higher predicted mean."""
        surrogate = RandomForestSurrogate()

        X_train = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        y_train = np.array([0.9, 0.5, 0.3])
        surrogate.fit(X_train, y_train)

        high_candidate = np.array([[0.95, 0.0, 0.0]])
        scores = surrogate.novelty_weighted_score(high_candidate)
        assert len(scores) == 1
        assert scores[0] >= -0.1, f"Expected score >= -0.1 but got {scores[0]}"

    def test_novelty_weighted_favors_diverse_candidates(self):
        """Score should be higher for candidates far from training data."""
        surrogate = RandomForestSurrogate()

        X_train = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        y_train = np.array([0.9, 0.5, 0.3])
        surrogate.fit(X_train, y_train)

        far_candidate = np.array([[0.0, 0.0, 0.0]])
        scores = surrogate.novelty_weighted_score(far_candidate)
        assert len(scores) == 1
        assert scores[0] >= -0.1, f"Expected score >= -0.1 but got {scores[0]}"

    def test_score_candidates_returns_top_indices(self):
        """Surrogate results should be reproducible with fixed random_state."""
        surrogate_a = RandomForestSurrogate(random_state=42)
        surrogate_b = RandomForestSurrogate(random_state=42)

        X = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        y = np.array([0.8, 0.5])

        surrogate_a.fit(X, y)
        surrogate_b.fit(X, y)

        candidates = np.array([[0.7, 0.3, 0.0]])
        scores_a = surrogate_a.novelty_weighted_score(candidates)
        scores_b = surrogate_b.novelty_weighted_score(candidates)

        np.testing.assert_array_almost_equal(scores_a, scores_b)

    def test_novelty_weighted_computes_bonus_correctly(self):
        """Novelty-weighted score should amplify by (1 + alpha * max_tanimoto_distance)."""
        surrogate = RandomForestSurrogate(random_state=42, alpha=2.0)

        rng = np.random.default_rng(42)
        X_train = np.zeros((10, 2053), dtype=np.float32)
        for i in range(10):
            X_train[i, :4] = rng.integers(0, 2, size=4)
        y_train = np.array([50.0 + 30.0 * rng.random() for _ in range(10)], dtype=np.float32)
        surrogate.fit(X_train, y_train)

        candidate_a = X_train[0:1].copy()

        candidate_b = np.zeros((1, 2053), dtype=np.float32)
        candidate_b[0, 2047] = 1.0

        nws_a = surrogate.novelty_weighted_score(candidate_a)
        nws_b = surrogate.novelty_weighted_score(candidate_b)

        assert np.isfinite(nws_a[0]), f"Score close candidate not finite: {nws_a[0]}"
        assert np.isfinite(nws_b[0]), f"Score far candidate not finite: {nws_b[0]}"

        assert nws_b[0] > nws_a[0], (
            f"Far candidate should have higher novelty-weighted score than close candidate: "
            f"nws_a={nws_a[0]:.4f}, nws_b={nws_b[0]:.4f}"
        )

        from rdkit.DataStructs import BulkTanimotoSimilarity, ExplicitBitVect

        ecfp_nbits = 2048
        fp_a = ExplicitBitVect(ecfp_nbits)
        for idx in np.flatnonzero(candidate_a[0, :ecfp_nbits] > 0.5):
            fp_a.SetBit(int(idx))
        train_fps = [
            ExplicitBitVect(ecfp_nbits)
            for _ in range(10)
        ]
        for j in range(10):
            bv = ExplicitBitVect(ecfp_nbits)
            for idx in np.flatnonzero(X_train[j, :ecfp_nbits] > 0.5):
                bv.SetBit(int(idx))
            train_fps[j] = bv

        sims_a = BulkTanimotoSimilarity(fp_a, train_fps)
        dist_a = 1.0 - max(sims_a)
        assert dist_a < 0.1, f"Close candidate Tanimoto distance should be near 0, got {dist_a}"

        fp_b = ExplicitBitVect(ecfp_nbits)
        for idx in np.flatnonzero(candidate_b[0, :ecfp_nbits] > 0.5):
            fp_b.SetBit(int(idx))
        sims_b = BulkTanimotoSimilarity(fp_b, train_fps)
        dist_b = 1.0 - max(sims_b)
        assert dist_b > 0.9, f"Far candidate Tanimoto distance should be near 1, got {dist_b}"
