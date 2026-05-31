"""Tests for Random Forest surrogate and Expected Improvement acquisition."""

from __future__ import annotations

import numpy as np
import pytest

from aurelius.agent.surrogate import RandomForestSurrogate


class TestRandomForestSurrogate:
    """Tests for the Gaussian Process surrogate model."""

    def test_fit_raises_on_insufficient_data(self):
        """GP requires at least 2 data points to fit."""
        surrogate = RandomForestSurrogate()
        X = np.array([[1.0, 0.0, 0.0]])
        y = np.array([0.5])
        with pytest.raises(ValueError, match="At least 2 samples"):
            surrogate.fit(X, y)

    def test_expected_improvement_requires_fit(self):
        """Scoring candidates before fitting must raise RuntimeError."""
        surrogate = RandomForestSurrogate()
        candidates = np.array([[1.0, 0.0, 0.0]])
        with pytest.raises(RuntimeError, match="must be fitted"):
            surrogate.expected_improvement(candidates)

    def test_score_candidates_requires_fit(self):
        """Selecting top candidates before fitting must raise RuntimeError."""
        surrogate = RandomForestSurrogate()
        candidates = np.array([[1.0, 0.0, 0.0]])
        with pytest.raises(RuntimeError, match="must be fitted"):
            surrogate.score_candidates(candidates, top_n=1)

    def test_expected_improvement_favors_high_mean(self):
        """EI should be higher for candidates with higher predicted mean."""
        surrogate = RandomForestSurrogate()

        # Simulate training data: 3 points with varying scores
        X_train = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        y_train = np.array([0.9, 0.5, 0.3])
        surrogate.fit(X_train, y_train)

        # Candidate with high predicted mean should have higher EI
        high_candidate = np.array([[0.95, 0.0, 0.0]])
        ei = surrogate.expected_improvement(high_candidate)
        assert len(ei) == 1
        assert ei[0] >= -0.1, f"Expected EI >= -0.1 but got {ei[0]}"

    def test_expected_improvement_favors_high_uncertainty(self):
        """EI should be higher for candidates with high predictive uncertainty."""
        surrogate = RandomForestSurrogate()

        X_train = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        y_train = np.array([0.9, 0.5, 0.3])
        surrogate.fit(X_train, y_train)

        # Candidate far from training data should have high uncertainty
        far_candidate = np.array([[0.0, 0.0, 0.0]])  # far from all training points
        ei = surrogate.expected_improvement(far_candidate)
        assert len(ei) == 1
        # High uncertainty near training data should yield non-zero EI
        assert ei[0] >= -0.1, f"Expected EI >= -0.1 but got {ei[0]}"

    def test_score_candidates_returns_top_indices(self):
        """Surrogate results should be reproducible with fixed random_state."""
        surrogate_a = RandomForestSurrogate(random_state=42)
        surrogate_b = RandomForestSurrogate(random_state=42)

        X = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        y = np.array([0.8, 0.5])

        surrogate_a.fit(X, y)
        surrogate_b.fit(X, y)

        candidates = np.array([[0.7, 0.3, 0.0]])
        ei_a = surrogate_a.expected_improvement(candidates)
        ei_b = surrogate_b.expected_improvement(candidates)

        np.testing.assert_array_almost_equal(ei_a, ei_b)

    def test_nwei_computes_novelty_bonus_correctly(self):
        """NWEI should amplify EI by (1 + alpha * max_tanimoto_distance)."""
        # Use a surrogate with alpha=2.0 so the bonus is clearly measurable
        surrogate = RandomForestSurrogate(random_state=42, alpha=2.0)

        # Training set: 10 samples, 2053-dim (2048 ECFP4 + 5 descriptors)
        rng = np.random.default_rng(42)
        X_train = np.zeros((10, 2053), dtype=np.float32)
        # Set first 4 bits as active in various patterns
        for i in range(10):
            X_train[i, :4] = rng.integers(0, 2, size=4)
        y_train = np.array([50.0 + 30.0 * rng.random() for _ in range(10)], dtype=np.float32)
        surrogate.fit(X_train, y_train)

        # Candidate A: identical to X_train[0] → Tanimoto distance ≈ 0
        candidate_a = X_train[0:1].copy()

        # Candidate B: first 4 bits all zero, bit 2047 set → no overlap → distance ≈ 1
        candidate_b = np.zeros((1, 2053), dtype=np.float32)
        candidate_b[0, 2047] = 1.0

        nwei_a = surrogate.expected_improvement(candidate_a)
        nwei_b = surrogate.expected_improvement(candidate_b)

        # Both must produce finite NWEI values
        assert np.isfinite(nwei_a[0]), f"NWEI close candidate not finite: {nwei_a[0]}"
        assert np.isfinite(nwei_b[0]), f"NWEI far candidate not finite: {nwei_b[0]}"

        # Novelty bonus behavior: far candidate (no overlap with training set)
        # should receive a higher novelty-weighted EI than the identical candidate.
        # NWEI = EI * (1 + alpha * max_tanimoto_distance), so with alpha=2.0,
        # candidate B (distance ≈ 1) gets a 3x multiplier vs candidate A (distance ≈ 0).
        assert nwei_b[0] > nwei_a[0], (
            f"Far candidate should have higher NWEI than close candidate: "
            f"nwei_a={nwei_a[0]:.4f}, nwei_b={nwei_b[0]:.4f}"
        )

        # Verify via RDKit BulkTanimotoSimilarity that Tanimoto distance for
        # candidate A is ≈ 0 (identical to training point) and for B is ≈ 1.
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
