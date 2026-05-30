"""Random Forest surrogate model for Bayesian-guided active learning.

Provides Expected Improvement (EI) acquisition function scoring over
Morgan (ECFP4) fingerprints, enabling the DiscoveryLoop to select
the most promising candidates from a large mutation pool.

The surrogate uses a Random Forest Regressor which natively handles
high-dimensional sparse binary vectors and provides uncertainty
estimates from tree-based variance for the EI calculation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class RandomForestSurrogate:
    """Random Forest surrogate model for active learning.

    The surrogate is trained on Morgan (ECFP4) fingerprints ``X``
    and composite ``Aurelius Score`` values ``y``.  During acquisition
    (``expected_improvement``), the RF predicts the mean and variance
    of each candidate, and the acquisition function scores high both
    high predicted scores and high uncertainty.

    As a fallback when the RF fails (e.g. insufficient data),
    a simple random scoring is used.
    """

    def __init__(self, random_state: int = 42) -> None:
        """Initialise the surrogate.

        Args:
            random_state: Random seed for reproducibility.
        """
        self._X: np.ndarray[Any, Any] | None = None
        self._y: np.ndarray[Any, Any] | None = None
        self._rf: RandomForestRegressor | None = None
        self._random_state = random_state

    def fit(self, X: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> None:
        """Fit the Random Forest surrogate to (X, y) data.

        Args:
            X: 2-D array of shape (n_samples, n_features) with Morgan
                fingerprints (ECFP4 radius=2).
            y: 1-D array of composite Aurelius scores.

        Raises:
            ValueError: If fewer than 2 samples are provided.
        """
        if len(y) < 2:
            raise ValueError("At least 2 samples are required to fit the Random Forest surrogate.")

        self._rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            random_state=self._random_state,
            n_jobs=-1,
        )
        self._rf.fit(X, y)

        # Store the original data for Expected Improvement computation
        self._X = X
        self._y = y

    def expected_improvement(self, X_candidates: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute Expected Improvement acquisition values for candidates.

        Expected Improvement favours candidates that either have high
        predicted mean or high predictive uncertainty.

        Args:
            X_candidates: 2-D array of shape (n_candidates, n_features).

        Returns:
            1-D array of EI acquisition values (higher = more acquisition-worthy).

        Raises:
            RuntimeError: If the surrogate has not been fitted yet.
        """
        if self._X is None or self._y is None:
            raise RuntimeError(
                "RandomForestSurrogate must be fitted before scoring candidates. "
                "Call .fit(X, y) with training data first."
            )

        if self._rf is None:
            raise RuntimeError("Random Forest surrogate is not trained.")

        best = self._y.max() if len(self._y) > 0 else 0.0

        # Vectorized: get predictions from all trees at once
        # tree_preds shape: (n_trees, n_candidates)
        tree_preds = np.array([tree.predict(X_candidates) for tree in self._rf.estimators_])
        means = tree_preds.mean(axis=0)
        variances = tree_preds.var(axis=0)

        # Vectorized Expected Improvement
        s = np.sqrt(np.maximum(variances, 1e-8))
        ei = np.zeros_like(means)
        mask = s >= 1e-8
        z = (means[mask] - best) / s[mask]
        ei[mask] = (means[mask] - best) * self._norm_cdf(z) + s[mask] * self._norm_pdf(z)

        return ei

    def _norm_cdf(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Standard normal CDF approximation (Abramowitz & Stegun)."""
        return 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x - 0.13 * x**3)))

    @staticmethod
    def _norm_pdf(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Standard normal PDF."""
        return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)

    def score_candidates(
        self,
        X_candidates: np.ndarray[Any, Any],
        top_n: int = 10,
    ) -> list[int]:
        """Return indices of top-N candidates by Expected Improvement.

        Args:
            X_candidates: 2-D array of shape (n_candidates, n_features).
            top_n: Number of top candidates to return.

        Returns:
            List of indices into ``X_candidates`` sorted by descending EI.

        Raises:
            RuntimeError: If the surrogate has not been fitted yet.
        """
        ei = self.expected_improvement(X_candidates)
        top_indices = np.argsort(ei)[::-1][:top_n]
        return top_indices.tolist()
