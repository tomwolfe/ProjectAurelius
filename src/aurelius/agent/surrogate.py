"""Random Forest surrogate model with true Expected Improvement acquisition.

The surrogate provides predictive mean and variance from the Random Forest
ensemble (standard deviation across ``n_estimators`` trees). The acquisition
function is the standard analytical Expected Improvement (EI):

    EI(x) = (μ - y_best - ξ) Φ(Z) + σ ϕ(Z)

where Z = (μ - y_best - ξ) / σ, Φ is the normal CDF, ϕ is the normal PDF,
μ is the mean prediction, σ is the standard deviation across trees, y_best
is the best observed score, and ξ is the exploration-exploitation trade-off
parameter.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor


class RandomForestSurrogate:
    """Random Forest surrogate with true Expected Improvement acquisition.

    The RF provides both mean (μ) and variance (σ²) natively — the standard
    deviation of predictions across all trees in the ensemble.  This enables
    the standard analytical Expected Improvement formula, which rigorously
    balances exploration (high σ) and exploitation (high μ).

    Usage:
        surrogate = RandomForestSurrogate(xi=0.01)
        surrogate.fit(X_train, y_train)
        ei = surrogate.expected_improvement(X_candidates)
        best_indices = surrogate.score_candidates(X_candidates, top_n=10)
    """

    def __init__(self, random_state: int = 42, xi: float = 0.01) -> None:
        self._X: np.ndarray[Any, Any] | None = None
        self._y: np.ndarray[Any, Any] | None = None
        self._rf: RandomForestRegressor | None = None
        self._random_state = random_state
        self._xi = xi
        self._y_best: float = -float("inf")

    def fit(self, X: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> None:
        if len(y) < 2:
            raise ValueError("At least 2 samples are required to fit the Random Forest surrogate.")

        self._rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            random_state=self._random_state,
            n_jobs=1,
        )
        self._rf.fit(X, y)

        self._X = X
        self._y = y
        self._y_best = float(np.max(y))

    def expected_improvement(self, X_candidates: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute the Expected Improvement acquisition function.

        Uses the standard analytical formula:

            EI(x) = (μ - y_best - ξ) Φ(Z) + σ ϕ(Z)

        where Z = (μ - y_best - ξ) / σ.

        The RF provides σ as the standard deviation of predictions across
        all 100 trees.  This naturally rewards candidates where the ensemble
        disagrees (exploration) while biasing toward high predicted scores
        (exploitation).

        Args:
            X_candidates: Feature matrix of shape (n_candidates, n_features).

        Returns:
            Array of EI values of shape (n_candidates,).
        """
        if self._X is None or self._y is None:
            raise RuntimeError(
                "RandomForestSurrogate must be fitted before scoring candidates. "
                "Call .fit(X, y) with training data first."
            )
        if self._rf is None:
            raise RuntimeError("Random Forest surrogate is not trained.")

        tree_preds = np.array([tree.predict(X_candidates) for tree in self._rf.estimators_])
        mu = np.mean(tree_preds, axis=0)
        sigma = np.std(tree_preds, axis=0, ddof=1)

        Z = np.divide(
            mu - self._y_best - self._xi,
            sigma,
            out=np.full_like(mu, -float("inf")),
            where=sigma > 1e-12,
        )

        ei = (mu - self._y_best - self._xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei = np.where(sigma > 1e-12, ei, np.maximum(mu - self._y_best - self._xi, 0.0))
        ei = np.maximum(ei, 0.0)

        return ei

    def score_candidates(
        self,
        X_candidates: np.ndarray[Any, Any],
        top_n: int = 10,
    ) -> list[int]:
        ei = self.expected_improvement(X_candidates)
        top_indices = np.argsort(ei)[::-1][:top_n]
        return top_indices.tolist()
