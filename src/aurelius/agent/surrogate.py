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
from rdkit.DataStructs import TanimotoSimilarity
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor


class RandomForestSurrogate:
    """Random Forest surrogate with Expected Improvement acquisition.

    The RF provides both mean (μ) and variance (σ²) — the standard deviation
    of predictions across all trees in the ensemble. The surrogate uses the
    full feature vector without dimensionality reduction, as tree-based models
    handle high-dimensional binary fingerprints natively.

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

        # The RF is fitted directly on the (N, n_features) feature matrix.
        # Tree-based models handle sparse/high-dimensional binary features
        # natively, so no dimensionality reduction is needed.
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
        fingerprints: list[Any] | None = None,
        top_n: int = 10,
        diversity_lambda: float = 0.5,
    ) -> list[int]:
        """Select top candidates with optional diversity-penalized batch selection.

        Uses greedy diversity penalization: select first candidate by max EI,
        then for each subsequent candidate apply ``Score = EI * (1 - lambda * max_tanimoto_to_selected)``.
        This ensures the batch covers diverse chemical space rather than collapsing
        on nearly identical top-EI structures.

        Args:
            X_candidates: Feature matrix of shape (n_candidates, n_features).
            fingerprints: List of RDKit ECFP4 fingerprints for Tanimoto calculation.
                If None, falls back to pure EI ranking.
            top_n: Number of candidates to select.
            diversity_lambda: Strength of diversity penalty (0 = no penalty, 1 = max penalty).

        Returns:
            List of selected candidate indices (length <= top_n).
        """
        ei = self.expected_improvement(X_candidates)

        if fingerprints is None or len(fingerprints) == 0:
            top_indices = np.argsort(ei)[::-1][:top_n]
            return top_indices.tolist()

        # Greedy diversity-penalized selection
        n = len(ei)
        selected: list[int] = []
        remaining = list(range(n))

        for _ in range(min(top_n, n)):
            if not remaining:
                break

            if not selected:
                best_idx = remaining[int(np.argmax(ei[remaining]))]
            else:
                best_score = -float("inf")
                best_idx = -1
                for i in remaining:
                    max_sim = max(
                        TanimotoSimilarity(fingerprints[i], fingerprints[j])
                        for j in selected
                    )
                    score = ei[i] * (1.0 - diversity_lambda * max_sim)
                    if score > best_score:
                        best_score = score
                        best_idx = i

            selected.append(best_idx)
            remaining.remove(best_idx)

        return selected
