"""Random Forest surrogate model for Bayesian-guided active learning.

Provides Novelty-Weighted Expected Improvement (NWEI) acquisition
function scoring over ECFP4 fingerprints augmented with global RDKit
descriptors, enabling the DiscoveryLoop to select candidates that are
both promising and structurally novel.

The surrogate uses a Random Forest Regressor which natively handles
high-dimensional sparse binary vectors and provides uncertainty
estimates from tree-based variance for the NWEI calculation.
The novelty bonus is computed as the maximum Tanimoto distance of each
candidate to the training set, rewarding out-of-distribution exploration.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class RandomForestSurrogate:
    """Random Forest surrogate model for active learning.

    The surrogate is trained on Morgan + global descriptor features ``X``
    and composite ``Aurelius Score`` values ``y``.  During acquisition
    (``expected_improvement``), the RF predicts the mean and variance
    of each candidate, and the acquisition function scores high both
    high predicted scores and high uncertainty.

    The Novelty-Weighted EI (NWEI) multiplies standard EI by
    ``1.0 + alpha * max_tanimoto_distance``, where the Tanimoto distance
    is computed against the training set using only the ECFP4 bits
    (first 2048 features).
    """

    def __init__(self, random_state: int = 42, alpha: float = 1.0) -> None:
        """Initialise the surrogate.

        Args:
            random_state: Random seed for reproducibility.
            alpha: Novelty bonus strength for NWEI.
        """
        self._X: np.ndarray[Any, Any] | None = None
        self._y: np.ndarray[Any, Any] | None = None
        self._rf: RandomForestRegressor | None = None
        self._random_state = random_state
        self._alpha = alpha

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
            n_jobs=1,
        )
        self._rf.fit(X, y)

        # Store the original data for Expected Improvement computation
        self._X = X
        self._y = y

    def expected_improvement(self, X_candidates: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute Novelty-Weighted Expected Improvement (NWEI) for candidates.

        NWEI favours candidates that either have high predicted mean,
        high predictive uncertainty, OR high structural novelty relative
        to the training set.

        Inter-tree variance from the Random Forest's ensemble of decision
        trees is used as a proxy for epistemic uncertainty in the EI
        calculation.

        The novelty bonus is computed as the maximum Tanimoto distance
        of each candidate's ECFP4 bits (first 2048 features) to the
        training set, scaled by ``alpha``.

        Args:
            X_candidates: 2-D array of shape (n_candidates, n_features).

        Returns:
            1-D array of NWEI acquisition values (higher = more acquisition-worthy).

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
        tree_preds = np.array([tree.predict(X_candidates) for tree in self._rf.estimators_])
        means = tree_preds.mean(axis=0)
        variances = tree_preds.var(axis=0)

        # Standard Expected Improvement
        s = np.sqrt(np.maximum(variances, 1e-8))
        ei = np.zeros_like(means)
        mask = s >= 1e-8
        z = (means[mask] - best) / s[mask]
        ei[mask] = (means[mask] - best) * self._norm_cdf(z) + s[mask] * self._norm_pdf(z)

        # Novelty bonus: max Tanimoto distance to training set (ECFP4 bits only)
        ecfp_nbits = min(2048, X_candidates.shape[1], self._X.shape[1])
        X_cand_ecfp = X_candidates[:, :ecfp_nbits]
        X_train_ecfp = self._X[:, :ecfp_nbits]

        dot_prod = X_cand_ecfp @ X_train_ecfp.T
        sum_cand = X_cand_ecfp.sum(axis=1, keepdims=True)
        sum_train = X_train_ecfp.sum(axis=1, keepdims=True).T
        denom = sum_cand + sum_train - dot_prod
        denom = np.maximum(denom, 1e-10)

        tanimoto = dot_prod / denom
        max_sim = tanimoto.max(axis=1)
        max_tanimoto_dist = 1.0 - max_sim

        # Novelty-weighted EI
        nwei = ei * (1.0 + self._alpha * max_tanimoto_dist)

        return nwei

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
        """Return indices of top-N candidates by Novelty-Weighted EI.

        Args:
            X_candidates: 2-D array of shape (n_candidates, n_features).
            top_n: Number of top candidates to return.

        Returns:
            List of indices into ``X_candidates`` sorted by descending NWEI.

        Raises:
            RuntimeError: If the surrogate has not been fitted yet.
        """
        nwei = self.expected_improvement(X_candidates)
        top_indices = np.argsort(nwei)[::-1][:top_n]
        return top_indices.tolist()
