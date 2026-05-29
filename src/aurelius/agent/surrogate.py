"""Gaussian Process surrogate model for Bayesian-guided active learning.

Provides Expected Improvement (EI) acquisition function scoring over
Morgan (ECFP4) fingerprints, enabling the DiscoveryLoop to select
the most promising candidates from a large mutation pool.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, RationalQuadratic
from sklearn.preprocessing import StandardScaler


class GaussianProcessSurrogate:
    """Gaussian Process surrogate model for active learning.

    The surrogate is trained on Morgan (ECFP4) fingerprints ``X``
    and composite ``Aurelius Score`` values ``y``.  During acquisition
    (``expected_improvement``), the GP predicts the mean and variance
    of each candidate, and the acquisition function scores high both
    high predicted scores and high uncertainty.

    As a fallback when the GP fails (e.g. singular kernel matrix),
    a Random Forest regressor is used to score candidates.
    """

    def __init__(self, random_state: int = 42) -> None:
        """Initialise the surrogate.

        Args:
            random_state: Random seed for reproducibility.
        """
        self._X: np.ndarray[Any, Any] | None = None
        self._y: np.ndarray[Any, Any] | None = None
        self._gp: GaussianProcessRegressor | None = None
        self._rf: RandomForestRegressor | None = None
        self._scaler_x: StandardScaler | None = None
        self._scaler_y: StandardScaler | None = None
        self._random_state = random_state

    def fit(self, X: np.ndarray[Any, Any], y: np.ndarray[Any, Any]) -> None:
        """Fit the Gaussian Process surrogate to (X, y) data.

        Args:
            X: 2-D array of shape (n_samples, n_features) with Morgan
                fingerprints (ECFP4 radius=2).
            y: 1-D array of composite Aurelius scores.

        Raises:
            ValueError: If fewer than 2 samples are provided.
        """
        if len(y) < 2:
            raise ValueError("At least 2 samples are required to fit the Gaussian Process surrogate.")

        # Scale inputs and targets for numerical stability
        self._scaler_x = StandardScaler()
        self._scaler_y = StandardScaler()
        X_scaled = self._scaler_x.fit_transform(X)
        y_scaled = self._scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        kernel = C(1.0, (0.1, 10.0)) * RBF(length_scale=5.0, length_scale_bounds=(0.1, 100.0))
        kernel += RationalQuadratic(alpha=1.0, beta=1.0)

        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            random_state=self._random_state,
            normalize_y=False,
        )
        self._gp.fit(X_scaled, y_scaled)

        # Store the original data for Expected Improvement computation
        self._X = X_scaled
        self._y = y_scaled

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
                "GaussianProcessSurrogate must be fitted before scoring candidates. "
                "Call .fit(X, y) with training data first."
            )

        if self._gp is None:
            raise RuntimeError("Gaussian Process surrogate is not trained.")

        # Predict mean and variance for candidates
        try:
            mean, std = self._gp.predict(X_candidates, return_std=True)
        except Exception:
            # Fallback to Random Forest when GP fails
            mean, std = self._rf_scores(X_candidates)

        best = self._y.max() if len(self._y) > 0 else 0.0
        ei = np.zeros(len(mean))

        for i in range(len(mean)):
            mu = mean[i]
            s = std[i]
            if s < 1e-8:
                ei[i] = 0.0
            else:
                z = (mu - best) / s
                ei[i] = (mu - best) * self._norm_cdf(z) + s * self._norm_pdf(z)

        return ei

    def _rf_scores(self, X: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Compute Random Forest predictions as a GP fallback.

        Returns:
            Tuple of (mean_predictions, uncertainty_estimates).
        """
        if self._rf is None:
            self._rf = RandomForestRegressor(
                n_estimators=100,
                random_state=self._random_state,
                n_jobs=-1,
            )
            if self._X is not None and self._y is not None:
                self._rf.fit(self._X, self._y)

        if self._rf is not None and self._X is not None and self._y is not None:
            mean = self._rf.predict(X)
            std = self._rf.predict(X) * 0.1  # Dummy uncertainty
            return mean, std
        return np.zeros(len(X)), np.zeros(len(X))

    @staticmethod
    def _norm_cdf(x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
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
