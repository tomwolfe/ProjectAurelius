"""MLX-native surrogate for GPR residual prediction and batch operations.

Ports the compute-intensive matrix operations in GPR inference from sklearn/CPU
to Apple Silicon GPU via MLX. Training stays on CPU (calibration sets are small,
~115-231 points), but batch inference — which dominates the EA loop — runs on
the M5 Pro GPU.

Key operations accelerated:
  - GPR posterior mean:  μ* = K(X*, X_train) @ α
  - GPR posterior variance: σ² = k(x*, x*) - vᵀv where v = L⁻¹ @ k(X*, X_train)ᵀ
  - Batch Tanimoto: already in oracle.py, re-exported here for convenience

Target: >2000 mol/sec for Tier-1 batch evaluation on M5 Pro.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from aurelius.utils.device import get_device

logger = logging.getLogger(__name__)


def _ecfp4_dense(mol: Chem.Mol, n_bits: int = 2048) -> np.ndarray:
    """Dense ECFP4 bit vector for a molecule."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    vec = np.zeros(n_bits, dtype=np.float32)
    for bit in fp.GetOnBits():
        vec[bit] = 1.0
    return vec


class MLXGPRSurrogate:
    """MLX-accelerated GPR inference for a fitted sklearn GPR model.

    Wraps a trained ``GaussianProcessRegressor`` and ports the matrix-heavy
    inference to MLX. The sklearn model is used as the source of truth for
    training data (``X_train_``, ``alpha_``, ``L_``); only the per-molecule
    kernel computation and matrix products are accelerated.

    Usage::

        surrogate = MLXGPRSurrogate(sklearn_gpr)
        means, stds = surrogate.predict_batch(mols)
    """

    def __init__(self, sklearn_gpr: Any, n_bits: int = 2048) -> None:
        """Wrap a fitted sklearn GPR for MLX inference.

        Extracts the precomputed training data and Cholesky factor from the
        sklearn model so inference never touches sklearn.
        """
        self._n_bits = n_bits
        self._fitted = False
        self._train_mlx: Any = None
        self._alpha_mlx: Any = None
        self._L_mlx: Any = None
        self._k_diag_train: Any = None
        self._length_scale: float = 1.0

        try:
            self._extract_from_sklearn(sklearn_gpr)
            self._fitted = True
        except Exception as exc:
            logger.debug("MLX surrogate init failed (%s); will use CPU fallback.", exc)

    def _extract_from_sklearn(self, model: Any) -> None:
        """Extract kernel matrix, alpha vector, and Cholesky factor."""
        import mlx.core as mx

        X_train = np.asarray(model.X_train_, dtype=np.float32)
        n_train = X_train.shape[0]

        # Extract length scale from the RBF kernel: kernel_ = C * RBF(l) + W(n)
        self._length_scale = 1.0
        try:
            rbf = model.kernel_.k1.k2  # ConstantKernel * RBF → .k2 is RBF
            ls = float(rbf.length_scale)
            if np.isfinite(ls) and ls > 0:
                self._length_scale = ls
        except (AttributeError, TypeError):
            pass

        self._train_mlx = mx.array(X_train)

        # alpha_ = K_inv @ (y_normalized) is precomputed by sklearn.
        # sklearn normalizes targets (normalize_y=True), so predictions must
        # be denormalized: y_std * (K_star @ alpha_) + y_mean.
        alpha = np.asarray(model.alpha_, dtype=np.float32)
        self._alpha_mlx = mx.array(alpha.reshape(n_train, 1))
        self._y_mean = float(getattr(model, "_y_train_mean", 0.0))
        self._y_std = float(getattr(model, "_y_train_std", 1.0))

        # L_ = cholesky(K + sigma²I), used for variance computation
        L = np.asarray(model.L_, dtype=np.float32)
        self._L_mlx = mx.array(L)

        # Kernel decomposition for inference:
        #   Cross-kernel K(X*, X_train): amplitude = C (signal only, no noise)
        #   Diagonal k(x*, x*): value = C + σ² (signal + noise)
        try:
            const_k = model.kernel_.k1.k1  # ConstantKernel
            noise_k = model.kernel_.k2  # WhiteKernel
            self._signal_var = float(const_k.constant_value)
            self._noise_var = float(noise_k.noise_level)
        except (AttributeError, TypeError):
            self._signal_var = 1.0
            self._noise_var = 0.0
        self._k_diag_test = self._signal_var + self._noise_var

    @property
    def is_available(self) -> bool:
        return self._fitted

    def _rbf_kernel_matrix(self, X_test: Any, X_train: Any) -> Any:
        """Compute RBF kernel matrix K(X_test, X_train) on MLX.

        Uses the identity ||a-b||² = ||a||² + ||b||² - 2·a·b for efficient
        pairwise squared distance computation. Includes the ConstantKernel
        amplitude (signal variance).
        """
        import mlx.core as mx

        # X_test: (n_test, n_bits), X_train: (n_train, n_bits)
        x_test_sq = (X_test ** 2).sum(axis=1, keepdims=True)  # (n_test, 1)
        x_train_sq = (X_train ** 2).sum(axis=1, keepdims=True)  # (n_train, 1)

        # Pairwise squared distances
        sq_dist = x_test_sq + x_train_sq.T - 2.0 * (X_test @ X_train.T)
        sq_dist = mx.maximum(sq_dist, 0.0)  # numerical safety

        # RBF kernel with signal variance: C * exp(-d² / (2*l²))
        # Cross-kernel has no noise term (test points ≠ training points).
        return self._signal_var * mx.exp(-sq_dist / (2.0 * self._length_scale ** 2))

    def predict_batch(
        self, mols: list[Chem.Mol], return_std: bool = True
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict residuals for a batch of molecules.

        Args:
            mols: List of RDKit molecules.
            return_std: If True, also return posterior standard deviations.

        Returns:
            (means, stds) where means is (n_mols,) and stds is (n_mols,) or None.
        """
        import mlx.core as mx

        if not mols:
            empty = np.array([], dtype=np.float32)
            return empty, empty if return_std else None

        # Encode molecules as ECFP4 vectors
        X_test = np.stack([_ecfp4_dense(m, self._n_bits) for m in mols]).astype(np.float32)
        X_test_mlx = mx.array(X_test)

        # K(X*, X_train)
        K_star = self._rbf_kernel_matrix(X_test_mlx, self._train_mlx)

        # Posterior mean (normalized): μ*_norm = K* @ alpha
        means_mlx = K_star @ self._alpha_mlx  # (n_test, 1)
        # Denormalize: y_std * μ*_norm + y_mean
        means_mlx = means_mlx * self._y_std + self._y_mean
        means = np.array(means_mlx.reshape(-1).astype(mx.float32))

        stds = None
        if return_std:
            # v = L⁻¹ @ K*ᵀ  → solve L @ v = K*ᵀ
            # MLX GPU doesn't support triangular solve, so force CPU stream.
            # The kernel matrix multiply (the expensive O(n_test·n_train) part)
            # still runs on GPU above; this solve is O(n_train²) which is cheap
            # since n_train is small (~115-231).
            K_star_T = K_star.T  # (n_train, n_test)
            v = mx.linalg.solve_triangular(
                self._L_mlx, K_star_T, upper=False, stream=mx.cpu,
            )  # (n_train, n_test)
            # σ² = k(x*,x*) - sum(v², axis=0), then scale by y_std for denormalization
            # k(x*,x*) = signal_var + noise_var (includes noise for test diagonal)
            var = self._k_diag_test - (v ** 2).sum(axis=0)
            var = mx.maximum(var, 0.0)
            stds_mlx = mx.sqrt(var) * self._y_std
            stds = np.array(stds_mlx.astype(mx.float32))

        return means, stds


def predict_deltas_batch_mlx(
    sklearn_gpr: Any,
    mols: list[Chem.Mol],
    return_std: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """One-shot MLX batch prediction for a sklearn GPR model.

    Convenience wrapper that creates a surrogate, runs batch prediction,
    and returns results. Falls back to sklearn if MLX is unavailable.

    Args:
        sklearn_gpr: Fitted GaussianProcessRegressor.
        mols: List of RDKit molecules.
        return_std: Whether to compute posterior std.

    Returns:
        (means, stds) arrays of shape (n_mols,).
    """
    device = get_device()
    if device != "mlx":
        return _predict_deltas_batch_sklearn(sklearn_gpr, mols, return_std)

    surrogate = MLXGPRSurrogate(sklearn_gpr)
    if not surrogate.is_available:
        return _predict_deltas_batch_sklearn(sklearn_gpr, mols, return_std)

    return surrogate.predict_batch(mols, return_std=return_std)


def predict_variance_batch_mlx(
    sklearn_gpr: Any,
    mols: list[Chem.Mol],
) -> np.ndarray:
    """Compute GPR posterior variance for a batch of molecules.

    Ports the posterior variance computation: sigma^2 = k** - v^T v where
    v = L^{-1} @ k(X*, X_train)^T. The kernel matrix multiply runs on the
    MLX GPU (M5 Pro); the triangular solve runs on the CPU stream since
    MLX GPU does not support ``solve_triangular``.

    This is used by BALD acquisition to rank candidates by epistemic
    uncertainty reduction (Phase 2 wiring in experiment_suggester.py).

    Args:
        sklearn_gpr: Fitted GaussianProcessRegressor.
        mols: List of RDKit molecules.

    Returns:
        1-D float32 array of posterior variances, same length as mols.
    """
    means, stds = predict_deltas_batch_mlx(sklearn_gpr, mols, return_std=True)
    if stds is None:
        return np.zeros(len(mols), dtype=np.float32)
    return np.asarray(stds, dtype=np.float32) ** 2


def _predict_deltas_batch_sklearn(
    sklearn_gpr: Any,
    mols: list[Chem.Mol],
    return_std: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """CPU fallback: batch prediction via sklearn."""
    X = np.stack([_ecfp4_dense(m) for m in mols]).astype(np.float64)
    if return_std:
        means, stds = sklearn_gpr.predict(X, return_std=True)
        return means.astype(np.float32), stds.astype(np.float32)
    means = sklearn_gpr.predict(X)
    return means.astype(np.float32), None
