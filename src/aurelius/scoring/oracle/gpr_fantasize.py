"""Fantasization-based batch acquisition for active learning.

Provides rank-1 GPR posterior updates that enable efficient simulation of
"what if we measured this candidate?" without refitting. Used by the
fantasization-based batch Expected Improvement acquisition in
``experiment_suggester.py``.

The core operation: given a fitted GPR with Cholesky factor L (where
K + sigma^2*I = L L^T), simulate adding a new training point (x*, y*) and
compute the posterior variance at test points under the updated model.

For a Gaussian posterior, adding a training point can only reduce posterior
variance. The reduction is computable in O(n_train^2) via a rank-1 Cholesky
update, versus O(n_train^3) for a full refit.

Reference: Chandra et al., "Fantasizing with the Batch Bayesian Optimization
Framework" (the "fantasize-and-observe" paradigm).
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def extract_gpr_state(gpr_model: object) -> dict[str, float | NDArray[np.float64]]:
    """Extract kernel parameters and precomputed quantities from a fitted sklearn GPR.

    Returns a dict with:
        X_train: (n_train, n_features) training feature matrix
        L: (n_train, n_train) Cholesky factor of K + sigma^2*I
        alpha: (n_train,) precomputed K_inv @ y_normalized
        y_mean: target normalization mean
        y_std: target normalization std
        signal_var: signal variance (ConstantKernel amplitude)
        noise_var: noise variance (WhiteKernel level)
        length_scale: RBF length scale
    """
    X_train = np.asarray(gpr_model.X_train_, dtype=np.float64)
    L = np.asarray(gpr_model.L_, dtype=np.float64)
    alpha = np.asarray(gpr_model.alpha_, dtype=np.float64)
    y_mean = float(getattr(gpr_model, "_y_train_mean", 0.0))
    y_std = float(getattr(gpr_model, "_y_train_std", 1.0))

    signal_var = 1.0
    noise_var = 0.1
    length_scale = 1.0
    try:
        kernel = gpr_model.kernel_
        const_k = kernel.k1.k1  # ConstantKernel
        rbf = kernel.k1.k2  # RBF
        noise_k = kernel.k2  # WhiteKernel
        signal_var = float(const_k.constant_value)
        noise_var = float(noise_k.noise_level)
        ls = float(rbf.length_scale)
        if np.isfinite(ls) and ls > 0:
            length_scale = ls
    except (AttributeError, TypeError):
        pass

    return {
        "X_train": X_train,
        "L": L,
        "alpha": alpha,
        "y_mean": y_mean,
        "y_std": y_std,
        "signal_var": signal_var,
        "noise_var": noise_var,
        "length_scale": length_scale,
    }


def rbf_kernel(X: NDArray[np.float64], Y: NDArray[np.float64], signal_var: float, length_scale: float) -> NDArray[np.float64]:
    """RBF (squared exponential) kernel with signal variance.

    K(X, Y) = signal_var * exp(-||x - y||^2 / (2 * length_scale^2))
    """
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)
    Y_sq = np.sum(Y ** 2, axis=1, keepdims=True)
    sq_dist = X_sq + Y_sq.T - 2.0 * (X @ Y.T)
    sq_dist = np.maximum(sq_dist, 0.0)
    return signal_var * np.exp(-sq_dist / (2.0 * length_scale ** 2))


def rank1_cholesky_update(L: NDArray[np.float64], k_star: NDArray[np.float64], k_diag: float, noise_var: float) -> NDArray[np.float64]:
    """Compute the updated Cholesky factor after adding a training point.

    Given the current Cholesky L (n x n) where K + sigma^2*I = L L^T,
    and a new point x* with kernel vector k* = K(X_train, x*) and
    k** = K(x*, x*), compute L_new such that:

        [ K + sigma^2*I   k*  ]   [ L   0 ] [ L^T  w^T ]
        [ k*^T            k** + s^2 ] = [ w   r ] [  0   r ]

    where w = L^{-1} @ k* and r = sqrt(k** + s^2 - ||w||^2).

    Returns L_new: (n+1, n+1) updated Cholesky factor.
    """
    n = L.shape[0]
    # Solve L @ w = k* via forward substitution
    w = np.linalg.solve(L, k_star)
    # Residual variance
    residual = k_diag + noise_var - np.dot(w, w)
    r = 1e-10 if residual <= 0 else np.sqrt(residual)

    L_new = np.zeros((n + 1, n + 1), dtype=np.float64)
    L_new[:n, :n] = L
    L_new[n, :n] = w
    L_new[n, n] = r
    return L_new


def posterior_variance_after_update(
    state: dict[str, float | NDArray[np.float64]],
    x_star: NDArray[np.float64],
    X_test: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute posterior variance at X_test after fantasizing x_star as a training point.

    Uses the rank-1 Cholesky update to avoid refitting. The variance reduction
    from adding x_star is proportional to K(X_test, x_star)^2 / (k** + sigma^2 - ||w||^2),
    which is always non-negative (adding data never increases variance).

    Args:
        state: GPR state dict from extract_gpr_state.
        x_star: (n_features,) the candidate to fantasize.
        X_test: (n_test, n_features) test points.

    Returns:
        (n_test,) posterior variance at X_test after the update.
    """
    X_train = state["X_train"]
    L = state["L"]
    signal_var = state["signal_var"]
    noise_var = state["noise_var"]
    length_scale = state["length_scale"]

    # Kernel between training set and x_star
    k_star = rbf_kernel(X_train, x_star.reshape(1, -1), signal_var, length_scale).ravel()
    # Self-kernel at x_star
    k_diag = signal_var  # K(x*, x*) = signal_var (RBF at zero distance)

    # Current posterior variance at X_test (before update)
    K_test = rbf_kernel(X_test, X_train, signal_var, length_scale)
    v = np.linalg.solve(L, K_test.T)  # (n_train, n_test)
    var_current = signal_var + noise_var - np.sum(v ** 2, axis=0)

    # Rank-1 update
    w = np.linalg.solve(L, k_star)
    denominator = k_diag + noise_var - np.dot(w, w)
    if denominator <= 1e-12:
        return np.maximum(var_current, 0.0)

    # Kernel between test points and x_star
    k_test_star = rbf_kernel(X_test, x_star.reshape(1, -1), signal_var, length_scale).ravel()
    # Solve L @ u = k* for the update term
    # Variance reduction: (K_test_star - K_test @ L^{-T} @ L^{-1} @ k*)^2 / denominator
    # = (k_test_star - v^T @ w)^2 / denominator
    update_term = (k_test_star - v.T @ w) ** 2 / denominator
    var_new = var_current - update_term
    return np.maximum(var_new, 0.0)


def fantasize_batch_scores(
    state: dict[str, float | NDArray[np.float64]],
    X_pool: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Score each candidate by expected variance reduction across the whole pool.

    For each candidate x_i in the pool, simulate adding it to the GPR training
    set and compute the total epistemic variance reduction across all *other*
    candidates. This captures subset-level information that greedy per-molecule
    scoring misses: a candidate that reduces uncertainty on many diverse
    molecules scores higher than one that reduces uncertainty only locally.

    Computational cost: O(n_pool * (n_train^2 + n_pool * n_train)) which is
    O(n_pool^2 * n_train) for n_pool >> n_train. For typical values
    (n_pool=200, n_train=70) this is ~200^2 * 70 = 2.8M kernel evaluations —
    fast in numpy.

    Args:
        state: GPR state dict from extract_gpr_state.
        X_pool: (n_pool, n_features) candidate feature matrix.

    Returns:
        (n_pool,) scores — higher means more informative if measured.
    """
    n_pool = X_pool.shape[0]
    if n_pool < 2:
        return np.zeros(n_pool, dtype=np.float64)

    scores = np.zeros(n_pool, dtype=np.float64)
    for i in range(n_pool):
        x_star = X_pool[i]
        # Variance at all OTHER candidates after fantasizing x_star
        mask = np.ones(n_pool, dtype=bool)
        mask[i] = False
        X_others = X_pool[mask]

        var_new = posterior_variance_after_update(state, x_star, X_others)

        # Current variance at others
        X_train = state["X_train"]
        L = state["L"]
        signal_var = state["signal_var"]
        noise_var = state["noise_var"]
        length_scale = state["length_scale"]
        K_others = rbf_kernel(X_others, X_train, signal_var, length_scale)
        v = np.linalg.solve(L, K_others.T)
        var_current = signal_var + noise_var - np.sum(v ** 2, axis=0)

        # Score = total variance reduction
        reduction = np.sum(var_current - var_new)
        scores[i] = max(reduction, 0.0)

    return scores
