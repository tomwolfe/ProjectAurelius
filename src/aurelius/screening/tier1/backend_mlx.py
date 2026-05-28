"""MLX backend for Tier 1 MLP model.

Contains the MLX-compatible 2-layer MLP for ECFP4 fingerprints used
for molecular viability scoring.

Architecture:
    - Input: 2048-bit ECFP4 fingerprint (float array)
    - Hidden: 128 units with ReLU activation
    - Output: 1 scalar viability score via sigmoid

Weights are initialized using Xavier/Glorot initialization
to ensure non-zero gradients during training and meaningful
inference output without requiring a pre-trained model.

Inherits from mlx.nn.Module for compatibility with MLX's
optimization and compilation graph.
"""

from __future__ import annotations

import json
import os
from typing import Any

import mlx.core as _mlx_core  # noqa: F401
import mlx.nn as _mlx_nn  # noqa: F401
import numpy as np  # noqa: F401

from aurelius.constants import FINGERPRINT_SIZE

try:
    from mlx.core import Array as MLXArray  # type: ignore[attr-defined]
except ImportError:
    MLXArray = Any  # type: ignore[misc, assignment]


class MLXBackend(_mlx_nn.Module):  # type: ignore[valid-type, misc, name-defined]
    """MLX-compatible 2-layer MLP for molecular viability scoring.

    Input: 2048-bit ECFP4 fingerprint (float array).
    Hidden: 128 units with ReLU activation.
    Output: 1 scalar viability score via sigmoid.

    Weights are initialized using Xavier/Glorot initialization
    to ensure non-zero gradients during training and meaningful
    inference output without requiring a pre-trained model.

    Inherits from mlx.nn.Module for compatibility with MLX's
    optimization and compilation graph.
    """

    def __init__(self, input_dim: int = FINGERPRINT_SIZE, hidden_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.linear1 = _mlx_nn.Linear(input_dim, hidden_dim)  # type: ignore[attr-defined]
        self.relu = _mlx_nn.ReLU()  # type: ignore[attr-defined]
        self.linear2 = _mlx_nn.Linear(hidden_dim, 1)  # type: ignore[attr-defined]

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize all weights using Xavier uniform initialization."""
        scale1 = np.sqrt(2.0 / (self.input_dim + self.hidden_dim))
        self.linear1.weight = _mlx_core.random.uniform(
            shape=(self.hidden_dim, self.input_dim),
            low=-scale1,
            high=scale1,
        )
        self.linear1.bias = _mlx_core.zeros((self.hidden_dim,))

        scale2 = np.sqrt(2.0 / (self.hidden_dim + 1))
        self.linear2.weight = _mlx_core.random.uniform(
            shape=(1, self.hidden_dim),
            low=-scale2,
            high=scale2,
        )
        self.linear2.bias = _mlx_core.zeros((1,))

    def __call__(self, x: MLXArray) -> MLXArray:
        """Forward pass through the 2-layer MLP."""
        h = self.linear1(x)
        h = self.relu(h)
        out = self.linear2(h)
        return _mlx_nn.sigmoid(out)  # type: ignore[no-any-return, attr-defined]

    def predict(self, x: MLXArray) -> MLXArray:
        """Run inference and return viability score.

        Args:
            x: Input tensor/array (N, 2048).

        Returns:
            Predicted viability score (N, 1) or (N,).
        """
        return self(x)

    def parameters(self) -> list[MLXArray]:
        """Return model parameters as a list of arrays.

        Uses MLX's native parameter tracking via the module's
        attribute introspection, consistent with MLX conventions.
        """
        params: list[MLXArray] = []
        for name in ["linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias"]:
            attr = getattr(self, name.split(".")[0])
            if name.count(".") == 1:
                sub_attr = name.split(".")[1]
                params.append(getattr(attr, sub_attr))
        return params

    def save_weights(self, path: str) -> None:
        """Save model weights to individual .npy files.

        Args:
            path: Directory path to save weights.
        """
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "W1.npy"), np.asarray(self.linear1.weight))
        np.save(os.path.join(path, "b1.npy"), np.asarray(self.linear1.bias))
        np.save(os.path.join(path, "W2.npy"), np.asarray(self.linear2.weight))
        np.save(os.path.join(path, "b2.npy"), np.asarray(self.linear2.bias))
        meta = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "architecture": "MLP-2048-128-1",
            "fp_type": "ECFP4_2048",
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    def load_weights(self, path: str) -> None:
        """Load model weights from individual .npy files.

        Args:
            path: Directory path containing saved weights.

        Raises:
            FileNotFoundError: If weight files are not found.
        """
        W1 = np.load(os.path.join(path, "W1.npy"))
        b1 = np.load(os.path.join(path, "b1.npy"))
        W2 = np.load(os.path.join(path, "W2.npy"))
        b2 = np.load(os.path.join(path, "b2.npy"))
        self.linear1.weight = _mlx_core.array(W1)
        self.linear1.bias = _mlx_core.array(b1)
        self.linear2.weight = _mlx_core.array(W2)
        self.linear2.bias = _mlx_core.array(b2)


class MLXBackendUnavailableError(RuntimeError):
    """Raised when MLX is not available."""


def _model_factory() -> MLXBackend:
    """Return an MLX backend instance for Tier 1.

    Returns:
        An MLXBackend instance.
    """
    return MLXBackend()
