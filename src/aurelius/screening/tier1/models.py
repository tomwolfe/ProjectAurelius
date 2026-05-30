"""Tier 1: Neural network model definitions.

Contains all MLP model classes used for molecular viability screening:
- MLXBackend: MLX-compatible 2-layer MLP for ECFP4 fingerprints
- PyTorchBackend: PyTorch MLP replicating ChemVLM2MLP architecture
- ModelFactory: Returns the appropriate backend based on framework availability

References:
    Delaney, S. J. "ESOL: Estimating Aqueous Solubility
    Directly from Structure." J. Chem. Inf. Model. 2004.
    Glorot, X. & Bengio, Y. "Understanding the difficulty
    of training deep feedforward neural networks." AISTATS 2010.
"""

from __future__ import annotations

import os
from importlib import resources
from typing import Any

import numpy as np

from aurelius.screening.tier1.backend_torch import (
    PyTorchBackend,
    PyTorchBackendUnavailableError,
)
from aurelius.screening.tier1.backend_torch import (
    _model_factory as _torch_model_factory,
)
from aurelius.utils.dependencies import HAS_MLX, HAS_RDKIT, HAS_TORCH

__all__ = [
    "DEFAULT_MODEL_DIR",
    "HAS_MLX",
    "HAS_RDKIT",
    "HAS_TORCH",
    "PyTorchBackend",
    "PyTorchBackendUnavailableError",
    "HUGGINGFACE_MODELS",
    "ModelFactory",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default model weight paths
DEFAULT_MODEL_DIR = os.environ.get(
    "AURELIUS_MODEL_DIR",
    str(resources.files("aurelius").joinpath("models")),
)

HUGGINGFACE_MODELS: dict[str, str] = {
    "esol_solubility": "aurelius/tier1-esol-mlp",
    "qm9_energy": "aurelius/tier1-qm9-mlp",
}


def ModelFactory() -> PyTorchBackend:  # type: ignore[return]  # noqa: N802
    """Return a PyTorch backend for molecular viability screening.

    Returns:
        PyTorchBackend instance.

    Raises:
        PyTorchBackendUnavailableError: If PyTorch is not available.
    """
    if HAS_TORCH:
        return _torch_model_factory()
    raise PyTorchBackendUnavailableError(
        "PyTorch is not available. Install torch to use model backends."
    )


class MLXBackend:
    """MLX-compatible 2-layer MLP for ECFP4 fingerprints.

    This backend provides a simple MLP that maps ECFP4 fingerprints
    to viability predictions. When MLX is unavailable, this class
    raises ``PyTorchBackendUnavailableError``.
    """

    def __init__(self) -> None:
        """Initialize the MLXBackend."""
        self._model: Any | None = None
        self._weights: dict[str, Any] | None = None

    def predict(self, x: Any) -> Any:
        """Run inference on the MLX model.

        Args:
            x: Input tensor or array.

        Returns:
            Predicted output.

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_weights() first.")
        return self._model(x)

    def load_weights(self, weights_dir: str) -> None:
        """Load model weights from a directory.

        Args:
            weights_dir: Directory containing weight files.
        """
        import mlx.core as mx
        import mlx.nn as nn

        weights: dict[str, Any] = {}
        for fname in ["W1", "b1", "W2", "b2"]:
            path = os.path.join(weights_dir, f"{fname}.npy")
            if os.path.isfile(path):
                weights[fname] = mx.array(np.load(path))

        self._weights = weights
        self._model = nn.Sequential(  # type: ignore[attr-defined]
            nn.Linear(2048, 128),  # type: ignore[attr-defined]
            nn.ReLU(),  # type: ignore[attr-defined]
            nn.Linear(128, 1),  # type: ignore[attr-defined]
        )
        if weights:
            self._model.update(nn.state.set_weights(self._model, weights))

    def save_weights(self, path: str) -> None:
        """Save model weights to a directory.

        Args:
            path: Directory to save weights.
        """
        os.makedirs(path, exist_ok=True)
        for name, value in self._weights.items():
            np.save(os.path.join(path, f"{name}.npy"), np.asarray(value))
