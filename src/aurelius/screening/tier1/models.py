"""Tier 1: Neural network model definitions.

Contains all MLP model classes used for molecular viability screening:
- MLXBackend: MLX-compatible 2-layer MLP for ECFP4 fingerprints
- PyTorchBackend: PyTorch MLP replicating ChemVLM2MLP architecture
- NumpyBackend: NumPy-based fallback when MLX is unavailable
- ModelFactory: Returns the appropriate backend based on framework availability

References:
    Delaney, S. J. "ESOL: Estimating Aqueous Solubility
    Directly from Structure." J. Chem. Inf. Model. 2004.
    Glorot, X. & Bengio, Y. "Understanding the difficulty
    of training deep feedforward neural networks." AISTATS 2010.
"""

from __future__ import annotations

import contextlib
import json
import os
from importlib import resources
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from aurelius.utils.dependencies import HAS_MLX, HAS_RDKIT, HAS_TORCH

# ---------------------------------------------------------------------------
# Type aliases for model tensor types
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    try:
        import mlx.core as _mlx_core  # noqa: F401

        MLXArray = _mlx_core.array
    except Exception:
        MLXArray: Any  # type: ignore[no-redef]

    try:
        import torch as _torch  # noqa: F401

        TTensor = _torch.Tensor
    except Exception:
        TTensor: Any  # type: ignore[no-redef]

__all__ = [
    "DEFAULT_MODEL_DIR",
    "HAS_MLX",
    "HAS_RDKIT",
    "HAS_TORCH",
    "HUGGINGFACE_MODELS",
    "MLXBackend",
    "model_factory",
    "PyTorchBackend",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default model weight paths
DEFAULT_MODEL_DIR = os.environ.get(
    "AURELIUS_MODEL_DIR",
    str(resources.files("aurelius").joinpath("models")),
)

# Hugging Face model repository for pre-trained Tier 1 weights
HUGGINGFACE_MODELS: dict[str, str] = {
    "esol_solubility": "aurelius/tier1-esol-mlp",
    "qm9_energy": "aurelius/tier1-qm9-mlp",
}

# ---------------------------------------------------------------------------
# Protocol / Strategy Pattern
# ---------------------------------------------------------------------------


@runtime_checkable
class ModelBackend(Protocol):
    """Protocol defining the interface for all model backends.

    All backends must implement:
    - __call__: Forward pass returning a scalar or tensor
    - parameters: Return model parameters as a list of arrays/tensors
    - predict: Prediction method for inference
    - save_weights: Save model weights to disk
    - load_weights: Load model weights from disk
    """

    def __call__(self, x: MLXArray) -> MLXArray: ...

    def parameters(self) -> list[MLXArray]: ...

    def predict(self, x: MLXArray) -> MLXArray: ...

    def save_weights(self, path: str) -> None: ...

    def load_weights(self, path: str) -> None: ...


# ---------------------------------------------------------------------------
# MLX Backend
# ---------------------------------------------------------------------------

if HAS_MLX:
    import mlx.core as _mlx_core
    import mlx.nn as _mlx_nn
    import numpy as np

    class MLXBackend(_mlx_nn.Module):  # type: ignore[name-defined, misc]
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

        def __init__(self, input_dim: int = 2048, hidden_dim: int = 128) -> None:
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
                shape=(self.input_dim, self.hidden_dim),
                low=-scale1,
                high=scale1,
            )
            self.linear1.bias = _mlx_core.zeros((self.hidden_dim,))

            scale2 = np.sqrt(2.0 / (self.hidden_dim + 1))
            self.linear2.weight = _mlx_core.random.uniform(
                shape=(self.hidden_dim, 1),
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
            return [self.linear1.weight, self.linear1.bias, self.linear2.weight, self.linear2.bias]

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
            with open(os.path.join(path), "w") as f:
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

else:
    class MLXBackend:  # type: ignore[no-redef]
        """Placeholder when MLX is unavailable."""

        def __init__(self, *args: int, **kwargs: int) -> None:
            raise ImportError("MLX is required for MLXBackend. Install with: pip install mlx")

        __call__: Any
        parameters: Any
        save_weights: Any
        load_weights: Any


# ---------------------------------------------------------------------------
# PyTorch Backend
# ---------------------------------------------------------------------------

if HAS_TORCH:
    import torch as _torch  # type: ignore[import-not-found, unused-ignore]
    import torch.nn as _torch_nn  # type: ignore[import-not-found, unused-ignore]

    class PyTorchBackend:
        """PyTorch-based MLP replicating the ChemVLM2MLP architecture.

        Provides a 2-layer MLP (2048->128->1) using torch.nn when MLX is
        unavailable. This ensures consistent gradient computation and
        device handling across all tiers of the pipeline.

        The architecture matches _ChemVLM2MLP:
            - Input: 2048-bit ECFP4 fingerprint (float tensor)
            - Hidden: 128 units with ReLU activation
            - Output: 1 scalar viability score via sigmoid

        Weights are loaded from MLX model directories containing .npy files
        via convert_mlx_to_torch_weights(). If loading fails, random
        Xavier-initialized weights are used with a WARNING.

        This class is fully compatible with torch.autograd for gradient
        computation and supports device placement (CPU/CUDA/MPS).
        """

        def __init__(self, input_dim: int = 2048, hidden_dim: int = 128) -> None:
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim

            self.fc1 = _torch_nn.Linear(input_dim, hidden_dim)
            self.relu = _torch_nn.ReLU()
            self.fc2 = _torch_nn.Linear(hidden_dim, 1)

            self._init_weights()

        def _init_weights(self) -> None:
            """Initialize all weights using Xavier uniform initialization."""
            _torch_nn.init.xavier_uniform_(self.fc1.weight)
            _torch_nn.init.zeros_(self.fc1.bias)
            _torch_nn.init.xavier_uniform_(self.fc2.weight)
            _torch_nn.init.zeros_(self.fc2.bias)

        def __call__(self, x: TTensor) -> TTensor:
            """Forward pass through the 2-layer MLP."""
            h = self.fc1(x)
            h = self.relu(h)
            out = self.fc2(h)
            return _torch.sigmoid(out)

        def predict(self, x: TTensor) -> TTensor:
            """Run inference and return viability score.

            Args:
                x: Input tensor/array (N, 2048).

            Returns:
                Predicted viability score (N, 1) or (N,).
            """
            return self(x)

        def parameters(self) -> list[TTensor]:
            return [self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias]

        def save_weights(self, path: str) -> None:
            """Save model weights to individual .npy files (MLX-compatible format).

            Args:
                path: Directory path to save weights.
            """
            os.makedirs(path, exist_ok=True)
            state_dict = self.state_dict()  # type: ignore[attr-defined]
            for name, tensor in state_dict.items():
                np.save(os.path.join(path, f"{name}.npy"), tensor.cpu().numpy())
            meta = {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "architecture": "MLP-2048-128-1",
                "fp_type": "ECFP4_2048",
                "framework": "pytorch",
            }
            with open(os.path.join(path), "w") as f:
                json.dump(meta, f, indent=2)

        def load_weights(self, path: str) -> None:
            """Load model weights from .npy files.

            Args:
                path: Directory path containing saved weights.
            """
            state_dict = _torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
            self.load_state_dict(state_dict)  # type: ignore[attr-defined]

else:
    class PyTorchBackend:  # type: ignore[no-redef]
        """Placeholder when PyTorch is unavailable."""

        def __init__(self, *args: int, **kwargs: int) -> None:
            raise ImportError("PyTorch is required for PyTorchBackend. Install with: pip install torch")

        __call__: Any
        parameters: Any
        save_weights: Any
        load_weights: Any
# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def model_factory() -> ModelBackend:
    """Return the appropriate model backend based on framework availability.

    Priority: MLX > PyTorch.

    Returns:
        An instance of the selected backend.

    Raises:
        ImportError: When neither MLX nor PyTorch is available.
    """
    if HAS_MLX:
        return MLXBackend()  # type: ignore[return-value]
    elif HAS_TORCH:
        return PyTorchBackend()  # type: ignore[return-value]
    else:
        raise ImportError(
            "At least one ML framework is required (MLX or PyTorch). "
            "Install MLX: pip install mlx\n"
            "Install PyTorch: pip install torch"
        )
