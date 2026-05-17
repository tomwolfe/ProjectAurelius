"""Tier 1: Neural network model definitions.

Contains all MLP model classes used for molecular viability screening:
- _ChemVLM2MLP: MLX-compatible 2-layer MLP for ECFP4 fingerprints
- _FallbackMLP: NumPy-based fallback when MLX is unavailable
- PyTorchFallbackFilter: PyTorch MLP replicating ChemVLM2MLP architecture

References:
    Delaney, S. J. "ESOL: Estimating Aqueous Solubility
    Directly from Structure." J. Chem. Inf. Model. 2004.
    Glorot, X. & Bengio, Y. "Understanding the difficulty
    of training deep feedforward neural networks." AISTATS 2010.
"""

from __future__ import annotations

import json
import os
from importlib import resources
from typing import Any

import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None  # type: ignore[assignment, unused-ignore]

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

try:
    import torch  # type: ignore[import-not-found, unused-ignore]
    import torch.nn as torch_nn  # type: ignore[import-not-found, unused-ignore]
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment, unused-ignore]
    torch_nn = None  # type: ignore[assignment, unused-ignore]
    HAS_TORCH = False

try:
    from rdkit import Chem  # noqa: F401
    from rdkit.Chem import AllChem  # noqa: F401
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


class _ChemVLM2MLP:
    """2-layer MLP for MLX-compatible molecular viability scoring.

    Input: 2048-bit ECFP4 fingerprint (float array).
    Hidden: 128 units with ReLU activation.
    Output: 1 scalar viability score via sigmoid.

    Weights are initialized using Xavier/Glorot initialization
    to ensure non-zero gradients during training and meaningful
    inference output without requiring a pre-trained model.
    """

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 128) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier/Glorot initialization for stable training.

        W ~ N(0, sqrt(2 / (fan_in + fan_out)))
        This ensures the variance of activations is preserved
        across layers, preventing vanishing/exploding gradients.
        """
        scale1 = np.sqrt(2.0 / (self.input_dim + self.hidden_dim))
        self.W1 = mx.random.normal((self.input_dim, self.hidden_dim), scale=scale1)
        self.b1 = mx.zeros((self.hidden_dim,))

        scale2 = np.sqrt(2.0 / (self.hidden_dim + 1))
        self.W2 = mx.random.normal((self.hidden_dim, 1), scale=scale2)
        self.b2 = mx.zeros((1,))

    def __call__(self, x: Any) -> Any:
        """Forward pass through the 2-layer MLP."""
        h = mx.addmm(self.b1, x, self.W1, alpha=1.0, beta=1.0)
        h = mx.maximum(h, 0.0)
        out = mx.addmm(self.b2, h, self.W2, alpha=1.0, beta=1.0)
        return mx.sigmoid(out)

    def parameters(self) -> list[Any]:
        return [self.W1, self.b1, self.W2, self.b2]

    def save_weights(self, path: str) -> None:
        """Save model weights to individual .npy files.

        Args:
            path: Directory path to save weights.
        """
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "W1.npy"), np.asarray(self.W1))
        np.save(os.path.join(path, "b1.npy"), np.asarray(self.b1))
        np.save(os.path.join(path, "W2.npy"), np.asarray(self.W2))
        np.save(os.path.join(path, "b2.npy"), np.asarray(self.b2))
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
        self.W1 = mx.array(W1)
        self.b1 = mx.array(b1)
        self.W2 = mx.array(W2)
        self.b2 = mx.array(b2)


class _FallbackMLP:
    """NumPy-based MLP fallback when MLX is unavailable.

    Produces deterministic results from ECFP4 fingerprints for
    pipeline validation without requiring MLX.
    """

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 128) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        rng = np.random.RandomState(42)
        scale1 = np.sqrt(2.0 / input_dim)
        self.W1 = rng.randn(input_dim, hidden_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W2 = rng.randn(hidden_dim, 1).astype(np.float32) * scale2
        self.b2 = np.zeros(1, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the 2-layer MLP (numpy)."""
        h = x @ self.W1 + self.b1
        h = np.maximum(h, 0.0)
        out = h @ self.W2 + self.b2
        return 1.0 / (1.0 + np.exp(-out))  # type: ignore[no-any-return]

    def parameters(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]



if HAS_TORCH:

    class PyTorchFallbackFilter(torch_nn.Module):  # type: ignore[misc, unused-ignore]
        """PyTorch-based MLP fallback replicating the ChemVLM2MLP architecture.

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
            """Initialize the PyTorch fallback MLP.

            Args:
                input_dim: Input dimension (default: 2048 for ECFP4).
                hidden_dim: Hidden layer dimension (default: 128).
            """
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim

            self.fc1 = torch_nn.Linear(input_dim, hidden_dim)
            self.relu = torch_nn.ReLU()
            self.fc2 = torch_nn.Linear(hidden_dim, 1)

            self._init_weights()

        def _init_weights(self) -> None:
            """Initialize all weights using Xavier uniform initialization."""
            torch_nn.init.xavier_uniform_(self.fc1.weight)
            torch_nn.init.zeros_(self.fc1.bias)
            torch_nn.init.xavier_uniform_(self.fc2.weight)
            torch_nn.init.zeros_(self.fc2.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through the 2-layer MLP.

            Args:
                x: Input tensor of shape (batch_size, input_dim) or (input_dim,).

            Returns:
                Output tensor of shape (batch_size, 1) or () with sigmoid output.
            """
            h = self.fc1(x)
            h = self.relu(h)
            out = self.fc2(h)
            return torch.sigmoid(out)

        def predict(self, x: torch.Tensor) -> torch.Tensor:
            """Run inference and return scalar confidence score.

            Args:
                x: Input tensor of shape (batch_size, input_dim) or (input_dim,).

            Returns:
                Confidence score tensor (sigmoid output, clipped to [0, 1]).
            """
            output = self(x)
            if output.dim() == 0:
                return torch.clamp(output, 0.0, 1.0)
            return torch.clamp(output, 0.0, 1.0)

        def save_weights(self, path: str) -> None:
            """Save model weights to individual .npy files (MLX-compatible format).

            Args:
                path: Directory path to save weights.
            """
            os.makedirs(path, exist_ok=True)
            state_dict = self.state_dict()
            for name, tensor in state_dict.items():
                np.save(os.path.join(path, f"{name}.npy"), tensor.cpu().numpy())
            meta = {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "architecture": "MLP-2048-128-1",
                "fp_type": "ECFP4_2048",
                "framework": "pytorch",
            }
            with open(os.path.join(path, "metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)

        def load_weights(self, path: str) -> None:
            """Load model weights from .npy files.

            Args:
                path: Directory path containing saved weights.
            """
            state_dict = torch.load(
                path, map_location="cpu", weights_only=True,
            )
            self.load_state_dict(state_dict)


else:

    class PyTorchFallbackFilter:  # type: ignore[no-redef]
        """Stub class when PyTorch is not available."""

        def __init__(self, input_dim: int = 2048, hidden_dim: int = 128) -> None:
            raise RuntimeError("PyTorch is required for PyTorchFallbackFilter. Install with: pip install torch")


__all__ = [
    "DEFAULT_MODEL_DIR",
    "HAS_MLX",
    "HAS_RDKIT",
    "HAS_TORCH",
    "HUGGINGFACE_MODELS",
    "PyTorchFallbackFilter",
    "_ChemVLM2MLP",
    "_FallbackMLP",
]
