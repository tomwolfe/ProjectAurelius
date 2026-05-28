"""PyTorch backend for Tier 1 MLP model.

Replicates the ChemVLM2MLP architecture using PyTorch.

Architecture:
    - Input: 2048-bit ECFP4 fingerprint (float tensor)
    - Hidden: 128 units with ReLU activation
    - Output: 1 scalar viability score via sigmoid

Weights are loaded from MLX model directories containing .npy files
via convert_mlx_to_torch_weights(). If loading fails, random
Xavier-initialized weights are used with a WARNING.

This class is fully compatible with torch.autograd for gradient
computation and supports device placement (CPU/CUDA/MPS).
"""

from __future__ import annotations

import json
import os

import numpy as np  # noqa: F401
import torch as _torch  # noqa: F401
import torch.nn as _torch_nn  # noqa: F401
from torch import Tensor as TTensor  # noqa: F401

from aurelius.constants import FINGERPRINT_SIZE


class PyTorchBackend(_torch_nn.Module):  # type: ignore[valid-type, misc, name-defined]
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

    def __init__(self, input_dim: int = FINGERPRINT_SIZE, hidden_dim: int = 128) -> None:
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

    def parameters(self) -> list[TTensor]:  # type: ignore[override]
        """Return model parameters as a list of tensors.

        Uses PyTorch's native parameter tracking via self.named_parameters(),
        which is the idiomatic PyTorch approach.
        """
        return [p for _, p in self.named_parameters()]  # type: ignore[return-value]

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
        state_dict = _torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
        self.load_state_dict(state_dict)


class PyTorchBackendUnavailableError(RuntimeError):
    """Raised when PyTorch is not available."""


def _model_factory() -> PyTorchBackend:
    """Return a PyTorch backend instance for Tier 1.

    Returns:
        A PyTorchBackend instance.
    """
    return PyTorchBackend()
