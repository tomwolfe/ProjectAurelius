"""Tier 1: Weight loading utilities.

Handles loading pre-trained model weights from HuggingFace Hub,
local directories, and converting between MLX and PyTorch formats.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from aurelius.screening.tier1.models import (
    DEFAULT_MODEL_DIR,
    HAS_TORCH,
    HUGGINGFACE_MODELS,
    PyTorchFallbackFilter,
    _ChemVLM2MLP,
)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment, unused-ignore]
    HAS_TORCH = False


def convert_mlx_to_torch_weights(mlx_weights_dir: str) -> dict[str, "torch.Tensor"]:
    """Convert MLX model weights (stored as .npy files) to PyTorch tensors.

    Loads .npy files from the MLX model directory and converts them
    directly to PyTorch tensors using torch.from_numpy(). Since the
    architecture is a standard MLP (2048->128->1), no complex topology
    mapping is needed - the NumPy arrays map directly to PyTorch tensor
    shapes.

    Args:
        mlx_weights_dir: Path to the directory containing MLX model weights
            (.npy files for W1, b1, W2, b2).

    Returns:
        Dictionary mapping parameter names to PyTorch tensors.
        Returns empty dict if loading fails or torch is unavailable.
    """
    if not HAS_TORCH:
        print("[Aurelius v6.0 Tier1] WARNING: PyTorch is not available. Cannot convert MLX weights.")
        return {}

    if not os.path.isdir(mlx_weights_dir):
        print(f"[Aurelius v6.0 Tier1] WARNING: MLX weights directory not found: {mlx_weights_dir}")
        return {}

    weight_files: dict[str, Any | None] = {
        "W1": None,
        "b1": None,
        "W2": None,
        "b2": None,
    }

    for fname in weight_files:
        fpath = os.path.join(mlx_weights_dir, f"{fname}.npy")
        if os.path.isfile(fpath):
            weight_files[fname] = np.load(fpath)
        else:
            print(f"[Aurelius v6.0 Tier1] WARNING: Missing weight file: {fpath}")
            return {}

    expected_shapes: dict[str, tuple[int, ...]] = {
        "W1": (2048, 128),
        "b1": (128,),
        "W2": (128, 1),
        "b2": (1,),
    }

    torch_weights: dict[str, torch.Tensor] = {}
    for name, arr in weight_files.items():
        if arr is None:
            continue
        expected = expected_shapes.get(name)
        if expected and arr.shape != expected:
            print(
                f"[Aurelius v6.0 Tier1] WARNING: Shape mismatch for {name}: "
                f"expected {expected}, got {arr.shape}. "
                "Using uninitialized PyTorch fallback weights. "
                "Run `aurelius train --task tier1` to train properly."
            )
            return {}
        torch_weights[name] = torch.from_numpy(arr.astype(np.float32))

    return torch_weights


def load_pytorch_fallback_with_mlx_weights(
    model: PyTorchFallbackFilter,
    mlx_weights_dir: str,
) -> PyTorchFallbackFilter:
    """Load PyTorch fallback model with weights converted from MLX format.

    Args:
        model: The PyTorchFallbackFilter instance to load weights into.
        mlx_weights_dir: Path to the MLX model weights directory.

    Returns:
        The PyTorchFallbackFilter with loaded (or randomly initialized) weights.
    """
    torch_weights = convert_mlx_to_torch_weights(mlx_weights_dir)

    if not torch_weights:
        print(
            "[Aurelius v6.0 Tier1] WARNING: Using uninitialized PyTorch fallback weights. "
            "Run `aurelius train --task tier1` to train properly."
        )
        return model

    state_mapping: dict[str, str] = {
        "W1": "fc1.weight",
        "b1": "fc1.bias",
        "W2": "fc2.weight",
        "b2": "fc2.bias",
    }

    state_dict: dict[str, torch.Tensor] = {}
    for mlx_name, torch_key in state_mapping.items():
        if mlx_name in torch_weights:
            state_dict[torch_key] = torch_weights[mlx_name]

    if state_dict:
        model.load_state_dict(state_dict, strict=False)
        print(f"[Aurelius v6.0 Tier1] Loaded PyTorch fallback weights from MLX: {mlx_weights_dir}")
    else:
        print(
            "[Aurelius v6.0 Tier1] WARNING: Could not map any weights from MLX format. "
            "Using random initialization."
        )

    return model


class HuggingFaceWeightLoader:
    """Load pre-trained model weights from Hugging Face Hub.

    Attempts to download pre-trained Tier 1 model weights from
    Hugging Face Hub. Falls back gracefully to local weights
    or re-training if neither is available.

    Supported models:
        - ESOL solubility predictor (logS prediction)
        - QM9 energy predictor (DFT-computed energies)
    """

    def __init__(self, model_dir: str | None = None) -> None:
        """Initialize the weight loader.

        Args:
            model_dir: Local directory to cache model weights.
                Defaults to AURELIUS_MODEL_DIR env var or
                <repo_root>/models/.
        """
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        self._hf_available = self._check_hf_dependencies()

    def _check_hf_dependencies(self) -> bool:
        """Check if huggingface_hub and datasets are available."""
        try:
            __import__("huggingface_hub")
            __import__("datasets")
            return True
        except ImportError:
            return False

    def load_model(
        self,
        task: str = "esol_solubility",
        local_only: bool = False,
    ) -> _ChemVLM2MLP | None:
        """Load a pre-trained model from Hugging Face Hub.

        Args:
            task: Model task identifier ("esol_solubility" or "qm9_energy").
            local_only: If True, only load from local files.

        Returns:
            _ChemVLM2MLP with loaded weights, or None if unavailable.
        """
        model_id = HUGGINGFACE_MODELS.get(task)
        if model_id is None:
            return None

        if self._hf_available and not local_only:
            model = self._load_from_hf_hub(model_id, task)
            if model is not None:
                return model

        if os.path.isdir(self.model_dir):
            model = self._load_from_local(task)
            if model is not None:
                return model

        return None

    def _load_from_hf_hub(self, model_id: str, task: str) -> _ChemVLM2MLP | None:
        """Attempt to load model weights from Hugging Face Hub.

        Args:
            model_id: Hugging Face model repository ID.
            task: Model task identifier.

        Returns:
            _ChemVLM2MLP with loaded weights, or None on failure.
        """
        try:
            from huggingface_hub import snapshot_download

            local_dir = os.path.join(self.model_dir, task, "hf_cache")
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
            )

            model = _ChemVLM2MLP()
            model.load_weights(local_dir)
            print(f"[Aurelius v5.2 Tier1] Loaded {task} model from Hugging Face Hub: {model_id}")
            return model

        except ImportError as e:
            print(f"[Aurelius v5.2 Tier1] Hugging Face import failed: {e}")
            return None
        except ValueError as e:
            print(f"[Aurelius v5.2 Tier1] Invalid model ID (ValueError): {e}")
            return None
        except ConnectionError as e:
            print(f"[Aurelius v5.2 Tier1] Network error from HF Hub: {e}")
            return None
        except Exception as e:
            print(f"[Aurelius v5.2 Tier1] HF Hub download failed: {e}")
            return None

    def _load_from_local(self, task: str) -> _ChemVLM2MLP | None:
        """Load model weights from local directory.

        Args:
            task: Model task identifier.

        Returns:
            _ChemVLM2MLP with loaded weights, or None if not found.
        """
        local_path = os.path.join(self.model_dir, task)
        if not os.path.isdir(local_path):
            return None

        try:
            model = _ChemVLM2MLP()
            model.load_weights(local_path)
            print(f"[Aurelius v5.2 Tier1] Loaded {task} model from local: {local_path}")
            return model
        except Exception as e:
            print(f"[Aurelius v5.2 Tier1] Local load failed: {e}")
            return None

    def save_model(self, model: _ChemVLM2MLP, task: str) -> str:
        """Save trained model weights to local directory.

        Args:
            model: The trained _ChemVLM2MLP instance.
            task: Model task identifier.

        Returns:
            Path to the saved model directory.
        """
        save_path = os.path.join(self.model_dir, task)
        model.save_weights(save_path)
        print(f"[Aurelius v5.2 Tier1] Saved {task} model to: {save_path}")
        return save_path


__all__ = [
    "DEFAULT_MODEL_DIR",
    "HUGGINGFACE_MODELS",
    "HuggingFaceWeightLoader",
    "convert_mlx_to_torch_weights",
    "load_pytorch_fallback_with_mlx_weights",
]
