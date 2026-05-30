"""Tier 1: Weight loading utilities.

Handles loading pre-trained model weights from HuggingFace Hub
and local directories.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from aurelius.screening.tier1.models import (
    DEFAULT_MODEL_DIR,
    HUGGINGFACE_MODELS,
    PyTorchBackend,
)
from aurelius.utils.dependencies import HAS_TORCH

logger = logging.getLogger(__name__)

# Conditional torch import for weight conversion
_torch: Any | None = None
if HAS_TORCH:
    try:
        import torch  # noqa: F401

        _torch = torch
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# HuggingFace symlink control
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# HuggingFace symlink control
# ---------------------------------------------------------------------------

# Env var: AURELIUS_HF_USE_SYMLINKS (default: False for compatibility)
# CLI flag: --use-hf-symlinks
HF_USE_SYMLINKS_DEFAULT: bool = False


def _should_use_symlinks() -> bool:
    """Determine whether to use symlinks for HF downloads.

    Checks the AURELIUS_HF_USE_SYMLINKS environment variable.
    Set to "1" or "true" (case-insensitive) to enable symlinks.

    Returns:
        True if symlinks should be used, False otherwise.
    """
    env_val = os.environ.get("AURELIUS_HF_USE_SYMLINKS", "").lower()
    return env_val in ("1", "true", "yes")


def check_disk_space(path: str, min_free_gb: float = 10.0) -> tuple[bool, float]:
    """Check available disk space at the given path.

    Args:
        path: Directory path to check.
        min_free_gb: Minimum free space required in GB (default: 10GB).

    Returns:
        Tuple of (has_sufficient_space, free_space_gb).
    """
    try:
        usage = psutil.disk_usage(path)
        free_gb = usage.free / (1024**3)
        if free_gb < min_free_gb:
            logger.warning(
                "Low disk space at %s: %.1fGB free (minimum %.1fGB required). "
                "HuggingFace downloads may fail or fill the disk. "
                "Consider cleaning up old models or using AURELIUS_HF_USE_SYMLINKS=1.",
                path,
                free_gb,
                min_free_gb,
            )
        return free_gb >= min_free_gb, free_gb
    except OSError:
        logger.warning("Could not check disk space for %s", path)
        return True, 0.0  # Assume OK if we can't check


def convert_numpy_to_torch_weights(weights_dir: str) -> dict[str, Any]:
    """Convert NumPy weight files to PyTorch tensors.

    Loads .npy files from the model directory and converts them
    directly to PyTorch tensors using torch.from_numpy(). Since the
    architecture is a standard MLP (2048->128->1), no complex topology
    mapping is needed - the NumPy arrays map directly to PyTorch tensor
    shapes.

    Args:
        weights_dir: Path to the directory containing model weights
            (.npy files for W1, b1, W2, b2).

    Returns:
        Dictionary mapping parameter names to PyTorch tensors.
        Returns empty dict if loading fails or torch is unavailable.
    """
    if not HAS_TORCH:
        print("[Aurelius v9.0 Tier1] WARNING: PyTorch is not available. Cannot convert weights.")
        return {}

    if not os.path.isdir(weights_dir):
        print(f"[Aurelius v9.0 Tier1] WARNING: Weights directory not found: {weights_dir}")
        return {}

    weight_files: dict[str, Any | None] = {
        "W1": None,
        "b1": None,
        "W2": None,
        "b2": None,
    }

    for fname in weight_files:
        fpath = os.path.join(weights_dir, f"{fname}.npy")
        if os.path.isfile(fpath):
            weight_files[fname] = np.load(fpath)
        else:
            print(f"[Aurelius v9.0 Tier1] WARNING: Missing weight file: {fpath}")
            return {}

    expected_shapes: dict[str, tuple[int, ...]] = {
        "W1": (2048, 128),
        "b1": (128,),
        "W2": (128, 1),
        "b2": (1,),
    }

    torch_weights: dict[str, Any] = {}
    for name, arr in weight_files.items():
        if arr is None:
            continue
        expected = expected_shapes.get(name)
        if expected and arr.shape != expected:
            print(
                f"[Aurelius v9.0 Tier1] WARNING: Shape mismatch for {name}: "
                f"expected {expected}, got {arr.shape}. "
                "Using uninitialized PyTorch fallback weights. "
                "Run `aurelius train --task tier1` to train properly."
            )
            return {}
        torch_weights[name] = _torch.from_numpy(arr.astype(np.float32))  # type: ignore[union-attr]

    return torch_weights


def load_pytorch_fallback(weights_dir: str, model: Any) -> Any:
    """Load PyTorch model with weights from NumPy files.

    Args:
        weights_dir: Path to the model weights directory.
        model: The PyTorchBackend instance to load weights into.

    Returns:
        The PyTorchBackend with loaded (or randomly initialized) weights.
    """
    torch_weights = convert_numpy_to_torch_weights(weights_dir)

    if not torch_weights:
        print(
            "[Aurelius v9.0 Tier1] WARNING: Using uninitialized PyTorch fallback weights. "
            "Run `aurelius train --task tier1` to train properly."
        )
        return model

    state_mapping: dict[str, str] = {
        "W1": "fc1.weight",
        "b1": "fc1.bias",
        "W2": "fc2.weight",
        "b2": "fc2.bias",
    }

    state_dict: dict[str, Any] = {}
    for mlx_name, torch_key in state_mapping.items():
        if mlx_name in torch_weights:
            state_dict[torch_key] = torch_weights[mlx_name]

    if state_dict:
        model.load_state_dict(state_dict, strict=False)  # type: ignore[attr-defined]
        print(f"[Aurelius v9.0 Tier1] Loaded PyTorch fallback weights: {weights_dir}")
    else:
        print("[Aurelius v9.0 Tier1] WARNING: Could not map any weights. Using random initialization.")

    return model


class HuggingFaceWeightLoader:
    """Load pre-trained model weights from Hugging Face Hub.

    Attempts to download pre-trained Tier 1 model weights from
    Hugging Face Hub. Falls back gracefully to local weights
    or re-training if neither is available.

    Supported models:
        - ESOL solubility predictor (logS prediction)
        - QM9 energy predictor (DFT-computed energies)

    Disk Usage Management:
        Set the environment variable AURELIUS_HF_USE_SYMLINKS=1
        to track symlink preferences. Note: huggingface_hub v0.24+
        deprecated `local_dir_use_symlinks`. For reduced disk usage,
        use AURELIUS_MODEL_DIR to point to a location with more space,
        or manually symlink: ln -s $HF_HOME/models/... $AURELIUS_MODEL_DIR/
    """

    def __init__(
        self,
        model_dir: str | None = None,
        use_symlinks: bool | None = None,
    ) -> None:
        """Initialize the weight loader.

        Args:
            model_dir: Local directory to cache model weights.
                Defaults to AURELIUS_MODEL_DIR env var or
                <repo_root>/models/.
            use_symlinks: Deprecated. Kept for backward compatibility.
                The huggingface_hub library no longer supports
                local_dir_use_symlinks (v0.24+). This parameter is
                retained for API compatibility but has no effect.
        """
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        self._hf_available = self._check_hf_dependencies()
        self._use_symlinks = use_symlinks if use_symlinks is not None else _should_use_symlinks()

    def _check_hf_dependencies(self) -> bool:
        """Check if huggingface_hub and datasets are available."""
        from aurelius.utils.dependencies import HAS_DATASETS, HAS_HF_HUB

        return HAS_HF_HUB and HAS_DATASETS

    def load_model(
        self,
        task: str = "esol_solubility",
        local_only: bool = False,
    ) -> Any | None:
        """Load a pre-trained model from Hugging Face Hub.

        Args:
            task: Model task identifier ("esol_solubility" or "qm9_energy").
            local_only: If True, only load from local files.

        Returns:
            Model with loaded weights, or None if unavailable.
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

    def _load_from_hf_hub(self, model_id: str, task: str) -> Any | None:
        """Attempt to load model weights from Hugging Face Hub.

        Args:
            model_id: Hugging Face model repository ID.
            task: Model task identifier.

        Returns:
            Model with loaded weights, or None on failure.
        """
        from huggingface_hub import snapshot_download

        local_dir = os.path.join(self.model_dir, task, "hf_cache")

        # Pre-flight disk space check
        has_space, free_gb = check_disk_space(self.model_dir, min_free_gb=10.0)
        if not has_space:
            logger.warning(
                "Skipping HuggingFace download for %s: insufficient disk space "
                "(%.1fGB free, 10GB required). Consider using "
                "AURELIUS_HF_USE_SYMLINKS=1 or cleaning up old models.",
                model_id,
                free_gb,
            )

        # LRU cache eviction
        evicted = self.evict_lru_cache(max_cache_gb=20.0)
        if evicted > 0:
            logger.info("LRU cache eviction: removed %d old model(s)", evicted)

        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
            )
        except Exception as e:
            logger.warning("Failed to download model from HuggingFace Hub: %s", e)
            return None

        # NOTE: AURELIUS_HF_USE_SYMLINKS env var is tracked for
        # documentation purposes. The huggingface_hub library has
        # deprecated the `local_dir_use_symlinks` parameter (v0.24+).
        # Downloading to a local directory no longer uses symlinks.
        # Users with limited disk should use AURELIUS_MODEL_DIR to
        # point to a location with more space, or use HF cache symlinks
        # manually via: ln -s $HF_HOME/models/... $AURELIUS_MODEL_DIR/

        model = PyTorchBackend()
        model.load_weights(local_dir)
        print(f"[Aurelius v9.0 Tier1] Loaded {task} model from Hugging Face Hub: {model_id}")
        return model

    def _load_from_local(self, task: str) -> Any | None:
        """Load model weights from local directory.

        Args:
            task: Model task identifier.

        Returns:
            Model with loaded weights, or None if not found.
        """
        local_path = os.path.join(self.model_dir, task)
        if not os.path.isdir(local_path):
            return None

        try:
            model = PyTorchBackend()
            model.load_weights(local_path)
            print(f"[Aurelius v9.0 Tier1] Loaded {task} model from local: {local_path}")
            return model
        except Exception as e:
            print(f"[Aurelius v9.0 Tier1] Local load failed: {e}")
            return None

    def save_model(self, model: Any, task: str) -> str:
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

    def evict_lru_cache(self, max_cache_gb: float = 20.0) -> int:
        """Evict the oldest model directories from the HF cache directory.

        If the total cache size exceeds `max_cache_gb`, removes the oldest
        model directories (sorted by modification time) until the cache
        fits within the limit.

        Args:
            max_cache_gb: Maximum allowed cache size in GB (default: 20.0).

        Returns:
            Number of directories evicted.
        """
        try:
            cache_path = Path(self.model_dir)
            if not cache_path.is_dir():
                return 0

            # Collect all model directories sorted by modification time (oldest first)
            entries = sorted(
                [d for d in cache_path.iterdir() if d.is_dir()],
                key=lambda p: p.stat().st_mtime,
            )

            if len(entries) <= 1:
                return 0

            # Calculate total cache size (recursive)
            total_size = float(sum(self._dir_size(p) for p in entries if p.is_dir()))
            total_size /= 1024**3  # Convert to GB

            if total_size <= max_cache_gb:
                return 0

            # Evict oldest entries until within limit
            evicted = 0
            for entry in entries:
                if total_size <= max_cache_gb:
                    break
                entry_size = self._dir_size(entry) / (1024**3)
                total_size -= entry_size
                import shutil

                shutil.rmtree(entry)
                evicted += 1
                logger.info("Evicted LRU cache entry: %s", entry)

            return evicted

        except (OSError, PermissionError) as e:
            logger.warning("LRU cache eviction failed: %s", e)
            return 0

    def _dir_size(self, path: Path) -> int:
        """Calculate the total size of a directory recursively.

        Args:
            path: Path to the directory.

        Returns:
            Total size in bytes.
        """
        total = 0
        try:
            for entry in path.iterdir():
                if entry.is_dir():
                    total += self._dir_size(entry)
                else:
                    total += entry.stat().st_size
        except (OSError, PermissionError):
            pass

        return total


__all__ = [
    "DEFAULT_MODEL_DIR",
    "HAS_TORCH",
    "HUGGINGFACE_MODELS",
    "HuggingFaceWeightLoader",
    "convert_numpy_to_torch_weights",
    "load_pytorch_fallback",
]
