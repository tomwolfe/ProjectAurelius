"""Tier 1: MLX-NA Filter package.

Accelerated molecular viability screening using pre-trained models
loaded from Hugging Face Hub or locally trained on real datasets.
"""

from __future__ import annotations

from aurelius.screening.tier1.filter import (
    HAS_MLX,
    HAS_RDKIT,
    HAS_TORCH,
    MLXNAFilter,
    _generate_ecfp4_fingerprint,
    _hash_fallback,
)
from aurelius.screening.tier1.loaders import (
    DEFAULT_MODEL_DIR,
    HUGGINGFACE_MODELS,
    HuggingFaceWeightLoader,
    convert_mlx_to_torch_weights,
    load_pytorch_fallback_with_mlx_weights,
)
from aurelius.screening.tier1.models import (
    PyTorchFallbackFilter,
    _ChemVLM2MLP,
    _FallbackMLP,
)
from aurelius.screening.tier1.training import (
    train_on_esol,
    train_on_qm9,
)

__all__ = [
    "DEFAULT_MODEL_DIR",
    "HAS_MLX",
    "HAS_RDKIT",
    "HAS_TORCH",
    "HUGGINGFACE_MODELS",
    "HuggingFaceWeightLoader",
    "MLXNAFilter",
    "PyTorchFallbackFilter",
    "_ChemVLM2MLP",
    "_FallbackMLP",
    "_generate_ecfp4_fingerprint",
    "_hash_fallback",
    "convert_mlx_to_torch_weights",
    "load_pytorch_fallback_with_mlx_weights",
    "train_on_esol",
    "train_on_qm9",
]
