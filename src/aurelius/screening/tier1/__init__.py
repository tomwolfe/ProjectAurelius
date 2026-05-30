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
)
from aurelius.screening.tier1.loaders import (
    DEFAULT_MODEL_DIR,
    HUGGINGFACE_MODELS,
    HuggingFaceWeightLoader,
    convert_numpy_to_torch_weights,
    load_pytorch_fallback,
)
from aurelius.screening.tier1.models import (
    PyTorchBackend,
)
from aurelius.screening.tier1.training import (
    train_on_esol,
    train_on_qm9,
)
from aurelius.utils.chem_utils import generate_ecfp4_fingerprint

__all__ = [
    "DEFAULT_MODEL_DIR",
    "HAS_MLX",
    "HAS_RDKIT",
    "HAS_TORCH",
    "HUGGINGFACE_MODELS",
    "HuggingFaceWeightLoader",
    "MLXNAFilter",
    "PyTorchBackend",
    "generate_ecfp4_fingerprint",
    "_generate_ecfp4_fingerprint",
    "convert_numpy_to_torch_weights",
    "load_pytorch_fallback",
    "train_on_esol",
    "train_on_qm9",
]
