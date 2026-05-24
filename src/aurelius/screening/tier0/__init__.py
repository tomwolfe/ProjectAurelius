"""Tier 0: MPNN Activation Energy Predictor package.

Lightweight Message Passing Neural Network for activation energies
in battery electrolyte screening.
"""

from __future__ import annotations

from aurelius.screening.tier0.data import (
    _build_molecular_graph,
    generate_synthetic_training_data,
    train_tier0_model,
)
from aurelius.screening.tier0.models import (
    HAS_TORCH,
    ModelFactory,
    PyTorchBackend,
)
from aurelius.screening.tier0.predictor import (
    Tier0ActivationPredictor,
    _LinearFallbackPredictor,
)

__all__ = [
    "HAS_TORCH",
    "ModelFactory",
    "PyTorchBackend",
    "_LinearFallbackPredictor",
    "_build_molecular_graph",
    "generate_synthetic_training_data",
    "train_tier0_model",
]
