"""Tier 0: Lightweight Message Passing Neural Network (MPNN) for Activation Energy Prediction.

Backward-compatible re-export module. All implementation code has been
refactored into submodules under screening/tier0/:

    - models.py   -> Tier0MPNN, MPNNEdgeBlock, MPNNReadoutMLP
    - data.py     -> _build_molecular_graph, generate_synthetic_training_data, train_tier0_model
    - predictor.py -> Tier0ActivationPredictor, _LinearFallbackPredictor

This file preserves the original public API for existing imports.

Architecture:
    - Node features: atomic number, degree, formal charge, aromaticity
    - Message passing: 2-layer edge-based MP with torch.scatter_add aggregation
    - Readout: MLP over pooled node embeddings
    - Output: 4 activation energies (eV) for EC/DMC reduction, PF6 decomposition, polymerization

References:
    Gilmer, J. et al. "Neural Message Passing for Quantum Chemistry." ICML 2017.
    Wu, Z. et al. "Molecular Graph Convolutions: Moving Beyond Fingerprints." JMLR 2021.
"""

from __future__ import annotations

from aurelius.screening.tier0.data import (
    _build_molecular_graph,
    generate_synthetic_training_data,
    train_tier0_model,
)
from aurelius.screening.tier0.models import (
    HAS_TORCH,
    MPNNEdgeBlock,
    MPNNReadoutMLP,
    Tier0MPNN,
)
from aurelius.screening.tier0.predictor import (
    Tier0ActivationPredictor,
    _LinearFallbackPredictor,
)

__all__ = [
    "HAS_TORCH",
    "MPNNEdgeBlock",
    "MPNNReadoutMLP",
    "Tier0ActivationPredictor",
    "Tier0MPNN",
    "_LinearFallbackPredictor",
    "_build_molecular_graph",
    "generate_synthetic_training_data",
    "train_tier0_model",
]
