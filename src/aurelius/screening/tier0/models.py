"""Tier 0: MPNN (Message Passing Neural Network) model definitions.

Implements a lightweight graph neural network for activation energy
prediction in battery electrolyte screening.

Architecture:
    - Node features: atomic number, degree, formal charge, aromaticity
    - Message passing: 2-layer edge-based MP with scatter_add aggregation
    - Readout: MLP over pooled node embeddings
    - Output: 4 activation energies (eV) for EC/DMC reduction, PF6 decomposition, polymerization

References:
    Gilmer, J. et al. "Neural Message Passing for Quantum Chemistry." ICML 2017.
    Wu, Z. et al. "Molecular Graph Convolutions: Moving Beyond Fingerprints." JMLR 2021.
"""

from __future__ import annotations

from aurelius.screening.tier0.backend_torch import (
    PyTorchBackend,
    PyTorchBackendUnavailableError,
    _model_factory,
)
from aurelius.utils.dependencies import HAS_TORCH

__all__ = ["HAS_TORCH", "PyTorchBackend", "PyTorchBackendUnavailableError", "_model_factory"]
