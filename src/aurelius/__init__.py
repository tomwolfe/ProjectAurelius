"""Project Aurelius - Hybrid Quantum/ML Active Learning for Molecule Discovery.

A physically-grounded pipeline that combines:
- RDKit-based structural filtering (electrolyte viability + LogP + MW gates)
- Hybrid property oracle: Quantum (xTB/TOM) for HOMO/LUMO + GC for bulk props
- Bayesian active learning loop (RF surrogate, NWEI acquisition)
- SMARTS/BRICS mutation engine with anti-gaming topology constraints

Key design decisions:
- **Hybrid physics**: Frontier orbitals via quantum chemistry (xTB/TOM);
  bulk properties (dielectric, viscosity, Li+ solvation) via GC fragment-additivity
- scikit-learn Random Forest for the surrogate (lightweight, no GPU)
- ECFP4 Morgan fingerprints (radius=2, 2048 bits) for molecular featurisation
- Gaussian-penalty objective function for battery-relevant scoring
- Threshold-based viability criteria (Aurelius Score >= 50 / 100)
- Single SMILES->Mol parsing per molecule per generation via MoleculeContext
"""

from __future__ import annotations

from importlib import metadata

__version__: str = metadata.version("aurelius")

from aurelius.agent.loop import DiscoveryLoop
from aurelius.agent.mutation import MutationEngine
from aurelius.agent.reporting import (
    generate_discoveries_sdf,
    generate_run_summary,
)
from aurelius.agent.state import (
    CheckpointManager,
    ConvergenceChecker,
    FeedbackAdapter,
)
from aurelius.pipeline import AureliusPipeline
from aurelius.scoring.oracle import PropertyOracle
from aurelius.types import MoleculeContext, ScreeningResult

__all__ = [
    "__version__",
    "AureliusPipeline",
    "CheckpointManager",
    "ConvergenceChecker",
    "DiscoveryLoop",
    "FeedbackAdapter",
    "MoleculeContext",
    "MutationEngine",
    "PropertyOracle",
    "ScreeningResult",
    "generate_discoveries_sdf",
    "generate_run_summary",
]
