"""Project Aurelius - Bayesian Active Learning for Novel Molecule Discovery.

A streamlined pipeline that combines:
- RDKit-based structural filtering (electrolyte viability + LogP + MW gates)
- Hybrid property oracle (RF on ECFP4 + fragment-additivity for S/P/F)
- Bayesian active learning loop (RF surrogate, Expected Improvement acquisition)
- QM9 + fragment-additivity training data

Key design decisions:
- scikit-learn Random Forest for in-domain prediction (lightweight, no GPU)
- Fragment-additivity corrections for electrolyte-specific motifs (S, P, F)
- ECFP4 Morgan fingerprints (radius=2, 2048 bits) for molecular featurisation
- Gaussian-penalty objective function for battery-relevant scoring
- Threshold-based viability criteria (Aurelius Score >= 50 / 100)
- Single SMILES->Mol parsing per molecule per generation via MoleculeContext
"""

from __future__ import annotations

from importlib import metadata

__version__: str = metadata.version("aurelius")

from aurelius.agent.loop import DiscoveryLoop
from aurelius.types import ScreeningResult
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
from aurelius.types import MoleculeContext

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
