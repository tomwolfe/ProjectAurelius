"""Project Aurelius v9.0 - Bayesian Active Learning for Novel Molecule Discovery.

A streamlined pipeline that combines:
- RDKit-based structural filtering (SA score + Lipinski)
- QSPR property oracles (Random Forest on ECFP4 fingerprints)
- SELFIES-based mutation engine
- Random Forest surrogate-driven active learning
"""

from __future__ import annotations

from importlib import metadata

__version__: str = metadata.version("aurelius")

from aurelius.agent.loop import DiscoveryLoop, ScreeningResult
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
from aurelius.config import (
    AureliusConfig,
    get_config,
)
from aurelius.pipeline import AureliusPipeline
from aurelius.scoring.oracle import PropertyOracle
from aurelius.types import MoleculeInput

__all__ = [
    "__version__",
    "AureliusConfig",
    "AureliusPipeline",
    "CheckpointManager",
    "ConvergenceChecker",
    "DiscoveryLoop",
    "FeedbackAdapter",
    "MoleculeInput",
    "MutationEngine",
    "PropertyOracle",
    "ScreeningResult",
    "generate_discoveries_sdf",
    "generate_run_summary",
    "get_config",
]
