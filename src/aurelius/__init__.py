"""Project Aurelius v9.0 - Bayesian Active Learning for Novel Molecule Discovery.

A streamlined pipeline that combines:
- MLX-based structural filtering
- RandomForest-based property oracles
- SELFIES-based mutation engine
- Gaussian Process surrogate-driven active learning
"""

from __future__ import annotations

from importlib import metadata

__version__: str = metadata.version("aurelius")

from aurelius.agent.loop import DiscoveryLoop, ScreeningResult
from aurelius.agent.mutation import MutationEngine
from aurelius.agent.reporting import (
    generate_chemical_insights,
    generate_discovery_results,
    generate_manifest,
    generate_screening_statistics,
    write_top_discoveries,
)
from aurelius.agent.state import (
    CheckpointManager,
    ConvergenceChecker,
    FeedbackAdapter,
)
from aurelius.config import (
    AureliusConfig,
    apply_global_config,
    get_config,
    initialize_environment,
)
from aurelius.pipeline import AureliusPipeline
from aurelius.scoring.oracle import Oracle, PropertyOracle
from aurelius.types import (
    MLXFilterResult,
    MoleculeInput,
    OracleResult,
)

initialize_environment()

__all__ = [
    "__version__",
    "AureliusConfig",
    "AureliusPipeline",
    "CheckpointManager",
    "ConvergenceChecker",
    "DiscoveryLoop",
    "FeedbackAdapter",
    "MLXFilterResult",
    "MLXNAFilter",
    "MoleculeInput",
    "MutationEngine",
    "Oracle",
    "PropertyOracle",
    "ScreeningResult",
    "generate_chemical_insights",
    "generate_discovery_results",
    "generate_manifest",
    "generate_screening_statistics",
    "get_config",
    "initialize_environment",
    "write_top_discoveries",
]
