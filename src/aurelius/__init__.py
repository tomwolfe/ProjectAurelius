"""Project Aurelius v8.0 - The GNN-Enhanced Release.

Accelerated computational chemistry screening pipeline with real ML-based
property oracles, SELFIES-based mutation engine, and Gaussian Process
surrogate-driven active learning, optimized for Apple M-series Neural Accelerators.
"""

from __future__ import annotations

from importlib import metadata

__version__: str = metadata.version("aurelius")

from aurelius.config import (
    AureliusConfig,
    apply_global_config,
    get_config,
    initialize_environment,
)
from aurelius.memory.profiler import MemoryProfiler
from aurelius.pipeline import AureliusPipeline
from aurelius.scoring.oracle import Oracle, PretrainedGNNOracle
from aurelius.screening.tier0 import PyTorchBackend
from aurelius.screening.tier0.predictor import Tier0ActivationPredictor
from aurelius.screening.tier1 import MLXNAFilter
from aurelius.types import (
    AureliusScoreResult,
    DesolvationPathResult,
    MLXFilterResult,
    MoleculeInput,
    Tier2Result,
)

initialize_environment()

__all__ = [
    "__version__",
    "AureliusConfig",
    "AureliusPipeline",
    "AureliusScoreResult",
    "DesolvationPathResult",
    "MLXFilterResult",
    "MLXNAFilter",
    "MemoryProfiler",
    "MoleculeInput",
    "Oracle",
    "PretrainedGNNOracle",
    "Tier0ActivationPredictor",
    "PyTorchBackend",
    "apply_global_config",
    "get_config",
]
