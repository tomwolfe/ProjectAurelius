"""Project Aurelius v7.0 - The GNN-Enhanced Release.

Accelerated computational chemistry screening pipeline with MPNN
activation energy prediction, cutoff-aware neighbor lists, and
HuggingFace integration, optimized for Apple M-series Neural Accelerators.
"""

from __future__ import annotations

from importlib import metadata

__version__: str = metadata.version("aurelius")

from aurelius import bridge
from aurelius.config import (
    AureliusConfig,
    apply_global_config,
    get_config,
    initialize_environment,
)
from aurelius.memory.manager import (
    MetalShaderConfig,
    QuantizationConfig,
    ZeroCopyMemoryManager,
)
from aurelius.memory.profiler import MemoryProfiler
from aurelius.pipeline import AureliusPipeline
from aurelius.scoring.engine import AureliusScoringEngine
from aurelius.screening.tier0 import ModelFactory, PyTorchBackend
from aurelius.screening.tier0.predictor import Tier0ActivationPredictor
from aurelius.screening.tier0.predictor import Tier0ActivationPredictor
from aurelius.screening.tier1 import MLXNAFilter
from aurelius.screening.tier2_mattersim import MatterSimMTSimulator
from aurelius.screening.tier3_gcmtwin import GCMDigitalTwin
from aurelius.types import (
    AureliusScoreResult,
    DesolvationPathResult,
    GCMDTConfig,
    GCMDTwinResult,
    MLXFilterResult,
    MoleculeInput,
    SEIEvolution,
    Tier2Result,
)

initialize_environment()

__all__ = [
    "__version__",
    "AureliusPipeline",
    "AureliusScoreResult",
    "DesolvationPathResult",
    "GCMDTConfig",
    "GCMDTwinResult",
    "GCMDigitalTwin",
    "MLXFilterResult",
    "AureliusConfig",
    "MemoryProfiler",
    "MoleculeInput",
    "MLXNAFilter",
    "MatterSimMTSimulator",
    "MetalShaderConfig",
    "QuantizationConfig",
    "SEIEvolution",
    "Tier0ActivationPredictor",
    "ModelFactory",
    "PyTorchBackend",
    "Tier0ActivationPredictor",
    "Tier2Result",
    "ZeroCopyMemoryManager",
    "apply_global_config",
    "bridge",
    "get_config",
    "AureliusScoringEngine",
]
