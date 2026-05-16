"""Project Aurelius v5.2 - The 2nm Fusion Edition.

Accelerated computational chemistry screening pipeline optimized
for Apple M-series Neural Accelerators.
"""

from aurelius import bridge
from aurelius.config import M5ProConfig, apply_global_config, get_config
from aurelius.memory.manager import (
    MetalShaderConfig,
    QuantizationConfig,
    ZeroCopyMemoryManager,
)
from aurelius.pipeline import AureliusPipeline
from aurelius.scoring.engine import AureliusScoringEngine
from aurelius.screening.tier1_mlx_filter import MLXNAFilter
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

__all__ = [
    "AureliusPipeline",
    "AureliusScoreResult",
    "DesolvationPathResult",
    "GCMDTConfig",
    "GCMDTwinResult",
    "GCMDigitalTwin",
    "MLXFilterResult",
    "M5ProConfig",
    "MoleculeInput",
    "MLXNAFilter",
    "MatterSimMTSimulator",
    "MetalShaderConfig",
    "QuantizationConfig",
    "SEIEvolution",
    "Tier2Result",
    "ZeroCopyMemoryManager",
    "apply_global_config",
    "bridge",
    "get_config",
]
