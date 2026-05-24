"""Phase 3: Accelerated Screening Pipeline package."""

from __future__ import annotations

from aurelius.screening.tier0 import (
    HAS_TORCH,
    ModelFactory,
    PyTorchBackend,
    Tier0ActivationPredictor,
)
from aurelius.screening.tier1 import (
    HAS_MLX,
    HAS_RDKIT,
    MLXNAFilter,
)
from aurelius.screening.tier1 import (
    HAS_TORCH as HAS_TORCH_TIER1,
)
from aurelius.screening.tier2_mattersim import MatterSimMTSimulator
from aurelius.screening.tier3_gcmtwin import GCMDigitalTwin
from aurelius.types import (
    GCMDTConfig,
    GCMDTwinResult,
    MLXFilterResult,
    Tier2Result,
)

__all__ = [
    "GCMDTConfig",
    "GCMDTwinResult",
    "GCMDigitalTwin",
    "HAS_MLX",
    "HAS_RDKIT",
    "HAS_TORCH",
    "HAS_TORCH_TIER1",
    "MLXFilterResult",
    "MLXNAFilter",
    "ModelFactory",
    "PyTorchBackend",
    "MatterSimMTSimulator",
    "Tier0ActivationPredictor",
    "Tier2Result",
]
