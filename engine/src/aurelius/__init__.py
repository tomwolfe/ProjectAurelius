"""Project Aurelius - Evolutionary Algorithm for Molecule Discovery.

A physically-grounded pipeline that combines:
- RDKit-based structural filtering (electrolyte viability + LogP + MW gates)
- Hybrid property oracle: Quantum (xTB/TOM) for HOMO/LUMO + GC for bulk props
- Evolutionary Algorithm loop (Tournament Selection, Tanimoto Diversity Penalty)
- SMARTS/BRICS mutation engine with anti-gaming topology constraints

Key design decisions:
- **Hybrid physics**: Frontier orbitals via quantum chemistry (xTB/TOM);
  bulk properties (dielectric, viscosity, Li+ solvation) via GC fragment-additivity
- Direct Oracle Evaluation (the oracle is fast enough without approximation)
- ECFP4 Morgan fingerprints (radius=2, 2048 bits) for molecular featurisation
- Gaussian-penalty objective function for battery-relevant scoring
- Threshold-based viability criteria (Aurelius Score >= 50 / 100)
- Single SMILES->Mol parsing per molecule per generation via MoleculeContext
"""

from __future__ import annotations

from importlib import metadata

try:
    __version__: str = metadata.version("aurelius-engine")
except metadata.PackageNotFoundError:
    __version__: str = metadata.version("aurelius")

from aurelius.agent.loop import DiscoveryLoop
from aurelius.agent.mutation import MutationEngine
from aurelius.agent.reporting import (
    generate_discoveries_sdf,
    generate_run_summary,
)
from aurelius.agent.state import LoopState
from aurelius.filter import Filter, is_structurally_viable
from aurelius.kernel_loader import JSONKernelLoader, KernelLoader, _load_demo_kernel
from aurelius.pipeline import AureliusPipeline
from aurelius.scorer import _OBJECTIVES, Objective, compute_score, format_score
from aurelius.scoring.oracle import PropertyOracle
from aurelius.types import MoleculeContext, ScreeningResult

__all__ = [
    "__version__",
    "AureliusPipeline",
    "DiscoveryLoop",
    "Filter",
    "is_structurally_viable",
    "JSONKernelLoader",
    "KernelLoader",
    "LoopState",
    "MoleculeContext",
    "MutationEngine",
    "Objective",
    "PropertyOracle",
    "ScreeningResult",
    "_OBJECTIVES",
    "_load_demo_kernel",
    "compute_score",
    "format_score",
    "generate_discoveries_sdf",
    "generate_run_summary",
]
