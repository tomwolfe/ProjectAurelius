"""Central type definitions for Project Aurelius.

All @dataclass definitions used across the pipeline are centralized here
to eliminate circular imports between pipeline.py, scoring/engine.py,
and the screening tier modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MoleculeInput:
    """Input molecule specification for the Aurelius screening pipeline."""

    smiles: str
    solvent_type: str = "ec:dmc"
    salt_type: str = "NaPF6"
    ion_type: str = "Na+"
    temperature_k: float = 298.15
    voltage_cutoff: float = 0.05
    max_sei_time_ps: float = 1000.0
    n_scan_cycles: int = 500


@dataclass
class MLXFilterResult:
    """Result from the MLX-NA tier 1 screening filter."""

    molecule_smiles: str
    is_viable: bool
    confidence_score: float
    inference_time_ms: float
    na_utilization_pct: float
    quantization_format: str = "MX4"


@dataclass
class OracleResult:
    """Result from the PropertyOracle — predicted HOMO/LUMO properties.

    This is the clean, single source of truth for oracle evaluation
    in the active learning loop.
    """

    homo_eV: float
    lumo_eV: float
    lumo_gap_eV: float
    dipole_debye: float


__all__ = [
    "MLXFilterResult",
    "MoleculeInput",
    "OracleResult",
]
