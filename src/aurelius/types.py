"""Central type definitions for Project Aurelius.

All dataclass definitions used across the pipeline are centralized here
to eliminate circular imports between modules.
"""

from __future__ import annotations

from dataclasses import dataclass


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
class OracleResult:
    """Result from the PropertyOracle — predicted HOMO/LUMO properties."""

    homo_eV: float
    lumo_eV: float
    lumo_gap_eV: float
    dipole_debye: float


__all__ = [
    "MoleculeInput",
    "OracleResult",
]
