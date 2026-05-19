"""Central type definitions for Project Aurelius v5.2.

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
class AureliusScoreResult:
    """Complete Aurelius v5.2 score breakdown."""

    molecule_smiles: str
    total_score: float = 0.0  # S_A_v5.2 (0-100 scale)

    # Component scores (0-100 each)
    sigma_score: float = 0.0
    desolvation_score: float = 0.0
    sei_homogeneity_score: float = 0.0
    mx_synthesis_score: float = 0.0
    gwp_penalty: float = 0.0

    # Weights used
    weight_sigma: float = 0.3
    weight_desolvation: float = 0.2
    weight_sei: float = 0.2
    weight_mx: float = 0.2
    weight_gwp: float = 0.1

    # Pass/fail
    is_viable: bool = False
    rejection_reasons: list[str] = field(default_factory=list)
    viability_threshold: float = 0.0

    # Metadata
    tier1_viable: bool = False
    tier2_viable: bool = False
    tier3_viable: bool = False


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
class DesolvationPathResult:
    """Result from the MatterSim-MT desolvation path integral calculation.

    Represents a static Potential Energy Surface (PES) scan where an
    ion is displaced through a frozen solvent field along a reaction
    coordinate. The scan produces an energy profile used to identify
    desolvation barriers.
    """

    molecule_smiles: str
    barrier_height_eV: float
    local_maxima_eV: float
    path_integral_eV_A: float
    rejected: bool
    rejection_reason: str | None = None
    n_scan_points: int = 500


@dataclass
class Tier2Result:
    """Result from the MatterSim-MT Tier 2 simulation."""

    molecule_smiles: str
    is_viable: bool
    desolvation_path: DesolvationPathResult
    simulation_time_ms: float
    memory_used_gb: float


@dataclass
class SEIEvolution:
    """SEI (Solid Electrolyte Interphase) evolution state."""

    time_ps: float
    thickness_angstrom: float
    homogeneity_score: float  # 0-1, higher = more homogeneous
    ionic_conductivity_s_cm: float
    electronic_insulation: bool
    components: list[str] = field(default_factory=list)


@dataclass
class GCMDTwinResult:
    """Result from the GCMD Digital Twin simulation."""

    molecule_smiles: str
    sei_evolution: SEIEvolution
    interface_stability: float  # 0-1
    memory_used_gb: float
    context_tokens_used: int
    simulation_time_ms: float


@dataclass
class GCMDTConfig:
    """Configuration for GCMD Digital Twin kMC simulation.

    Controls the kinetic Monte Carlo simulation parameters
    for SEI (Solid Electrolyte Interphase) evolution modeling.
    """

    max_simulation_steps: int = 5000
    record_interval: int = 50  # Record thickness every N steps
    transport_limit_thickness_angstrom: float = 15.0  # Beyond this, diffusion limits rates
    seed_from_inputs: bool = True  # Deterministic seeding from molecular inputs
    use_mass_transport_limitation: bool = True  # As SEI grows, solvent concentration at interface drops


__all__ = [
    "AureliusScoreResult",
    "DesolvationPathResult",
    "GCMDTConfig",
    "GCMDTwinResult",
    "MLXFilterResult",
    "MoleculeInput",
    "SEIEvolution",
    "Tier2Result",
]
