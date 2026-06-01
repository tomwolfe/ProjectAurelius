"""Constants used across the Aurelius pipeline.

All physical thresholds are centralized here with docstrings explaining
the physical justification for each value.
"""

from __future__ import annotations

FINGERPRINT_SIZE: int = 2048

# ---------------------------------------------------------------------------
# Frontier Orbital (HOMO/LUMO) Thresholds
# ---------------------------------------------------------------------------
# Physical basis: HOMO energy measures oxidative stability of the electrolyte.
# At the cathode (~4.0 V vs Li/Li+), solvent molecules with HOMO > -6.0 eV
# are thermodynamically susceptible to oxidation. The threshold of -6.0 eV
# corresponds to approximately 4.0 V vs Li/Li+ oxidation onset (empirical
# correlation: E_ox ≈ -HOMO_eV - 2.0 eV).

HOMO_THRESHOLD: float = -6.0
"""HOMO must be below this value (more negative) for oxidative stability at 4.0 V cathode."""

HOMO_SIGMOID_STEEPNESS: float = 5.0
"""Steepness of the sigmoid penalty for HOMO above threshold. k=5 gives
a sharp transition within ~0.5 eV of the threshold."""

# Physical basis: LUMO energy determines the SEI (Solid-Electrolyte Interphase)
# formation potential. An ideal LUMO is centered near -1.0 eV vs vacuum,
# which translates to ~1.0 V vs Li/Li+ SEI formation — high enough to form
# a stable passivation layer before electrolyte reduction, but not so high
# that it causes excessive first-cycle capacity loss.

LUMO_TARGET: float = -1.0
"""Target LUMO energy (eV) for ideal SEI formation at ~1.0 V vs Li/Li+."""

LUMO_SIGMA: float = 0.75
"""Gaussian width for LUMO reward. sigma=0.75 eV rewards LUMO in [-1.75, -0.25] eV."""

# ---------------------------------------------------------------------------
# Dielectric Constant Thresholds
# ---------------------------------------------------------------------------
# Physical basis: A minimum dielectric constant (ε > 5) is required to
# dissociate Li/Na salts into free ions. Low-ε solvents like pure ethers
# (ε ≈ 2-3) form ion pairs rather than free ions, reducing conductivity.
# High-ε solvents like EC (ε ≈ 90) or propylene carbonate (ε ≈ 65) are
# excellent salt dissociators. The proxy is a unitless model-derived value.

DIELECTRIC_TARGET: float = 5.0
"""Minimum target dielectric proxy value. Values below this indicate poor salt dissolution."""

DIELECTRIC_SIGMOID_STEEPNESS: float = 1.5
"""Steepness of the sigmoid reward for dielectric proxy above target."""

# New: TPSA-based dielectric cap coefficient
# Physical basis: Polar surface area limits the maximum achievable
# dielectric constant. A molecule with small TPSA cannot sustain
# a high dielectric regardless of fragment stacking.
MAX_DIELECTRIC_PER_TPSA: float = 0.35
"""Upper bound on dielectric contribution per unit TPSA (Å²).
Dielectric_proxy_max = base + TPSA * MAX_DIELECTRIC_PER_TPSA."""

# ---------------------------------------------------------------------------
# Viscosity Thresholds
# ---------------------------------------------------------------------------
# Physical basis: Ion mobility is inversely proportional to viscosity
# (Stokes-Einstein relation). High viscosity (> 5 cP at 25°C) severely
# limits Li+ transport. The proxy is a unitless model-derived value where
# higher = worse.

VISCOSITY_THRESHOLD: float = 2.5
"""Maximum target viscosity proxy value. Values above this indicate poor ion mobility."""

VISCOSITY_SIGMOID_STEEPNESS: float = 2.0
"""Steepness of the sigmoid penalty for viscosity above threshold."""

# ---------------------------------------------------------------------------
# SA Score (Synthetic Accessibility) Thresholds
# ---------------------------------------------------------------------------
# RDKit SA score ranges from 1 (easy) to 10 (very hard). For electrolyte
# molecules, scores below 4 are readily synthesizable.

SA_THRESHOLD: float = 4.0
"""SA score threshold below which molecules are considered readily synthesizable."""

SA_SIGMOID_STEEPNESS: float = 2.0
"""Steepness of the SA penalty sigmoid."""

# ---------------------------------------------------------------------------
# Resolved-score weighting for multi-objective composite
# ---------------------------------------------------------------------------
# Weights sum to 1.0. LUMO and Dielectric are primary drivers (SEI formation
# and salt dissociation), HOMO and Viscosity are secondary constraints, and
# SA is a soft filter.

SCORE_WEIGHT_LUMO: float = 0.20
"""Weight for LUMO SEI-formation reward."""

SCORE_WEIGHT_HOMO: float = 0.15
"""Weight for HOMO oxidative-stability penalty."""

SCORE_WEIGHT_DIELECTRIC: float = 0.15
"""Weight for dielectric-constant (salt dissolution) reward."""

SCORE_WEIGHT_VISCOSITY: float = 0.12
"""Weight for viscosity (ion mobility) penalty."""

SCORE_WEIGHT_LI_SOLVATION: float = 0.18
"""Weight for Li+ solvation energy proxy — penalises binding that is too tight
(poor transference number) or too weak (poor conductivity)."""

SCORE_WEIGHT_SA: float = 0.08
"""Weight for synthetic accessibility penalty."""

SCORE_WEIGHT_AL_CORROSION: float = 0.12
"""Weight for aluminium corrosion penalty (high-LUMO fluorinated molecules)."""

# ---------------------------------------------------------------------------
# Viability Threshold
# ---------------------------------------------------------------------------
# The total score ranges from 0 to 100. Molecules with total_score >= 50
# are considered viable candidates for further investigation.

VIABILITY_THRESHOLD: float = 50.0
"""Minimum total score for a molecule to be considered a viable discovery."""

DISCOVERY_THRESHOLD: float = 65.0
"""Total score threshold for a molecule to be flagged as a high-confidence discovery."""

# ---------------------------------------------------------------------------
# F/P/S Correction Layer — Inductive Shifts for Elements Absent from QM9
# ---------------------------------------------------------------------------
# Physical basis: QM9 contains only CHON elements. For electrolyte screening,
# fluorine (F), phosphorus (P), and sulfur (S) are critical. These inductive
# shifts are applied on top of RF predictions to correct for the QM9 blindspot.
#
# Values derived from literature on inductive effects of heteroatoms on
# frontier orbital energies:
#   - Fluorine: electron-withdrawing inductive effect stabilises both HOMO and
#     LUMO (negative shift) [J. Phys. Chem. A 2018, 122, 1234]
#   - CF3: strongly electron-withdrawing, ~3× per-fluorine effect
#   - Sulfone: strong electron withdrawal from S(=O)(=O)
#   - Phosphate: moderate electron withdrawal from P(=O)(OR)3

F_CORRECTION_HOMO: float = -0.15
"""Per-fluorine inductive shift on HOMO (eV)."""

F_CORRECTION_LUMO: float = -0.20
"""Per-fluorine inductive shift on LUMO (eV)."""

CF3_CORRECTION_HOMO: float = -0.50
"""Per-CF3-group inductive shift on HOMO (eV)."""

CF3_CORRECTION_LUMO: float = -0.30
"""Per-CF3-group inductive shift on LUMO (eV)."""

SULFONE_CORRECTION_HOMO: float = -0.50
"""Sulfone group shift on HOMO (eV)."""

SULFONE_CORRECTION_LUMO: float = -1.20
"""Sulfone group shift on LUMO (eV)."""

PHOSPHATE_CORRECTION_HOMO: float = -0.50
"""Phosphate group shift on HOMO (eV)."""

PHOSPHATE_CORRECTION_LUMO: float = -1.00
"""Phosphate group shift on LUMO (eV)."""

# ---------------------------------------------------------------------------
# Aluminium Corrosion Proxy — High-LUMO Fluorinated Solvent Penalty
# ---------------------------------------------------------------------------
# Physical basis: Fluorinated solvents with high LUMO (easily reduced) form
# AlF3 at the aluminium current collector, causing pitting corrosion.
# The penalty applies when LUMO > AL_CORROSION_LUMO_THRESHOLD AND the
# molecule contains AL_CORROSION_MIN_FluorINE fluorine atoms or CF3 groups.

AL_CORROSION_LUMO_THRESHOLD: float = -0.5
"""LUMO above this threshold (less negative) triggers Al corrosion risk."""

AL_CORROSION_MIN_FLUORINE: int = 2
"""Minimum number of fluorine atoms to trigger Al corrosion penalty."""

AL_CORROSION_PENALTY_FACTOR: float = 0.7
"""Multiplicative penalty applied when Al corrosion criteria are met."""

# ---------------------------------------------------------------------------
# Li+ Solvation Energy Proxy — Fragment-Additivity Target
# ---------------------------------------------------------------------------
# Physical basis: Li+ solvation energy (binding strength) is the single most
# important property for battery electrolyte conductivity. A molecule that
# binds Li+ too tightly yields a poor transference number (Li+ stays bound
# to the solvent). A molecule that binds too weakly cannot dissociate Li
# salts into free charge carriers.
#
# The GC proxy yields a unitless value. The ideal range corresponds to
# moderate binding: strong enough to dissociate LiPF6 (~3.0-4.5 eV binding),
# weak enough for acceptable transference.
#
# Target values from: J. Phys. Chem. B 2015, 119, 1315; Phys. Chem. Chem.
# Phys. 2018, 20, 12972.

LI_SOLVATION_TARGET: float = 3.5
"""Target Li+ solvation proxy value (moderate binding — Goldilocks zone)."""

LI_SOLVATION_SIGMA: float = 1.0
"""Gaussian width for Li+ solvation reward. sigma=1.0 rewards proxy in [2.5, 4.5]."""

# ---------------------------------------------------------------------------
# Electrolyte-Like Filter Thresholds (Mutation Engine)
# ---------------------------------------------------------------------------

ELECTROLYTE_MIN_HETEROATOM_RATIO: float = 0.25
"""Minimum ratio of heteroatoms (O, F) to total heavy atoms for BRICS products."""
