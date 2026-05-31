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

SCORE_WEIGHT_LUMO: float = 0.30
"""Weight for LUMO SEI-formation reward."""

SCORE_WEIGHT_HOMO: float = 0.20
"""Weight for HOMO oxidative-stability penalty."""

SCORE_WEIGHT_DIELECTRIC: float = 0.25
"""Weight for dielectric-constant (salt dissolution) reward."""

SCORE_WEIGHT_VISCOSITY: float = 0.15
"""Weight for viscosity (ion mobility) penalty."""

SCORE_WEIGHT_SA: float = 0.10
"""Weight for synthetic accessibility penalty."""

# ---------------------------------------------------------------------------
# Viability Threshold
# ---------------------------------------------------------------------------
# The total score ranges from 0 to 100. Molecules with total_score >= 50
# are considered viable candidates for further investigation.

VIABILITY_THRESHOLD: float = 50.0
"""Minimum total score for a molecule to be considered a viable discovery."""

DISCOVERY_THRESHOLD: float = 65.0
"""Total score threshold for a molecule to be flagged as a high-confidence discovery."""
