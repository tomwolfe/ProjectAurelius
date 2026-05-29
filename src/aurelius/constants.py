"""Physical constants used across the Aurelius pipeline.

Centralizes all physics constants to provide a single source of truth
and avoid magic numbers scattered across the codebase.

Constants:
    COULOMB_EV_A: Coulomb conversion factor (eV*A), used in electrostatic
        energy calculations. Equivalent to k_e / (4 * pi * epsilon_0)
        in SI units, converted to eV*Angstrom.
    BOLTZMANN_EV_K: Boltzmann constant in eV/K, used in Arrhenius
        rate constant calculations for kMC simulations.
    BOLTZMANN_J_K: Boltzmann constant in J/K (SI units).
    AVOGADRO: Avogadro's number, used for molar energy conversions.
    FINGERPRINT_SIZE: Default ECFP4 fingerprint bit size.
    MAX_ATOMIC_NUMBER: Maximum atomic number in the periodic table.
    DEFAULT_LJ_CUTOFF: Default Lennard-Jones cutoff distance in Angstroms.
"""

from __future__ import annotations

# Coulomb conversion factor: eV * Angstrom
# k_e / (4 * pi * epsilon_0) in eV*Angstrom units
COULOMB_EV_A: float = 14.3996

# Boltzmann constant
BOLTZMANN_EV_K: float = 8.617333262e-5  # eV/K
BOLTZMANN_J_K: float = 1.380649e-23  # J/K

# Avogadro's number
AVOGADRO: float = 6.02214076e23

# ML / chemistry constants
FINGERPRINT_SIZE: int = 2048
MAX_ATOMIC_NUMBER: int = 119
DEFAULT_LJ_CUTOFF: float = 10.0
