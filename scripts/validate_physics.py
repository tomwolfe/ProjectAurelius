#!/usr/bin/env python3
"""Physics validation script for Project Aurelius.

Runs small simulations and compares results against known physical
behavior to verify the physics engine is calibrated correctly.

Usage:
    python scripts/validate_physics.py
    python scripts/validate_physics.py --strict
    python scripts/validate_physics.py --tier 2
    python scripts/validate_physics.py --tier 3

This script validates:
    - Tier 2: Energy conservation, force gradients, finite energies
    - Tier 3: Arrhenius temperature dependence, concentration dependence
    - Solvation engine: Born charge interpolation, dielectric lookup
"""

from __future__ import annotations

from aurelius.cli_scripts.validate_physics import main as _main

if __name__ == "__main__":
    _main()
