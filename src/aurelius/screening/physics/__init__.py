"""Physics module for MatterSim-MT.

This package contains the vectorized physics engine for:
1. Lennard-Jones + Coulombic potentials
2. Grid-based neighbor list construction
3. Path integral desolvation simulation
"""

from aurelius.screening.physics.simulator import MatterSimMTSimulator

__all__ = ["MatterSimMTSimulator"]

