"""Property packs for modular group-contribution models.

Each pack defines fragment patterns and prediction methods for a
specific chemical domain. The default ``ElectrolytePack`` is defined
in ``aurelius.scoring.oracle.gc``.
"""

from __future__ import annotations

from aurelius.scoring.oracle.packs.organic_electronics import OrganicElectronicsPack

__all__ = [
    "OrganicElectronicsPack",
]
