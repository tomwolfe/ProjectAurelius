"""Re-export the Tier 1 Filter class and structural pre-check for convenient access.

Usage:
    from aurelius.filter import Filter, is_structurally_viable
"""

from aurelius.screening.structural import is_structurally_viable
from aurelius.screening.tier1.filter import Filter

__all__ = ["Filter", "is_structurally_viable"]
