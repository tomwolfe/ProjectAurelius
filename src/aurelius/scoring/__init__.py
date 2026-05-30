"""Scoring and oracle package.

Provides:
- Oracle abstract base class
- PropertyOracle for real ML-based property evaluation
"""

from aurelius.scoring.oracle import Oracle, PropertyOracle

__all__ = [
    "Oracle",
    "PropertyOracle",
]
