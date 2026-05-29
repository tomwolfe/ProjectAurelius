"""Scoring and oracle package.

Provides:
- Oracle abstract base class
- PretrainedGNNOracle for real ML-based property evaluation
"""

from aurelius.scoring.oracle import Oracle, PretrainedGNNOracle

__all__ = [
    "Oracle",
    "PretrainedGNNOracle",
]
