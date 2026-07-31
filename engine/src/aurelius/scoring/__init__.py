"""Scoring and oracle package.

Provides the ``PropertyOracle`` class for QSPR-based HOMO/LUMO prediction,
multi-objective NSGA-II optimization, and retrosynthetic pathway verification.
"""

from aurelius.scoring.multi_objective import (
    extract_pareto_front,
    nsga_ii_select,
)
from aurelius.scoring.oracle import PropertyOracle

__all__ = [
    "PropertyOracle",
    "extract_pareto_front",
    "nsga_ii_select",
]
