"""Tier 1: Deterministic structural-viability filter.

Provides a fast, interpretable screening gate using RDKit's
Synthetic Accessibility score and Lipinski Rule-of-5.
"""

from __future__ import annotations

from aurelius.screening.tier1.filter import Filter

__all__ = [
    "Filter",
]
