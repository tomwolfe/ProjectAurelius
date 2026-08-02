"""Aurelius dashboard package.

Provides a Streamlit-based visualization dashboard for:
- Discovery trajectory (score vs generation, scaffold novelty vs time)
- Chemical space exploration (UMAP/t-SNE of discovered molecules)
- Pareto front visualization (interactive 3D plot)
- Molecule viewer with property annotations
"""

from __future__ import annotations

__all__ = ["app"]
