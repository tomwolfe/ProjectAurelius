"""Aurelius agent package.

Provides autonomous screening capabilities including:
- Mutation engine for chemical generation
- State management and checkpointing
- Reporting and analysis
"""

from aurelius.agent.mutation import MutationEngine
from aurelius.agent.reporting import (
    generate_chemical_insights,
    generate_discovery_results,
    generate_manifest,
    generate_screening_statistics,
    write_top_discoveries,
)

__all__ = [
    "MutationEngine",
    "generate_chemical_insights",
    "generate_discovery_results",
    "generate_manifest",
    "generate_screening_statistics",
    "write_top_discoveries",
]
