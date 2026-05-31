"""Aurelius agent package.

Provides autonomous screening capabilities including:
- SELFIES-based mutation engine for chemical generation
- State management and checkpointing
- Reporting and analysis
"""

from aurelius.agent.mutation import MutationEngine
from aurelius.agent.reporting import (
    generate_discoveries_sdf,
    generate_run_summary,
)

__all__ = [
    "MutationEngine",
    "generate_discoveries_sdf",
    "generate_run_summary",
]
