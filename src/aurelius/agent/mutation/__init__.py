"""Molecule mutation engine for battery electrolyte discovery.

Generates candidate molecules from seed SMILES using two strategies
in priority order:

1. **SMARTS functional-group replacement** — targeted electrolyte-relevant
   transformations (fluorination, methylation, ether/carbonate edits).
2. **BRICS fragmentation + reassembly** — scaffold hopping by breaking
   and reconnecting fragments at retrosynthetically sensible bonds.
"""

from __future__ import annotations

from aurelius.agent.mutation.brics import (
    _BRICS_LINKER_FRAGMENTS,
    _MAX_HARVESTED_FRAGMENTS,
    brics_building_block_coverage as _brics_building_block_coverage,
    get_brics_types as _get_brics_types,
)
from aurelius.agent.mutation.engine import MutationEngine
from aurelius.agent.mutation.smarts import (
    _ELECTROLYTE_CHECKS,
    ELECTROLYTE_FRAGMENT_POOL,
    ELECTROLYTE_SMARTS,
    find_max_conjugated_path as _find_max_conjugated_path,
    is_electrolyte_like as _is_electrolyte_like,
)

__all__ = [
    "MutationEngine",
    "ELECTROLYTE_SMARTS",
    "ELECTROLYTE_FRAGMENT_POOL",
    "_ELECTROLYTE_CHECKS",
    "_is_electrolyte_like",
    "_find_max_conjugated_path",
    "_BRICS_LINKER_FRAGMENTS",
    "_MAX_HARVESTED_FRAGMENTS",
    "_get_brics_types",
    "_brics_building_block_coverage",
]
