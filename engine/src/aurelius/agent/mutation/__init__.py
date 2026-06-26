"""Molecule mutation engine for battery electrolyte discovery.

Generates candidate molecules from seed SMILES using two strategies
in priority order:

1. **SMARTS functional-group replacement** — targeted electrolyte-relevant
   transformations (fluorination, methylation, ether/carbonate edits).
2. **BRICS fragmentation + reassembly** — scaffold hopping by breaking
   and reconnecting fragments at retrosynthetically sensible bonds.

Decomposed into:
  - ``MutationEngine`` — orchestrator (strategy selection, public API)
  - ``NoveltyValidator`` — novelty / trivial-extension / scaffold-gate checks
  - ``FragmentHarvester`` — dynamic BRICS fragment pool management
"""

from __future__ import annotations

from aurelius.agent.mutation.base import (
    BricsStrategy,
    MutationStrategy,
    SmartsStrategy,
    StrategyContext,
)
from aurelius.agent.mutation.brics import (
    _BRICS_LINKER_FRAGMENTS,
    _MAX_HARVESTED_FRAGMENTS,
)
from aurelius.agent.mutation.brics import (
    brics_building_block_coverage as _brics_building_block_coverage,
)
from aurelius.agent.mutation.brics import (
    combined_grounding_score as _combined_grounding_score,
)
from aurelius.agent.mutation.brics import (
    functional_group_coverage as _functional_group_coverage,
)
from aurelius.agent.mutation.brics import (
    get_brics_types as _get_brics_types,
)
from aurelius.agent.mutation.engine import MutationEngine
from aurelius.agent.mutation.harvester import FragmentHarvester
from aurelius.agent.mutation.novelty import NoveltyValidator
from aurelius.agent.mutation.smarts import (
    _ELECTROLYTE_CHECKS,
    ELECTROLYTE_FRAGMENT_POOL,
    ELECTROLYTE_SMARTS,
)
from aurelius.agent.mutation.smarts import (
    find_max_conjugated_path as _find_max_conjugated_path,
)
from aurelius.agent.mutation.smarts import (
    is_electrolyte_like as _is_electrolyte_like,
)

__all__ = [
    "MutationEngine",
    "MutationStrategy",
    "SmartsStrategy",
    "BricsStrategy",
    "StrategyContext",
    "NoveltyValidator",
    "FragmentHarvester",
    "ELECTROLYTE_SMARTS",
    "ELECTROLYTE_FRAGMENT_POOL",
    "_ELECTROLYTE_CHECKS",
    "_is_electrolyte_like",
    "_find_max_conjugated_path",
    "_BRICS_LINKER_FRAGMENTS",
    "_MAX_HARVESTED_FRAGMENTS",
    "_get_brics_types",
    "_brics_building_block_coverage",
    "_combined_grounding_score",
    "_functional_group_coverage",
]
