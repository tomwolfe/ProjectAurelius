"""GC Uncertainty Quantification — canonical re-export module.

Single source of truth: implementation lives in ``gc.py`` under the
``GcUqEnsemble`` class. This module re-exports the ensemble and its
supporting constants so that consumers can import from the canonical path:

    from aurelius.scoring.oracle.gc_uq import GcUqEnsemble

Anti-failure rationale (UQ penalty):
    When the ensemble's prediction variance exceeds 15% of the mean
    prediction magnitude for either dielectric or viscosity, the molecule
    is flagged as out-of-distribution. A 0.9x multiplicative penalty is
    applied to the total score (via ``PropertyOracle._compute_uq_penalty``
    and ``_apply_domain_penalty``), and the molecule's SMILES is added
    to the active learning queue for full QuantumOracle (xTB/TOM)
    evaluation in the next generation.

    This prevents the surrogate model from confidently extrapolating
    into chemically invalid regions — the "model extrapolation" failure
    mode identified by the Invert-Always-Invert analysis.
"""

from __future__ import annotations

from aurelius.scoring.oracle.gc import (
    _UQ_PENALTY as _UQ_PENALTY,
)
from aurelius.scoring.oracle.gc import (
    _UQ_THRESHOLD_FRACTION as _UQ_THRESHOLD_FRACTION,
)
from aurelius.scoring.oracle.gc import (
    GcUqEnsemble as GcUqEnsemble,
)

__all__ = [
    "GcUqEnsemble",
    "_UQ_PENALTY",
    "_UQ_THRESHOLD_FRACTION",
]
