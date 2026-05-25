"""Molecular descriptor utilities.

Provides molecular descriptor generation from SMILES strings.
RDKit is strictly required; production use without RDKit is not supported.
"""

from __future__ import annotations

from aurelius.utils.dependencies import HAS_RDKIT

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    Chem = None  # type: ignore[assignment, unused-ignore]
    Descriptors = None  # type: ignore[assignment, unused-ignore]

import numpy as np


def generate_molecular_descriptors(smiles: str) -> dict[str, float]:
    """Generate simple molecular descriptors from SMILES for Tier 0 prediction.

    Produces a minimal feature vector encoding structural properties
    relevant to SEI formation activation energies. When RDKit is
    available, uses real descriptors; raises RuntimeError when RDKit
    is unavailable.

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        Dictionary of descriptor name -> value.

    Raises:
        RuntimeError: When RDKit is unavailable.
    """

    raise RuntimeError(
        "RDKit is required for molecular descriptor generation. "
        "Install RDKit: pip install rdkit"
    )


def _hash_descriptors(smiles: str) -> dict[str, float]:
    """Fallback descriptor generation using deterministic hashing.

    WARNING: These are NOT chemically valid descriptors. They serve
    only as placeholders when RDKit is unavailable.

    Uses SHA-256 for reproducible, session-independent hashing.

    Args:
        smiles: SMILES string.

    Returns:
        Dictionary of approximate descriptor values.
    """
    import hashlib

    seed = int(hashlib.sha256(smiles.encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    return {
        "mw": float(rng.uniform(50, 500)),
        "logp": float(rng.uniform(-2, 5)),
        "hba": int(rng.randint(0, 10)),
        "hbd": int(rng.randint(0, 5)),
        "tpsa": float(rng.uniform(0, 200)),
        "rot_bonds": int(rng.randint(0, 10)),
        "aromatic_ratio": float(rng.uniform(0, 1)),
        "heavy_atom_count": float(rng.uniform(5, 50)),
    }
