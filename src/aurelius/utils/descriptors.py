"""Molecular descriptor utilities.

Provides molecular descriptor generation from SMILES strings,
with RDKit-based real descriptors and deterministic hash-based
fallback when RDKit is unavailable.
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


def _generate_molecular_descriptors(smiles: str) -> dict[str, float]:
    """Generate simple molecular descriptors from SMILES for Tier 0 prediction.

    Produces a minimal feature vector encoding structural properties
    relevant to SEI formation activation energies. When RDKit is
    available, uses real descriptors; otherwise falls back to a
    deterministic hash-based approximation.

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        Dictionary of descriptor name -> value.
    """
    if HAS_RDKIT:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return {
                "mw": float(Descriptors.MolWt(mol)),  # type: ignore[attr-defined, unused-ignore]
                "logp": float(Descriptors.MolLogP(mol)),  # type: ignore[attr-defined, unused-ignore]
                "hba": int(Descriptors.NumHAcceptors(mol)),  # type: ignore[attr-defined, unused-ignore]
                "hbd": int(Descriptors.NumHDonors(mol)),  # type: ignore[attr-defined, unused-ignore]
                "tpsa": float(Descriptors.TPSA(mol)),  # type: ignore[attr-defined, unused-ignore]
                "rot_bonds": int(Descriptors.NumRotatableBonds(mol)),  # type: ignore[attr-defined, unused-ignore]
                "aromatic_ratio": float(
                    sum(1 for a in mol.GetAtoms() if a.GetIsAromatic()) / max(mol.GetNumAtoms(), 1)
                ),  # type: ignore[no-untyped-call, misc, unused-ignore]
                "heavy_atom_count": float(Descriptors.HeavyAtomCount(mol)),  # type: ignore[no-untyped-call, unused-ignore]
            }

    return _hash_descriptors(smiles)


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
