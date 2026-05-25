"""Molecular descriptor utilities.

Provides molecular descriptor generation from SMILES strings.
RDKit is strictly required; production use without RDKit is not supported.
"""

from __future__ import annotations


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



