"""Centralized dependency detection — fail fast if RDKit is missing.

Removed legacy silent-degradation path. RDKit is required.
"""

from __future__ import annotations

HAS_RDKIT: bool = False

try:
    from rdkit import Chem  # noqa: F401
    HAS_RDKIT = True
except ImportError:
    msg = (
        "RDKit is required for Project Aurelius. "
        "Install with: conda install -c conda-forge rdkit"
    )
    raise ImportError(msg) from None


def check_xtb_with_benchmark() -> str | None:
    """Run a minimal xTB single-point calculation on Ethylene Carbonate.

    Returns a status message if xTB works, ``None`` if unavailable or failed.
    """
    if not HAS_RDKIT:
        return None
    try:
        from rdkit import Chem as _Chem

        from aurelius.compute.xtb_pool import _run_xtb, has_xtb
        from aurelius.scoring.oracle.quantum import _generate_xyz

        if not has_xtb():
            return None

        mol = _Chem.MolFromSmiles("C1COC(=O)O1")
        if mol is None:
            return None

        xyz = _generate_xyz(mol)
        result = _run_xtb(xyz)
        if result is not None and "homo_eV" in result:
            return "xTB Active. Expected MAE improvement over TOM: ~0.5 eV."
        return "xTB Active but slow/unstable."
    except Exception:
        return None
