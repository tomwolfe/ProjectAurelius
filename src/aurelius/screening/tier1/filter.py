"""Tier 1: Electrolyte Viability Filter.

This module provides a fast, deterministic screening step based on
electrolyte-specific viability criteria for battery discovery:

1. **MW < 300 Da** — Electrolyte solvents and additives are small molecules.
2. **H-Bond Donors == 0** — Protic HBDs cause parasitic reactions with
   alkali-metal anodes (Li, Na), leading to SEI instability.
3. **Rotatable Bonds <= 6** — conformational rigidity improves SEI
   packing and reductive stability.
4. **At least one H-Bond Acceptor (O, N, F)** — required for Li+/Na+
   ion solvation.

Together these provide a fast, interpretable gate that eliminates
chemically unsuitable candidates before the expensive Oracle step.

Usage:
    from aurelius.screening.tier1 import Filter

    filt = Filter()
    result = filt.screen_molecule("CC(=O)OC1=CC=CC=C1")
    print(result["is_viable"])  # True
"""

from __future__ import annotations

import time
from typing import Any

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


class Filter:
    """Deterministic electrolyte-viability filter.

    No model training or weight loading occurs — all criteria are
    computed directly from the molecular graph using RDKit.

    Screening criteria:
        - MW < 300 Da (small-molecule electrolyte)
        - H-Bond Donors == 0 (no protic H for reductive stability)
        - Rotatable Bonds <= 6 (conformational rigidity for SEI)
        - At least one H-Bond Acceptor (O, N, F) for ion solvation

    A molecule passes only if **all** criteria are met.

    Example:
        >>> filt = Filter()
        >>> result = filt.screen_molecule("CCOC(=O)C")
        >>> result["is_viable"]
        True
    """

    __slots__ = ()

    # Electrolyte-relevant heteroatoms for HBA count
    _HBA_ELEMENTS = {7, 8, 9}  # N, O, F atomic numbers

    def __init__(self) -> None:
        pass

    def screen_molecule(self, smiles: str) -> dict[str, Any]:
        """Screen a single molecule for electrolyte viability.

        Args:
            smiles: SMILES string of the molecule.

        Returns:
            Dict with keys:
                - ``is_viable`` (bool): True if molecule passes all gates.
                - ``electrolyte_violations`` (list[str]): Reason strings for
                  any electrolyte rule violations.
                - ``inference_time_ms`` (float): Wall-clock time in ms.

        Raises:
            ValueError: If SMILES is invalid or cannot be parsed.
        """
        start = time.perf_counter()

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            raise ValueError(f"Cannot sanitise molecule from SMILES: {smiles}") from exc

        violations: list[str] = []

        mw = rdMolDescriptors.CalcExactMolWt(mol)
        h_donors = rdMolDescriptors.CalcNumHBD(mol)
        n_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)

        # Count HBA by specific heteroatoms relevant to ion solvation
        hba_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in self._HBA_ELEMENTS)

        if mw >= 300:
            violations.append(f"MW too high ({mw:.1f} >= 300)")
        if h_donors > 0:
            violations.append(f"H-bond donors present ({h_donors} > 0) — parasitic SEI reactions")
        if n_rotatable > 6:
            violations.append(f"Too many rotatable bonds ({n_rotatable} > 6)")
        if hba_count < 1:
            violations.append("No H-bond acceptors (O, N, or F) — insufficient ion solvation")

        is_viable = len(violations) == 0

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "is_viable": is_viable,
            "electrolyte_violations": violations,
            "inference_time_ms": round(elapsed_ms, 3),
        }

    def screen_batch(self, smiles_list: list[str], batch_size: int = 32) -> list[dict[str, Any]]:
        """Screen a batch of molecules through the filter.

        Args:
            smiles_list: List of SMILES strings.
            batch_size: Ignored (batching is not necessary for this
                deterministic filter).

        Returns:
            List of result dicts (same structure as ``screen_molecule``).
        """
        return [self.screen_molecule(smiles) for smiles in smiles_list]

    @staticmethod
    def is_viable_smiles(smiles: str) -> bool:
        """Quick check whether a SMILES is structurally viable.

        Convenience wrapper that returns only the viability boolean.

        Args:
            smiles: SMILES string.

        Returns:
            True if the molecule passes all screening gates.
        """
        try:
            result = Filter().screen_molecule(smiles)
            return result["is_viable"]
        except Exception:
            return False
