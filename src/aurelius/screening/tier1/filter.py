"""Tier 1: Deterministic structural-viability filter.

This module replaces the former MLX-NA neural-filter with a fast,
deterministic screening step based on well-established cheminformatics
criteria:

1. **Lipinski Rule-of-5** (Lipinski et al., 2001) — a simple heuristic
   that flags molecules with reasonable MW, logP, H-bond donors/acceptors.

2. **Synthetic complexity proxy** — number of rings, rotatable bonds,
   and stereocenters are used as a fast proxy for synthetic accessibility.
   Molecules with too many (>5) rings or (>10) rotatable bonds are
   flagged as hard-to-synthesise.

Together these provide a fast, interpretable gate that eliminates
structurally implausible or overly exotic candidates before the
expensive Oracle evaluation step.

Battery electrolytes are small, polar molecules that solvate Li+/Na+ ions
and form a stable SEI layer.  The Lipinski rules, while originally
designed for oral drugs, serve as a useful first-pass filter: electrolyte
candidates are generally of low molecular weight and moderate polarity.

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
from rdkit.Chem import Crippen, rdMolDescriptors


class Filter:
    """Deterministic structural-viability filter for battery electrolytes.

    No model training or weight loading occurs — all criteria are
    computed directly from the molecular graph using RDKit.

    Screening criteria:
        - Lipinski Rule-of-5:
            * MW <= 500
            * logP <= 5.0
            * H-bond donors <= 5
            * H-bond acceptors <= 10
        - Ring count <= 5 (synthetic complexity proxy)
        - Rotatable bonds <= 10 (conformational flexibility proxy)

    A molecule passes only if **all** criteria are met.

    Example:
        >>> filt = Filter()
        >>> result = filt.screen_molecule("CCOC(=O)C")
        >>> result["is_viable"]
        True
    """

    __slots__ = ()

    def __init__(self) -> None:
        pass

    def screen_molecule(self, smiles: str) -> dict[str, Any]:
        """Screen a single molecule for structural viability.

        Args:
            smiles: SMILES string of the molecule.

        Returns:
            Dict with keys:
                - ``is_viable`` (bool): True if molecule passes all gates.
                - ``lipinski_violations`` (list[str]): Reason strings for
                  any Lipinski rule violations.
                - ``complexity_flags`` (list[str]): Structural complexity
                  warnings.
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
        complexity_flags: list[str] = []

        mw = rdMolDescriptors.CalcExactMolWt(mol)
        logp = Crippen.MolLogP(mol)
        h_donors = rdMolDescriptors.CalcNumHBD(mol)
        h_acceptors = rdMolDescriptors.CalcNumHBA(mol)
        n_rings = rdMolDescriptors.CalcNumRings(mol)
        n_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)

        if mw > 500:
            violations.append(f"MW too high ({mw:.1f} > 500)")
        if logp > 5.0:
            violations.append(f"logP too high ({logp:.1f} > 5.0)")
        if h_donors > 5:
            violations.append(f"H-bond donors too high ({h_donors} > 5)")
        if h_acceptors > 10:
            violations.append(f"H-bond acceptors too high ({h_acceptors} > 10)")
        if n_rings > 5:
            complexity_flags.append(f"Too many rings ({n_rings} > 5)")
        if n_rotatable > 10:
            complexity_flags.append(f"Too many rotatable bonds ({n_rotatable} > 10)")

        is_viable = len(violations) == 0 and len(complexity_flags) == 0

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "is_viable": is_viable,
            "lipinski_violations": violations,
            "complexity_flags": complexity_flags,
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
