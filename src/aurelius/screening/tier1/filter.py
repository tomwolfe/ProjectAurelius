"""Tier 1: Electrolyte Viability Filter.

Provides a fast, deterministic screening step based on
electrolyte-specific viability criteria for battery discovery:

1. **MW < 300 Da** — Electrolyte solvents and additives are small molecules.
2. **H-Bond Donors == 0** — Protic HBDs cause parasitic reactions with
   alkali-metal anodes (Li, Na), leading to SEI instability.
3. **Rotatable Bonds <= 6** — conformational rigidity improves SEI
   packing and reductive stability.
4. **At least one H-Bond Acceptor (O, N, F)** — required for Li+/Na+
   ion solvation.
5. **LogP <= 2.5** — Electrolytes must be highly polar to dissolve
   Li/Na salts; high LogP indicates poor salt solubility.

Accepts a pre-parsed ``MoleculeContext`` to avoid redundant RDKit parsing.

Usage:
    from aurelius.screening.tier1 import Filter
    from aurelius.types import MoleculeContext

    ctx = MoleculeContext.from_smiles("CC(=O)OC1=CC=CC=C1")
    result = filt.screen(ctx)
    print(result["is_viable"])
"""

from __future__ import annotations

import time
from typing import Any

from rdkit.Chem import Descriptors, rdMolDescriptors

from aurelius.types import MoleculeContext


class Filter:
    """Deterministic electrolyte-viability filter.

    No model training or weight loading occurs — all criteria are
    computed directly from the molecular graph using RDKit.
    """

    __slots__ = ()

    _HBA_ELEMENTS = {7, 8, 9}

    def __init__(self) -> None:
        pass

    def screen(self, ctx: MoleculeContext) -> dict[str, Any]:
        """Screen a pre-parsed molecule for electrolyte viability.

        Args:
            ctx: Pre-parsed MoleculeContext.

        Returns:
            Dict with keys:
                - is_viable (bool)
                - electrolyte_violations (list[str])
                - inference_time_ms (float)

        Raises:
            TypeError: If ctx is not a MoleculeContext.
        """
        if not isinstance(ctx, MoleculeContext):
            raise TypeError(
                f"Filter.screen() requires a MoleculeContext, got {type(ctx).__name__}."
            )

        start = time.perf_counter()
        violations: list[str] = []

        mw = rdMolDescriptors.CalcExactMolWt(ctx.mol)
        h_donors = rdMolDescriptors.CalcNumHBD(ctx.mol)
        n_rotatable = rdMolDescriptors.CalcNumRotatableBonds(ctx.mol)
        logp = Descriptors.MolLogP(ctx.mol)

        hba_count = sum(
            1 for a in ctx.mol.GetAtoms() if a.GetAtomicNum() in self._HBA_ELEMENTS
        )

        if mw >= 300:
            violations.append(f"MW too high ({mw:.1f} >= 300)")
        if h_donors > 0:
            violations.append(f"H-bond donors present ({h_donors} > 0) — parasitic SEI reactions")
        if n_rotatable > 6:
            violations.append(f"Too many rotatable bonds ({n_rotatable} > 6)")
        if hba_count < 1:
            violations.append("No H-bond acceptors (O, N, or F) — insufficient ion solvation")
        if logp > 2.5:
            violations.append(f"LogP too high ({logp:.2f} > 2.5) — poor salt solubility")

        is_viable = len(violations) == 0
        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "is_viable": is_viable,
            "electrolyte_violations": violations,
            "inference_time_ms": round(elapsed_ms, 3),
        }

    def screen_smiles(self, smiles: str) -> dict[str, Any]:
        """Convenience: parse SMILES then screen.

        Args:
            smiles: SMILES string.

        Returns:
            Result dict (same as screen()).
        """
        ctx = MoleculeContext.from_smiles(smiles)
        if ctx is None:
            return {
                "is_viable": False,
                "electrolyte_violations": ["Invalid SMILES"],
                "inference_time_ms": 0.0,
            }
        return self.screen(ctx)

    def screen_batch(self, contexts: list[MoleculeContext]) -> list[dict[str, Any]]:
        """Screen a batch of pre-parsed molecules.

        Args:
            contexts: List of MoleculeContext objects.

        Returns:
            List of result dicts.
        """
        return [self.screen(ctx) for ctx in contexts]

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
            result = Filter().screen_smiles(smiles)
            return result["is_viable"]
        except Exception:
            return False
