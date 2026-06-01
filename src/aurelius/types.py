"""Central type definitions for Project Aurelius.

All dataclass definitions used across the pipeline are centralized here
to eliminate circular imports between modules.

MoleculeContext is the absolute single source of truth for molecular
parsing. No module should call ``Chem.MolFromSmiles`` outside of this
class or the mutation engine's fragment pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


@dataclass(frozen=True)
class ScreeningResult:
    """Result from a single molecule screening."""

    smiles: str
    total_score: float
    is_viable: bool
    rejection_reasons: list[str]
    fingerprint: np.ndarray[Any, Any] | None = None
    novelty_to_seed: float | None = None
    homo_eV: float | None = None
    lumo_eV: float | None = None
    dielectric_proxy: float | None = None
    viscosity_proxy: float | None = None
    li_solvation_proxy: float | None = None
    sa_score: float | None = None
    sub_scores: dict[str, float] | None = None


@dataclass
class MoleculeContext:
    """Unified molecular context — parsed exactly once per screening step.

    Holds the SMILES string and its pre-parsed RDKit Mol object, along with
    pre-computed fingerprint and descriptors to avoid redundant computation
    across the Filter, Oracle, and Featurizer stages.

    All RDKit parsing and featurization flows through this class.
    No module should call ``Chem.MolFromSmiles`` or fingerprint generation
    outside of ``MoleculeContext``.

    Usage:
        ctx = MoleculeContext.from_smiles("CCO")
        Pipeline.screen_molecule(ctx)
    """

    smiles: str
    mol: Chem.Mol
    fingerprint_ecfp4: Any | None = None
    feature_vector: np.ndarray[Any, Any] | None = None

    @classmethod
    def from_smiles(cls, smiles: str) -> MoleculeContext | None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None
        return cls(smiles=smiles, mol=mol)

    def get_ecfp4(self) -> Any:
        if self.fingerprint_ecfp4 is None:
            self.fingerprint_ecfp4 = AllChem.GetMorganFingerprintAsBitVect(
                self.mol, radius=2, nBits=2048
            )
        return self.fingerprint_ecfp4

    def get_feature_vector(self) -> np.ndarray[Any, Any]:
        """Get or compute 2053-dim feature vector (lazy).

        Layout:
          - [0:2048]  ECFP4 binary fingerprint (Morgan radius=2, 2048 bits)
          - [2048]    Exact molecular weight
          - [2049]    MolLogP
          - [2050]    TPSA
          - [2051]    Ring count
          - [2052]    NumRotatableBonds
        """
        if self.feature_vector is None:
            fp = self.get_ecfp4()
            arr = np.zeros(2053, dtype=np.float32)
            for idx in fp.GetOnBits():
                arr[idx] = 1.0
            arr[2048] = Descriptors.ExactMolWt(self.mol)
            arr[2049] = Descriptors.MolLogP(self.mol)
            arr[2050] = Descriptors.TPSA(self.mol)
            arr[2051] = Descriptors.RingCount(self.mol)
            arr[2052] = Descriptors.NumRotatableBonds(self.mol)
            self.feature_vector = arr
        return self.feature_vector

    def is_valid_electrolyte_mol(self) -> bool:
        mw = Descriptors.ExactMolWt(self.mol)
        if mw < 30.0 or mw > 1000.0:
            return False
        h_acceptors = Descriptors.NumHAcceptors(self.mol)
        return h_acceptors >= 1

    def count_heteroatoms(self) -> dict[int, int]:
        counts: dict[int, int] = {8: 0, 9: 0, 15: 0, 16: 0}
        for atom in self.mol.GetAtoms():
            z = atom.GetAtomicNum()
            if z in counts:
                counts[z] += 1
        return counts

    def get_tpsa(self) -> float:
        return float(Descriptors.TPSA(self.mol))

    def get_mw(self) -> float:
        return float(Descriptors.ExactMolWt(self.mol))


__all__ = [
    "MoleculeContext",
    "ScreeningResult",
]
