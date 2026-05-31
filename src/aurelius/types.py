"""Central type definitions for Project Aurelius.

All dataclass definitions used across the pipeline are centralized here
to eliminate circular imports between modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


@dataclass
class MoleculeInput:
    """Input molecule specification for the Aurelius screening pipeline."""

    smiles: str


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
        """Parse a SMILES string into a MoleculeContext.

        Args:
            smiles: SMILES string.

        Returns:
            MoleculeContext with parsed Mol, or None if parsing fails.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None
        return cls(smiles=smiles, mol=mol)

    def get_ecfp4(self) -> Any:
        """Get or compute ECFP4 fingerprint (lazy)."""
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
        """Check chemical validity and molecular weight bounds.

        Battery electrolyte candidates must have 30 < MW < 1000 Da and
        at least one hydrogen-bond acceptor (essential for Li/Na ion solvation).
        """
        mw = Descriptors.ExactMolWt(self.mol)
        if mw < 30.0 or mw > 1000.0:
            return False
        h_acceptors = Descriptors.NumHAcceptors(self.mol)
        return h_acceptors >= 1

    def count_heteroatoms(self) -> dict[int, int]:
        """Count heteroatoms relevant to electrolytes: O(8), F(9), P(15), S(16)."""
        counts: dict[int, int] = {8: 0, 9: 0, 15: 0, 16: 0}
        for atom in self.mol.GetAtoms():
            z = atom.GetAtomicNum()
            if z in counts:
                counts[z] += 1
        return counts

    def get_tpsa(self) -> float:
        """Return the topological polar surface area."""
        return float(Descriptors.TPSA(self.mol))

    def get_mw(self) -> float:
        """Return the exact molecular weight."""
        return float(Descriptors.ExactMolWt(self.mol))


@dataclass
class OracleResult:
    """Result from the PropertyOracle — predicted HOMO/LUMO properties."""

    homo_eV: float
    lumo_eV: float
    gap_eV: float
    score_eV: float


@dataclass
class AureliusScore:
    """Composite Aurelius score for battery electrolyte screening.

    ``total_score`` is computed via Gaussian penalty approach:
      - LUMO rewarded via Gaussian centered at -1.0 eV, sigma=0.75
      - HOMO penalised via sigmoid when above -6.0 eV
      - SA score penalty for synthetic accessibility
      - Hydrolytic instability penalty
      - Domain applicability penalty for OOD extrapolation
      - Al corrosion penalty for high-LUMO fluorinated molecules

    where total_score is normalized to [0, 100].
    """

    total_score: float
    is_viable: bool
    rejection_reasons: list[str]


__all__ = [
    "AureliusScore",
    "MoleculeContext",
    "MoleculeInput",
    "OracleResult",
]
