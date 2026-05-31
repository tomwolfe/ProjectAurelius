"""Central type definitions for Project Aurelius.

All dataclass definitions used across the pipeline are centralized here
to eliminate circular imports between modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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

    Usage:
        ctx = MoleculeContext.from_smiles("CCO")
        Pipeline.screen_molecule(ctx)
    """

    smiles: str
    mol: Any  # RDKit Mol object
    fingerprint_ecfp4: Any | None = None  # RDKit ExplicitBitVect (2048 bits)
    feature_vector: Any | None = None  # numpy array (2053-dim)
    _mol_owner: bool = True

    @classmethod
    def from_smiles(cls, smiles: str) -> MoleculeContext | None:
        """Parse a SMILES string into a MoleculeContext.

        Args:
            smiles: SMILES string.

        Returns:
            MoleculeContext with parsed Mol, or None if parsing fails.
        """
        from rdkit import Chem

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
            from rdkit.Chem import AllChem

            self.fingerprint_ecfp4 = AllChem.GetMorganFingerprintAsBitVect(
                self.mol, radius=2, nBits=2048
            )
        return self.fingerprint_ecfp4

    def get_feature_vector(self) -> Any:
        """Get or compute 2053-dim feature vector (lazy)."""
        if self.feature_vector is None:
            import numpy as np
            from rdkit.Chem import AllChem, Descriptors

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
