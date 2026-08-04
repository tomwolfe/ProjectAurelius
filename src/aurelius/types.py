"""Central type definitions for Project Aurelius.

All dataclass definitions used across the pipeline are centralized here
to eliminate circular imports between modules.

MoleculeContext is the absolute single source of truth for molecular
parsing. No module should call ``Chem.MolFromSmiles`` outside of this
class or the mutation engine's fragment pool.

Binary mixtures are represented as compound SMILES: ``SMILES_A|SMILES_B|frac_A``
where ``frac_A`` is the volume fraction of the first component in [0.1, 0.9].
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

MIXTURE_SEPARATOR: str = "|"


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
    lip_solvation_proxy: float | None = None
    sa_score: float | None = None
    synthesis_depth: int | None = None
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

    @classmethod
    def from_brics_fragment(cls, smiles: str) -> MoleculeContext | None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return cls(smiles=smiles, mol=mol)

    def get_ecfp4(self) -> Any:
        if self.fingerprint_ecfp4 is None:
            self.fingerprint_ecfp4 = AllChem.GetMorganFingerprintAsBitVect(
                self.mol, radius=2, nBits=2048
            )
        return self.fingerprint_ecfp4

    @cached_property
    def mw(self) -> float:
        return float(Descriptors.ExactMolWt(self.mol))

    @cached_property
    def logp(self) -> float:
        return float(Descriptors.MolLogP(self.mol))

    @cached_property
    def tpsa(self) -> float:
        return float(Descriptors.TPSA(self.mol))

    @cached_property
    def ring_count(self) -> int:
        return Descriptors.RingCount(self.mol)

    @cached_property
    def rotatable_bonds(self) -> int:
        return Descriptors.NumRotatableBonds(self.mol)

    @cached_property
    def hbd(self) -> int:
        return Descriptors.NumHDonors(self.mol)

    @cached_property
    def hba(self) -> int:
        return Descriptors.NumHAcceptors(self.mol)

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
            arr[2048] = self.mw
            arr[2049] = self.logp
            arr[2050] = self.tpsa
            arr[2051] = self.ring_count
            arr[2052] = self.rotatable_bonds
            self.feature_vector = arr
        return self.feature_vector

    def is_valid_electrolyte_mol(self) -> bool:
        if self.mw < 30.0 or self.mw > 1000.0:
            return False
        return self.hba >= 1

    def count_heteroatoms(self) -> dict[int, int]:
        counts: dict[int, int] = {8: 0, 9: 0, 15: 0, 16: 0}
        for atom in self.mol.GetAtoms():
            z = atom.GetAtomicNum()
            if z in counts:
                counts[z] += 1
        return counts


# ---------------------------------------------------------------------------
# Binary Mixture Support
# ---------------------------------------------------------------------------

_MIXTURE_SEP: str = "|"


def is_mixture_smiles(smiles: str) -> bool:
    """Check if a SMILES string represents a binary mixture (contains '|')."""
    return _MIXTURE_SEP in smiles


def parse_mixture_smiles(smiles: str) -> tuple[str, str, float] | None:
    """Parse a mixture SMILES ``SMILES_A|SMILES_B|frac_A``.

    Returns (smiles_a, smiles_b, frac_a) or None if parsing fails.
    """
    try:
        parts = smiles.split(_MIXTURE_SEP)
        if len(parts) != 3:
            return None
        smi_a, smi_b, frac_str = parts
        frac = float(frac_str)
        if not (0.0 <= frac <= 1.0):
            return None
        return smi_a, smi_b, frac
    except (ValueError, TypeError):
        return None


def format_mixture_smiles(smi_a: str, smi_b: str, frac_a: float) -> str:
    """Format a mixture SMILES string."""
    return f"{smi_a}{_MIXTURE_SEP}{smi_b}{_MIXTURE_SEP}{frac_a:.4f}"


__all__ = [
    "MoleculeContext",
    "ScreeningResult",
    "is_mixture_smiles",
    "parse_mixture_smiles",
    "format_mixture_smiles",
]
