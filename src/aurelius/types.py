"""Central type definitions for Project Aurelius.

All dataclass definitions used across the pipeline are centralized here
to eliminate circular imports between modules.

MoleculeContext is the absolute single source of truth for molecular
parsing. No module should call ``Chem.MolFromSmiles`` outside of this
class or the mutation engine's fragment pool.

Mixtures are represented as compound SMILES with N-1 trailing fractions:
    binary:  ``SMILES_A|SMILES_B|frac_A``
    ternary: ``SMILES_A|SMILES_B|SMILES_C|frac_A|frac_B``
where the last fraction is implied (1 - sum of the others). Each
fraction is a volume fraction in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

MIXTURE_SEPARATOR: str = "|"


def get_ecfp4_batch(contexts: list[MoleculeContext]) -> list[Any]:
    """Generate ECFP4 fingerprints for a batch of MoleculeContext objects.

    Uses a list comprehension over RDKit's MorganGenerator for
    consistent, fast batch generation.

    Args:
        contexts: List of pre-parsed MoleculeContext objects.

    Returns:
        List of RDKit fingerprint objects (one per context).
    """
    return [
        AllChem.GetMorganFingerprintAsBitVect(ctx.mol, radius=2, nBits=2048)
        for ctx in contexts
    ]


def fingerprints_to_numpy(
    fps: list[Any], n_bits: int = 2048, dtype: np.dtype[np.float32] = np.float32,
) -> np.ndarray[Any, Any]:
    """Convert a list of RDKit fingerprints to a 2D numpy array.

    Args:
        fps: List of RDKit fingerprint objects.
        n_bits: Number of bits in each fingerprint.
        dtype: Numpy dtype for the output array.

    Returns:
        2D array of shape (n_fingerprints, n_bits).
    """
    arr = np.zeros((len(fps), n_bits), dtype=dtype)
    for i, fp in enumerate(fps):
        for idx in fp.GetOnBits():
            arr[i, idx] = 1.0
    return arr


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

    @classmethod
    def get_ecfp4_batch(cls, contexts: list[MoleculeContext]) -> list[Any]:
        """Generate ECFP4 fingerprints for a batch of MoleculeContext objects.

        Uses a list comprehension over RDKit's MorganGenerator for
        consistent, fast batch generation.

        Args:
            contexts: List of pre-parsed MoleculeContext objects.

        Returns:
            List of RDKit fingerprint objects (one per context).
        """
        return [
            AllChem.GetMorganFingerprintAsBitVect(ctx.mol, radius=2, nBits=2048)
            for ctx in contexts
        ]

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
# Mixture Support (binary and ternary)
# ---------------------------------------------------------------------------

_MIXTURE_SEP: str = "|"


def is_mixture_smiles(smiles: str) -> bool:
    """Check if a SMILES string represents a mixture (contains '|')."""
    return _MIXTURE_SEP in smiles


def parse_mixture_smiles_n(smiles: str) -> tuple[list[str], list[float]] | None:
    """Parse an N-component mixture SMILES.

    Format: ``SMI_1|...|SMI_N|frac_1|...|frac_{N-1}`` where the final
    component fraction is implied. A binary mixture is ``A|B|frac_A`` and
    a ternary mixture is ``A|B|C|frac_A|frac_B``.

    Returns:
        (list_of_smiles, list_of_fractions) with fractions summing to 1.0,
        or None if the string cannot be parsed.
    """
    parts = smiles.split(_MIXTURE_SEP)
    n_fields = len(parts)
    if n_fields < 3 or n_fields % 2 == 0:
        return None
    n_components = (n_fields + 1) // 2
    smi_list = parts[:n_components]
    frac_list: list[float] = []
    for frac_str in parts[n_components:]:
        try:
            frac = float(frac_str)
        except ValueError:
            return None
        if not (0.0 <= frac <= 1.0):
            return None
        frac_list.append(frac)
    last = 1.0 - sum(frac_list)
    if not (0.0 <= last <= 1.0):
        return None
    frac_list.append(last)
    return smi_list, frac_list


def parse_mixture_smiles(smiles: str) -> tuple[str, str, float] | None:
    """Parse a binary mixture SMILES ``SMILES_A|SMILES_B|frac_A``.

    Returns (smiles_a, smiles_b, frac_a) or None if parsing fails.
    """
    parsed = parse_mixture_smiles_n(smiles)
    if parsed is None:
        return None
    smi_list, fracs = parsed
    if len(smi_list) != 2:
        return None
    return smi_list[0], smi_list[1], fracs[0]


def format_mixture_smiles(smi_a: str, smi_b: str, frac_a: float) -> str:
    """Format a binary mixture SMILES string."""
    return f"{smi_a}{_MIXTURE_SEP}{smi_b}{_MIXTURE_SEP}{frac_a:.4f}"


def format_mixture_smiles_n(smi_list: list[str], fractions: list[float]) -> str:
    """Format an N-component mixture SMILES string.

    ``fractions`` must be the full N-fraction list summing to 1.0; the
    first N-1 values are written and the last is implied as the remainder.
    """
    parts: list[str] = list(smi_list)
    parts.extend(f"{f:.4f}" for f in fractions[:-1])
    return _MIXTURE_SEP.join(parts)


__all__ = [
    "MoleculeContext",
    "ScreeningResult",
    "is_mixture_smiles",
    "parse_mixture_smiles",
    "parse_mixture_smiles_n",
    "format_mixture_smiles",
    "format_mixture_smiles_n",
    "get_ecfp4_batch",
    "fingerprints_to_numpy",
]
