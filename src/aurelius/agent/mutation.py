"""RDKit-based molecule mutation engine.

Generates candidate molecules from seed SMILES using:
- BRICS reassembly
- Fluorination
- Methylation
- Electrochemical stability filtering
- Diversity-based rejection
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

import contextlib
import logging

import numpy as np

from aurelius.utils.chem_utils import (
    _deserialize_fp,
    _is_valid_mol,
    _mol_to_fp,
    _safe_mol_from_smiles,
    _tanimoto,
)
from aurelius.utils.dependencies import HAS_RDKIT

if HAS_RDKIT:
    from rdkit import Chem  # type: ignore[import-not-found, unused-ignore]
    from rdkit.Chem import (
        BRICS,  # type: ignore[import-not-found, unused-ignore]
        Descriptors,  # type: ignore[import-not-found, unused-ignore]
    )

logger = logging.getLogger(__name__)


class MutationEngine:
    """RDKit-based molecule mutation engine with BRICS reassembly.

    Generates candidate molecules from seed SMILES using:
    - BRICS reassembly
    - Fluorination
    - Methylation
    - Electrochemical stability filtering
    - Diversity-based rejection
    """

    def __init__(self, seed_smiles: list[str], known_fps_hex: list[str] | None = None) -> None:
        """Initialize the mutation engine.

        Args:
            seed_smiles: List of seed SMILES strings.
            known_fps_hex: Optional list of known fingerprint hex strings
                for novelty checking.
        """
        self.seed_pool: list[str] = list(set(seed_smiles))
        self.known_fps: list[Any] = []
        for h in known_fps_hex or []:
            with contextlib.suppress(Exception):
                self.known_fps.append(_deserialize_fp(h))
        self._rng = np.random.RandomState(42)

    def fingerprint_db_size(self) -> int:
        """Return the number of known fingerprints in the database."""
        return len(self.known_fps)

    def add_to_db(self, smiles: str) -> None:
        """Add a SMILES molecule to the known fingerprint database.

        Args:
            smiles: SMILES string to add.
        """
        mol = _safe_mol_from_smiles(smiles)
        if mol is not None:
            self.known_fps.append(_mol_to_fp(mol))

    def _novelty_check(self, mol: Any) -> bool:
        """Return True if molecule is novel (Tanimoto < 0.75 vs all known).

        Args:
            mol: RDKit Mol object.

        Returns:
            True if novel (all Tanimoto < 0.75).
        """
        fp = _mol_to_fp(mol)
        return all(_tanimoto(fp, known) < 0.75 for known in self.known_fps)

    def _brics_reassemble(self, mol: Any) -> list[str]:
        """BRICS decomposition + random reassembly using proper RDKit types."""
        generated: list[str] = []
        try:
            frag_smiles = list(BRICS.BRICSDecompose(mol))  # type: ignore[no-untyped-call]
            if len(frag_smiles) < 2:
                return generated

            frag_mols = [Chem.MolFromSmiles(s) for s in frag_smiles]
            frag_mols = [m for m in frag_mols if m is not None]
            if len(frag_mols) < 2:
                return generated

            for _ in range(20):
                rng = np.random.RandomState(self._rng.randint(0, 2**31))
                idx = rng.choice(len(frag_mols), size=min(2, len(frag_mols)), replace=False)
                try:
                    result_gen = BRICS.BRICSBuild([frag_mols[idx[0]], frag_mols[idx[1]]])  # type: ignore[no-untyped-call]
                    for r_mol in result_gen:
                        if r_mol is not None:
                            try:
                                Chem.SanitizeMol(r_mol)
                                s = Chem.MolToSmiles(r_mol, isomericSmiles=True)
                                generated.append(s)
                            except (RuntimeError, ValueError) as e:
                                logger.debug("RDKit operation failed: %s", e)
                except (RuntimeError, ValueError) as e:
                    logger.debug("BRICS build failed: %s", e)
        except (RuntimeError, ValueError) as e:
            logger.debug("BRICS reassembly failed: %s", e)
        return list(set(generated))

    def _fluorinate(self, mol: Any) -> list[str]:
        """Add fluorine to non-carbonyl carbons using RDKit RWMol."""
        generated: list[str] = []
        try:
            mol_h = Chem.AddHs(mol)
            c_atoms = [
                atom.GetIdx() for atom in mol_h.GetAtoms() if atom.GetAtomicNum() == 6 and atom.GetTotalDegree() < 4  # type: ignore[no-untyped-call]
            ]
            if not c_atoms:
                return generated

            rng = np.random.RandomState(self._rng.randint(0, 2**31))
            for idx in rng.choice(c_atoms, size=min(5, len(c_atoms)), replace=False):
                rw_mol = Chem.RWMol(mol_h)
                h_idx = None
                for neighbor in rw_mol.GetNeighbors(rw_mol.GetAtomWithIdx(idx)):
                    if neighbor.GetAtomicNum() == 1:
                        h_idx = neighbor.GetIdx()
                        break
                if h_idx is not None:
                    rw_mol.ReplaceAtom(h_idx, Chem.Atom(9))  # 9 = Fluorine
                    try:
                        Chem.SanitizeMol(rw_mol)
                        final_mol = Chem.RemoveHs(rw_mol)
                        s = Chem.MolToSmiles(final_mol, isomericSmiles=True)
                        if Descriptors.ExactMolWt(final_mol) < 450:  # type: ignore[attr-defined]
                            generated.append(s)
                    except (RuntimeError, ValueError) as e:
                        logger.debug("RDKit operation failed: %s", e)
        except (RuntimeError, ValueError) as e:
            logger.debug("Fluorination failed: %s", e)
        return generated

    def _methylate(self, mol: Any) -> list[str]:
        """Add methyl groups using RDKit RWMol."""
        generated: list[str] = []
        try:
            mol_h = Chem.AddHs(mol)
            c_atoms = [
                atom.GetIdx() for atom in mol_h.GetAtoms() if atom.GetAtomicNum() == 6 and atom.GetTotalDegree() < 4  # type: ignore[no-untyped-call]
            ]
            if not c_atoms:
                return generated

            rng = np.random.RandomState(self._rng.randint(0, 2**31))
            for idx in rng.choice(c_atoms, size=min(5, len(c_atoms)), replace=False):
                rw_mol = Chem.RWMol(mol_h)
                h_idx = None
                for neighbor in rw_mol.GetNeighbors(rw_mol.GetAtomWithIdx(idx)):
                    if neighbor.GetAtomicNum() == 1:
                        h_idx = neighbor.GetIdx()
                        break
                if h_idx is not None:
                    rw_mol.ReplaceAtom(h_idx, Chem.Atom(6))  # 6 = Carbon (Methyl)
                    try:
                        Chem.SanitizeMol(rw_mol)
                        final_mol = Chem.RemoveHs(rw_mol)
                        s = Chem.MolToSmiles(final_mol, isomericSmiles=True)
                        if Descriptors.ExactMolWt(final_mol) < 450:  # type: ignore[attr-defined]
                            generated.append(s)
                    except (RuntimeError, ValueError) as e:
                        logger.debug("RDKit operation failed: %s", e)
        except (RuntimeError, ValueError) as e:
            logger.debug("Methylation failed: %s", e)
        return generated

    def mutate(self, smiles: str, batch_size: int = 50) -> list[str]:
        """Generate up to batch_size mutated variants of a seed molecule.

        Applies electrochemical stability filters and diversity checks.

        Args:
            smiles: SMILES string of the seed molecule.
            batch_size: Maximum number of variants to return.

        Returns:
            List of candidate SMILES strings.
        """
        mol = _safe_mol_from_smiles(smiles)
        if mol is None:
            return []

        candidates: set[str] = set()
        candidates.add(smiles)

        # Priority: BRICS first
        brics_results = self._brics_reassemble(mol)
        for s in brics_results:
            m = _safe_mol_from_smiles(s)
            if m is not None and _is_valid_mol(m) and self._novelty_check(m):
                candidates.add(s)

        # Fallback templates
        for func in [self._fluorinate, self._methylate]:
            results = func(mol)
            for s in results:
                m = _safe_mol_from_smiles(s)
                if m is not None and _is_valid_mol(m) and self._novelty_check(m):
                    candidates.add(s)

        # If BRICS yielded nothing, try fallback templates more aggressively
        if len(brics_results) == 0:
            for func in [self._fluorinate, self._methylate]:
                results = func(mol)
                for s in results:
                    m = _safe_mol_from_smiles(s)
                    if m is not None and _is_valid_mol(m) and self._novelty_check(m):
                        candidates.add(s)

        result_list = list(candidates)
        if len(result_list) > batch_size:
            indices = self._rng.choice(len(result_list), size=batch_size, replace=False)
            result_list = [result_list[i] for i in indices]
        return result_list

    def mutate_batch(self, batch_smiles: list[str], batch_size: int = 50) -> list[str]:
        """Mutate a batch of seed molecules, returning all variants.

        Args:
            batch_smiles: List of seed SMILES strings.
            batch_size: Maximum number of variants per seed.

        Returns:
            Deduplicated list of candidate SMILES strings.
        """
        all_variants: list[str] = []
        for smi in batch_smiles:
            variants = self.mutate(smi, batch_size)
            all_variants.extend(variants)
        return list(set(all_variants))

