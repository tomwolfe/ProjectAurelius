"""Molecule mutation engine for battery electrolyte discovery.

Generates candidate molecules from seed SMILES using two strategies
in priority order:

1. **SMARTS functional-group replacement** — targeted electrolyte-relevant
   transformations (fluorination, methylation, ether/carbonate edits).
2. **BRICS fragmentation + reassembly** — scaffold hopping by breaking
   and reconnecting fragments at retrosynthetically sensible bonds.

Requirements:
    - RDKit for SMARTS and BRICS
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from aurelius.utils.chem_utils import (
    _deserialize_fp,
    _is_valid_mol,
    _safe_mol_from_smiles,
)
from aurelius.utils.dependencies import HAS_RDKIT

if HAS_RDKIT:
    from rdkit import Chem
    from rdkit.Chem import (
        BRICS,
        AllChem,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Electrolyte-relevant SMARTS reaction library
# ---------------------------------------------------------------------------
# These reactions guide mutations toward chemical motifs commonly found in
# battery electrolytes: fluorinated chains, carbonates, ethers, and esters.
ELECTROLYTE_SMARTS: list[tuple[str, str]] = [
    ("[CH3:1]>>[F:1]", "Methyl to fluorine"),
    ("[CH3:1]>>[C:1](F)(F)F", "Methyl to trifluoromethyl"),
    ("[OH:1]>>[F:1]", "Hydroxyl to fluorine"),
    ("[C:1]>>[C:1](C)", "Methylation"),
    ("[C:1](=O)[O:2]>>[C:1](=O)[O:2]C", "Ester to methyl ester"),
    ("[C:1]>>[C:1]OC", "Add methoxy"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class MutationEngine:
    """Multi-strategy molecule mutation engine for battery electrolytes.

    Generates candidate molecules from seed SMILES using two strategies
    in priority order:

    1. **SMARTS functional-group replacement** (high probability)
    2. **BRICS fragmentation + reassembly** (medium probability)
    """

    def __init__(self, seed_smiles: list[str], known_fps_hex: list[str] | None = None) -> None:
        """Initialise the mutation engine.

        Args:
            seed_smiles: List of seed SMILES strings.
            known_fps_hex: Optional list of known fingerprint hex strings
                for novelty checking.
        """
        self.seed_pool: list[str] = list(set(seed_smiles))
        self.seed_fingerprints: list[Any] = []
        for smi in self.seed_pool:
            mol = _safe_mol_from_smiles(smi)
            if mol is not None:
                self.seed_fingerprints.append(
                    AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                )
        self.known_fps: list[Any] = []
        for h in known_fps_hex or []:
            try:
                self.known_fps.append(_deserialize_fp(h))
            except Exception:
                continue
        self._rng = np.random.default_rng(42)
        # Pre-decoded SMARTS reactions
        self._smarts_rxns: list[tuple[Any, str]] = []
        for smarts, name in ELECTROLYTE_SMARTS:
            try:
                rxn = AllChem.ReactionFromSmarts(smarts)
                self._smarts_rxns.append((rxn, name))
            except Exception:
                logger.debug("Failed to parse SMARTS '%s' (%s)", smarts, name)

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
            self.known_fps.append(
                AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            )

    def _novelty_check(self, mol: Any) -> bool:
        """Return True if molecule is novel (Tanimoto < 0.75 vs all known).

        Args:
            mol: RDKit Mol object.

        Returns:
            True if novel (all Tanimoto < 0.75).
        """
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        if not self.known_fps:
            return True

        from rdkit.DataStructs import TanimotoSimilarity

        return all(TanimotoSimilarity(fp, known) < 0.75 for known in self.known_fps)

    # ------------------------------------------------------------------
    # Strategy 1: SMARTS functional-group replacement (highest priority)
    # ------------------------------------------------------------------

    def _apply_smarts_reactions(self, smiles: str) -> list[str]:
        """Apply electrolyte-relevant SMARTS transformations to a seed molecule.

        Each reaction produces chemically meaningful variants such as
        fluorination, methylation, or functional-group interconversion.

        Args:
            smiles: Seed SMILES string.

        Returns:
            List of valid, novel product SMILES strings.
        """
        mol = _safe_mol_from_smiles(smiles)
        if mol is None:
            return []

        results: list[str] = []
        for rxn, name in self._smarts_rxns:
            try:
                products = rxn.RunReactants((mol,))
                for product_tuple in products:
                    for product in product_tuple:
                        if product is None:
                            continue
                        try:
                            Chem.SanitizeMol(product)
                        except Exception:
                            continue
                        if not _is_valid_mol(product):
                            continue
                        if not self._novelty_check(product):
                            continue
                        p_smi = Chem.MolToSmiles(product)
                        if p_smi and p_smi != smiles:
                            results.append(p_smi)
            except Exception:
                logger.debug("SMARTS reaction '%s' failed for %s", name, smiles)

        return list(set(results))

    # ------------------------------------------------------------------
    # Strategy 2: BRICS fragmentation + reassembly
    # ------------------------------------------------------------------

    def _brics_from_pool(self, mol: Any) -> list[str]:
        """BRICS decomposition + reassembly using a shared fragment pool.

        Decomposes the input molecule into BRICS fragments, then combines
        fragments from the global seed pool to generate novel scaffolds.

        Args:
            mol: RDKit Mol object to decompose.

        Returns:
            List of candidate SMILES strings from BRICS reassembly.
        """
        generated: list[str] = []

        try:
            frag_smiles = list(BRICS.BRICSDecompose(mol))
            if len(frag_smiles) < 2:
                return generated
            frag_mols = [Chem.MolFromSmiles(s) for s in frag_smiles if Chem.MolFromSmiles(s) is not None]
            if len(frag_mols) < 2:
                return generated

            for _ in range(30):
                rng = np.random.default_rng(self._rng.integers(0, 2**31))
                idx = rng.choice(len(frag_mols), size=min(2, len(frag_mols)), replace=False)
                try:
                    for r_mol in BRICS.BRICSBuild([frag_mols[idx[0]], frag_mols[idx[1]]]):
                        if r_mol is None:
                            continue
                        try:
                            Chem.SanitizeMol(r_mol)
                            s = Chem.MolToSmiles(r_mol, isomericSmiles=True)
                            if _is_valid_mol(r_mol) and self._novelty_check(r_mol) and s:
                                generated.append(s)
                        except Exception:
                            continue
                except Exception:
                    continue
        except Exception:
            logger.debug("BRICS reassembly failed for pool", exc_info=True)

        return list(set(generated))

    # ------------------------------------------------------------------
    # Public mutation API
    # ------------------------------------------------------------------

    def mutate(self, smiles: str, batch_size: int = 50) -> list[str]:
        """Generate up to batch_size mutated variants of a seed molecule.

        Applies the two mutation strategies in priority order:
        1. SMARTS functional-group replacement
        2. BRICS fragmentation + reassembly

        Args:
            smiles: SMILES string of the seed molecule.
            batch_size: Maximum number of variants to return.

        Returns:
            List of candidate SMILES strings.
        """
        candidates: set[str] = set()

        # Strategy 1: SMARTS functional-group replacement
        smarts_results = self._apply_smarts_reactions(smiles)
        candidates.update(smarts_results)

        # Strategy 2: BRICS reassembly
        if len(candidates) < batch_size:
            mol = _safe_mol_from_smiles(smiles)
            if mol is not None:
                brics_results = self._brics_from_pool(mol)
                candidates.update(brics_results)

        result_list = list(candidates)
        if len(result_list) > batch_size:
            indices = self._rng.choice(len(result_list), size=batch_size, replace=False)
            result_list = [result_list[i] for i in indices]

        logger.info(
            "Mutation of %s: %d candidates (%d SMARTS, %d BRICS)",
            smiles,
            len(result_list),
            len(smarts_results),
            len(result_list) - len(smarts_results),
        )
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

    def propose_candidates(
        self,
        n_candidates: int = 1000,
        batch_size: int = 50,
    ) -> list[str]:
        """Generate a large pool of candidate molecules from the seed pool.

        Mutates each seed molecule through the two-strategy pipeline
        (SMARTS → BRICS) and returns a deduplicated pool for
        the Bayesian optimisation loop.

        Args:
            n_candidates: Total number of unique candidates to propose.
            batch_size: Maximum variants per seed molecule.

        Returns:
            Deduplicated list of candidate SMILES strings.
        """
        all_variants: list[str] = []
        for smi in self.seed_pool:
            variants = self.mutate(smi, batch_size)
            all_variants.extend(variants)

        unique = list(dict.fromkeys(all_variants))
        if len(unique) > n_candidates:
            indices = self._rng.choice(len(unique), size=n_candidates, replace=False)
            unique = [unique[i] for i in indices]

        return unique
