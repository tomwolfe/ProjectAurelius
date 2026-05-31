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
        rdMolDescriptors,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Electrolyte-relevant SMARTS reaction library
# ---------------------------------------------------------------------------
# These reactions guide mutations toward chemical motifs commonly found in
# battery electrolytes: fluorinated chains, carbonates, ethers, sulfones,
# nitriles, and esters.  Each reaction is a simple functional-group
# replacement that preserves the molecular scaffold while introducing
# electrolyte-essential heteroatoms (F, O, S, P).
ELECTROLYTE_SMARTS: list[tuple[str, str]] = [
    # --- Fluorination ---
    ("[CH3:1]>>[F:1]", "Methyl to fluorine"),
    ("[CH3:1]>>[C:1](F)(F)F", "Methyl to trifluoromethyl"),
    ("[OH:1]>>[F:1]", "Hydroxyl to fluorine"),
    ("[C:1]>>[C:1]OC(F)(F)F", "Add trifluoromethoxy"),
    ("[CH2:1]>>[C:1](F)F", "Methylene to difluoromethylene"),
    # --- Carbonate / ester formation ---
    ("[C:1](=O)[O:2]>>[C:1](=O)[O:2]C", "Ester to methyl ester"),
    ("[C:1](=O)[OH:1]>>[C:1](=O)[O:1]C", "Carboxylic acid to methyl ester"),
    ("[OH:1]>>[O:1]C(=O)OC", "Hydroxyl to carbonate"),
    ("[OH:1]>>[O:1]C(=O)OCC", "Hydroxyl to ethyl carbonate"),
    ("[OH:1]>>[O:1]C(=O)OC(F)(F)F", "Hydroxyl to fluorinated carbonate"),
    # --- Ether / alkoxy ---
    ("[C:1]>>[C:1]OC", "Add methoxy"),
    ("[C:1]>>[C:1]OCC", "Add ethoxy"),
    ("[C:1]>>[C:1]OCCOC", "Add diethylene glycol ether"),
    # --- Sulfone / sulfonyl ---
    ("[C:1]>>[C:1]S(=O)(=O)C", "Add methyl sulfone"),
    ("[C:1]>>[C:1]S(=O)(=O)F", "Add sulfonyl fluoride"),
    ("[C:1]>>[C:1]S(=O)(=O)CF", "Add fluoromethyl sulfone"),
    # --- Nitrile ---
    ("[Br:1]>>[C:1]#N", "Bromo to nitrile"),
    ("[C:1]I>>[C:1]#N", "Iodo to nitrile"),
    ("[C:1]>>[C:1]C#N", "Add acetonitrile"),
    # --- Phosphate ---
    ("[OH:1]>>[O:1]P(=O)(OC)OC", "Hydroxyl to dimethyl phosphate"),
    # --- Alkylation ---
    ("[C:1]>>[C:1](C)", "Methylation"),
    ("[C:1]>>[C:1]CC", "Ethylation"),
]

# ---------------------------------------------------------------------------
# Electrolyte-specific fragment pool for BRICS-guided reassembly
# ---------------------------------------------------------------------------
# These are common SEI-forming motifs and electrolyte building blocks.
# The BRICS reassembly is biased to favor connecting these fragments
# rather than generic drug-like fragments.
ELECTROLYTE_FRAGMENT_POOL: list[str] = [
    # Carbonates
    "COC(=O)OC",        # dimethyl carbonate
    "CCOC(=O)OCC",      # diethyl carbonate
    "O=C1OCCCO1",       # propylene carbonate
    "O=C1OCCO1",        # ethylene carbonate
    "O=C1OC(F)CO1",     # fluoroethylene carbonate
    "FC(F)(F)OCOC(=O)OC(F)(F)F",  # fluorinated carbonate
    # Ethers
    "CCOC",             # ethyl methyl ether
    "CCOCC",            # diethyl ether
    "COCCOC",           # dimethoxyethane (glyme)
    "COCCOCCOC",        # triglyme
    "C1CCOC1",          # THF
    "C1COCCO1",         # 1,4-dioxane
    # Sulfones
    "CS(=O)(=O)C",      # dimethyl sulfone
    "CS(=O)(=O)CC",     # ethyl methyl sulfone
    "FC(F)(F)S(=O)(=O)C(F)(F)F",  # perfluorinated sulfone
    # Nitriles
    "CC#N",             # acetonitrile
    "N#CCC#N",          # succinonitrile
    "N#CCCC#N",         # adiponitrile
    # Sulfates / sulfonates
    "COS(=O)(=O)OC",    # dimethyl sulfate
    "CF",               # fluoromethane
    "C(F)(F)F",         # fluoroform (trifluoromethyl)
    "CC(F)(F)F",        # 1,1,1-trifluoroethane
    # Phosphates
    "COP(=O)(OC)OC",    # trimethyl phosphate
    # Borates
    "OB(OC)OC",         # trimethyl borate
    # SEI additives
    "O=C1OC=CO1",       # vinylene carbonate
    "O=S1(=O)OCC1",     # 1,3-propane sultone
    "O=S1(=O)OCCO1",    # ethylene sulfite
    # Fluorinated alkyls
    "FC(F)(F)C(F)(F)F",  # perfluoroethane
    "FC(F)(F)C(F)(F)C(F)(F)F",  # perfluoropropane
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

    def __init__(self, seed_smiles: list[str] | None = None, known_fps_hex: list[str] | None = None) -> None:
        """Initialise the mutation engine.

        Args:
            seed_smiles: List of seed SMILES strings. If None, loads from
                the bundled tier0_seed_smiles.json.
            known_fps_hex: Optional list of known fingerprint hex strings
                for novelty checking.
        """
        if seed_smiles is None:
            import json
            import os

            json_path = os.path.join(os.path.dirname(__file__), "..", "data", "tier0_seed_smiles.json")
            with open(json_path) as f:
                seed_smiles = json.load(f)
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

    @staticmethod
    def _load_electrolyte_fragments() -> list[Any]:
        """Load the electrolyte fragment pool as RDKit Mol objects.

        Returns:
            List of sanitized RDKit Mol objects for electrolyte fragments.
        """
        fragments: list[Any] = []
        for smi in ELECTROLYTE_FRAGMENT_POOL:
            mol = _safe_mol_from_smiles(smi)
            if mol is not None:
                fragments.append(mol)
        return fragments

    @staticmethod
    def _is_electrolyte_like(mol: Any) -> bool:
        """Check if a molecule resembles an electrolyte rather than a drug-like compound.

        Battery electrolytes should have:
          - At least one heteroatom (O, S, F, P) for ion solvation
          - Limited aromatic character (drug-like molecules tend to be highly aromatic)

        Args:
            mol: RDKit Mol object.

        Returns:
            True if the molecule is electrolyte-like.
        """
        from rdkit.Chem import rdMolDescriptors

        n_arom = rdMolDescriptors.CalcNumAromaticRings(mol)
        if n_arom > 2:
            return False
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() in (8, 16, 9, 15):
                return True
        return False

    def _brics_from_pool(self, mol: Any) -> list[str]:
        """BRICS decomposition + electrolyte-fragment-guided reassembly.

        Decomposes the input molecule into BRICS fragments, then extends
        the pool with pre-defined electrolyte building blocks (carbonates,
        sulfones, nitriles, fluorinated alkyls, etc.). Reassembly is
        biased toward connecting these electrolyte fragments to generate
        novel electrolyte-relevant scaffolds.

        Args:
            mol: RDKit Mol object to decompose.

        Returns:
            List of candidate SMILES strings from BRICS reassembly.
        """
        generated: list[str] = []

        try:
            # Decompose the seed molecule
            frag_smiles = list(BRICS.BRICSDecompose(mol))
            frag_mols = [Chem.MolFromSmiles(s) for s in frag_smiles if Chem.MolFromSmiles(s) is not None]

            # Add electrolyte fragment pool
            electrolyte_mols = self._load_electrolyte_fragments()
            all_frags = frag_mols + electrolyte_mols

            if len(all_frags) < 2:
                return generated

            # Filter fragments for compatibility
            filtered: list[Any] = []
            for f_mol in all_frags:
                f_mw = rdMolDescriptors.CalcExactMolWt(f_mol)
                f_hbd = rdMolDescriptors.CalcNumHBD(f_mol)
                if f_mw > 250:
                    continue
                if f_hbd > 0:
                    continue
                filtered.append(f_mol)
            if len(filtered) < 2:
                return generated
            all_frags = filtered

            # Separate seed fragments from electrolyte fragments for biased sampling
            n_seed = len(frag_mols)
            n_electrolyte = len(electrolyte_mols)
            # Bias: 70% chance to include at least one electrolyte fragment
            for _ in range(40):
                rng = np.random.default_rng(self._rng.integers(0, 2**31))
                if n_electrolyte > 0 and rng.random() < 0.7:
                    # Pick one from electrolyte pool, one from anywhere
                    idx1 = n_seed + rng.integers(0, n_electrolyte) if n_electrolyte > 0 else 0
                    idx2 = rng.integers(0, len(all_frags))
                    if idx1 == idx2:
                        idx2 = (idx2 + 1) % len(all_frags)
                    idx = [idx1, idx2]
                else:
                    idx = rng.choice(len(all_frags), size=min(2, len(all_frags)), replace=False)

                try:
                    for r_mol in BRICS.BRICSBuild([all_frags[i] for i in idx]):
                        if r_mol is None:
                            continue
                        try:
                            Chem.SanitizeMol(r_mol)
                            s = Chem.MolToSmiles(r_mol, isomericSmiles=True)
                            if (
                                _is_valid_mol(r_mol)
                                and self._novelty_check(r_mol)
                                and self._is_electrolyte_like(r_mol)
                                and s
                            ):
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
