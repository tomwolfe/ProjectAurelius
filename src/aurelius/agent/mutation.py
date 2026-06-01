"""Molecule mutation engine for battery electrolyte discovery.

Generates candidate molecules from seed SMILES using two strategies
in priority order:

1. **SMARTS functional-group replacement** — targeted electrolyte-relevant
   transformations (fluorination, methylation, ether/carbonate edits).
2. **BRICS fragmentation + reassembly** — scaffold hopping by breaking
   and reconnecting fragments at retrosynthetically sensible bonds.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS, AllChem, rdMolDescriptors

from aurelius.constants import ELECTROLYTE_MIN_HETEROATOM_RATIO
from aurelius.types import MoleculeContext
from aurelius.utils.chem_utils import _deserialize_fp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Electrolyte-relevant SMARTS reaction library
# ---------------------------------------------------------------------------
ELECTROLYTE_SMARTS: list[tuple[str, str]] = [
    ("[CH3:1]>>[F:1]", "Methyl to fluorine"),
    ("[CH3:1]>>[C:1](F)(F)F", "Methyl to trifluoromethyl"),
    ("[OH:1]>>[F:1]", "Hydroxyl to fluorine"),
    ("[C:1]>>[C:1]OC(F)(F)F", "Add trifluoromethoxy"),
    ("[CH2:1]>>[C:1](F)F", "Methylene to difluoromethylene"),
    ("[C:1](=O)[O:2]>>[C:1](=O)[O:2]C", "Ester to methyl ester"),
    ("[C:1](=O)[OH:1]>>[C:1](=O)[O:1]C", "Carboxylic acid to methyl ester"),
    ("[OH:1]>>[O:1]C(=O)OC", "Hydroxyl to carbonate"),
    ("[OH:1]>>[O:1]C(=O)OCC", "Hydroxyl to ethyl carbonate"),
    ("[OH:1]>>[O:1]C(=O)OC(F)(F)F", "Hydroxyl to fluorinated carbonate"),
    ("[C:1]>>[C:1]OC", "Add methoxy"),
    ("[C:1]>>[C:1]OCC", "Add ethoxy"),
    ("[C:1]>>[C:1]OCCOC", "Add diethylene glycol ether"),
    ("[C:1]>>[C:1]S(=O)(=O)C", "Add methyl sulfone"),
    ("[C:1]>>[C:1]S(=O)(=O)F", "Add sulfonyl fluoride"),
    ("[C:1]>>[C:1]S(=O)(=O)CF", "Add fluoromethyl sulfone"),
    ("[Br:1]>>[C:1]#N", "Bromo to nitrile"),
    ("[C:1]I>>[C:1]#N", "Iodo to nitrile"),
    ("[C:1]>>[C:1]C#N", "Add acetonitrile"),
    ("[OH:1]>>[O:1]P(=O)(OC)OC", "Hydroxyl to dimethyl phosphate"),
    ("[C:1]>>[C:1](C)", "Methylation"),
    ("[C:1]>>[C:1]CC", "Ethylation"),
]

# ---------------------------------------------------------------------------
# Electrochemical Stability SMARTS — filter during mutation to save compute
# ---------------------------------------------------------------------------
# Physical basis: these motifs decompose at anode/cathode potentials:
#   - Peroxides: O-O bond homolysis at < 1 V vs Li/Li+
#   - Acetals: hydrolytic instability in acidic LiPF6 electrolyte
#   - Hemiacetals: same instability as acetals
#   - Epoxides / aziridines: strained 3-rings open at anode potential
#   - geminal diols: unstable toward dehydration

_ELECTROCHEMICALLY_UNSTABLE_SMARTS: list[tuple[str, str]] = [
    ("[OX2][OX2]",            "peroxide"),
    ("[CX4H1]([OX2H0])([OX2H0])",  "acetal"),
    ("[CX4H1]([OX2H0])([OH])",     "hemiacetal"),
    ("[OX2]1[OX2][OX2]1",     "trioxirane"),
    ("[CH2]1[CH2][CH2]1",     "cyclopropane"),
    ("[CH2]1[CH2][CH2][CH2]1", "cyclobutane"),
]

# Hydrolytically unstable patterns (mirrors pipeline.py — checked here too
# to reject unstable candidates before compute-heavy scoring)
_HYDROLYTICALLY_UNSTABLE_SMARTS: list[tuple[str, str]] = [
    ("[CX3](=[OX1])[OX2][CX3](=[OX1])[OX2]", "anhydride"),
    ("[CX3](=[OX1])[OX2][CX2]#[N]",           "acyl_cyanide"),
    ("[SX4](=[OX1])(=[OX1])[OX2][CX3](=[OX1])", "sulfonate_ester"),
    ("[PX4](=[OX1])([OX2][CX4])[OX2][CX4]",   "phosphate_ester"),
    ("[Si]([OX2])[OX2]",                      "silyl_ether"),
    ("[CX3](=[OX1])[OX2][CX2]=[CX2]",         "enol_ester"),
    ("[#6][CX3](=[OX1])[OX2][CX3](=[OX1])[#6]", "geminal_diester"),
]

ELECTROLYTE_FRAGMENT_POOL: list[str] = [
    "COC(=O)OC",
    "CCOC(=O)OCC",
    "O=C1OCCCO1",
    "O=C1OCCO1",
    "O=C1OC(F)CO1",
    "FC(F)(F)OCOC(=O)OC(F)(F)F",
    "CCOC",
    "CCOCC",
    "COCCOC",
    "COCCOCCOC",
    "C1CCOC1",
    "C1COCCO1",
    "CS(=O)(=O)C",
    "CS(=O)(=O)CC",
    "FC(F)(F)S(=O)(=O)C(F)(F)F",
    "CC#N",
    "N#CCC#N",
    "N#CCCC#N",
    "COS(=O)(=O)OC",
    "CF",
    "C(F)(F)F",
    "CC(F)(F)F",
    "COP(=O)(OC)OC",
    "OB(OC)OC",
    "O=C1OC=CO1",
    "O=S1(=O)OCC1",
    "O=S1(=O)OCCO1",
    "FC(F)(F)C(F)(F)F",
    "FC(F)(F)C(F)(F)C(F)(F)F",
]


# ---------------------------------------------------------------------------
# Anti-gaming topology helpers
# ---------------------------------------------------------------------------


def _find_max_conjugated_path(mol: Chem.Mol) -> int:
    """Find the longest conjugated π-system in a molecule (atom count).

    Prevents the mutation engine from creating infinitely conjugated
    structures that would "game" additive property models.
    """
    visited: set[int] = set()
    max_path = [0]

    def _conjugated(a: Chem.Atom, b: Chem.Atom) -> bool:
        bond = mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx())
        if bond is None:
            return False
        if bond.GetIsConjugated():
            return True
        bt = bond.GetBondType()
        if bt in (Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC):
            return True
        return a.GetIsAromatic() or b.GetIsAromatic()

    def _dfs(idx: int, length: int) -> None:
        visited.add(idx)
        max_path[0] = max(max_path[0], length)
        atom = mol.GetAtomWithIdx(idx)
        for nb in atom.GetNeighbors():
            n_idx = nb.GetIdx()
            if n_idx not in visited and _conjugated(atom, nb):
                _dfs(n_idx, length + 1)
        visited.discard(idx)

    for atom in mol.GetAtoms():
        _dfs(atom.GetIdx(), 1)

    return max_path[0]


# ---------------------------------------------------------------------------


class MutationEngine:
    """Multi-strategy molecule mutation engine for battery electrolytes.

    Generates candidate molecules from seed SMILES using two strategies
    in priority order:

    1. SMARTS functional-group replacement (high priority)
    2. BRICS fragmentation + reassembly (medium priority)

    Seeds are stored as MoleculeContext objects to enforce single-point parsing.
    """

    def __init__(self, seed_smiles: list[str] | None = None, known_fps_hex: list[str] | None = None) -> None:
        if seed_smiles is None:
            import json
            import os

            json_path = os.path.join(os.path.dirname(__file__), "..", "data", "tier0_seed_smiles.json")
            with open(json_path) as f:
                seed_smiles = json.load(f)
        self.seed_pool: list[str] = list(set(seed_smiles))

        # Pre-parse seeds into MoleculeContext for fingerprint generation
        self.seed_contexts: list[MoleculeContext] = []
        self.seed_fingerprints: list[Any] = []
        for smi in self.seed_pool:
            ctx = MoleculeContext.from_smiles(smi)
            if ctx is not None:
                self.seed_contexts.append(ctx)
                self.seed_fingerprints.append(ctx.get_ecfp4())

        self.known_fps: list[Any] = []
        for h in known_fps_hex or []:
            try:
                self.known_fps.append(_deserialize_fp(h))
            except Exception:
                continue
        self._load_known_electrolytes()
        self._rng = np.random.default_rng(42)
        self._smarts_rxns: list[tuple[Any, str]] = []
        for smarts, name in ELECTROLYTE_SMARTS:
            try:
                rxn = AllChem.ReactionFromSmarts(smarts)
                self._smarts_rxns.append((rxn, name))
            except Exception:
                logger.debug("Failed to parse SMARTS '%s' (%s)", smarts, name)

    def _load_known_electrolytes(self) -> None:
        """Load known commercial electrolytes into the fingerprint database."""
        import json
        import os

        json_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "known_electrolytes.json"
        )
        try:
            with open(json_path) as f:
                smiles_list = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return

        existing_smis = set()
        for ctx in self.seed_contexts:
            try:
                canon = Chem.MolToSmiles(ctx.mol)
                existing_smis.add(canon)
            except Exception:
                continue

        for smi in smiles_list:
            ctx = MoleculeContext.from_smiles(smi)
            if ctx is not None:
                canon = Chem.MolToSmiles(ctx.mol)
                if canon not in existing_smis:
                    self.known_fps.append(ctx.get_ecfp4())

        logger.info(
            "Loaded %d known electrolyte fingerprints for global novelty checking.",
            len(self.known_fps),
        )

    def fingerprint_db_size(self) -> int:
        return len(self.known_fps)

    def add_to_db(self, smiles: str) -> None:
        """Add a SMILES molecule to the known fingerprint database."""
        ctx = MoleculeContext.from_smiles(smiles)
        if ctx is not None:
            self.known_fps.append(ctx.get_ecfp4())

    def _novelty_check(self, ctx: MoleculeContext) -> bool:
        """Return True if molecule is novel (Tanimoto < 0.85 vs all known)."""
        fp = ctx.get_ecfp4()
        if not self.known_fps:
            return True
        from rdkit.DataStructs import TanimotoSimilarity
        return all(TanimotoSimilarity(fp, known) < 0.85 for known in self.known_fps)

    # ------------------------------------------------------------------
    # Strategy 1: SMARTS functional-group replacement
    # ------------------------------------------------------------------

    def _apply_smarts_reactions(self, ctx: MoleculeContext) -> list[str]:
        """Apply electrolyte-relevant SMARTS transformations.

        Args:
            ctx: Pre-parsed MoleculeContext of the seed molecule.

        Returns:
            List of valid, novel product SMILES strings.
        """
        results: list[str] = []
        for rxn, name in self._smarts_rxns:
            try:
                products = rxn.RunReactants((ctx.mol,))
                for product_tuple in products:
                    for product in product_tuple:
                        if product is None:
                            continue
                        try:
                            Chem.SanitizeMol(product)
                        except Exception:
                            continue
                        product_ctx = MoleculeContext(
                            smiles=Chem.MolToSmiles(product),
                            mol=product,
                        )
                        if not product_ctx.is_valid_electrolyte_mol():
                            continue
                        if not self._novelty_check(product_ctx):
                            continue
                        p_smi = product_ctx.smiles
                        if p_smi and p_smi != ctx.smiles:
                            results.append(p_smi)
            except Exception:
                logger.debug("SMARTS reaction '%s' failed for %s", name, ctx.smiles)
        return list(set(results))

    # ------------------------------------------------------------------
    # Strategy 2: BRICS fragmentation + reassembly
    # ------------------------------------------------------------------

    @staticmethod
    def _load_electrolyte_fragments() -> list[MoleculeContext]:
        """Load the electrolyte fragment pool as MoleculeContext objects."""
        fragments: list[MoleculeContext] = []
        for smi in ELECTROLYTE_FRAGMENT_POOL:
            ctx = MoleculeContext.from_smiles(smi)
            if ctx is not None:
                fragments.append(ctx)
        return fragments

    @staticmethod
    def _is_electrolyte_like(ctx: MoleculeContext) -> bool:
        """Check if a molecule resembles an electrolyte rather than a drug-like compound.

        Battery electrolytes should have:
          - At least one heteroatom (O, S, F, P) for ion solvation
          - Limited aromatic character (drug-like molecules tend to be highly aromatic)
          - A minimum ratio of heteroatoms (O, F) to total heavy atoms
            to prevent BRICS from generating drug-like garbage
          - No electrochemically unstable motifs (peroxides, acetals, strained rings)
          - No hydrolytically unstable motifs (anhydrides, silyl ethers, etc.)
          - [Anti-gaming] Limited conjugated system size (prevents infinitely conjugated
            "Frankenstein" molecules the mutation engine might optimise for)
          - [Anti-gaming] Minimum sp³ carbon fraction (ensures 3D structural complexity)

        Args:
            ctx: Pre-parsed MoleculeContext.

        Returns:
            True if the molecule is electrolyte-like.
        """
        n_arom = rdMolDescriptors.CalcNumAromaticRings(ctx.mol)
        if n_arom > 2:
            return False

        hetero_atoms = {8, 9, 15, 16}
        n_hetero = sum(1 for a in ctx.mol.GetAtoms() if a.GetAtomicNum() in hetero_atoms)
        if n_hetero < 1:
            return False

        # Tightened filter: require minimum heteroatom-to-carbon ratio for BRICS products
        n_total_heavy = sum(1 for a in ctx.mol.GetAtoms() if a.GetAtomicNum() > 1)

        if n_total_heavy > 0:
            o_f_count = sum(
                1 for a in ctx.mol.GetAtoms() if a.GetAtomicNum() in (8, 9)
            )
            ratio = o_f_count / n_total_heavy
            if ratio < ELECTROLYTE_MIN_HETEROATOM_RATIO:
                return False

        # Halogen spam filter: reject molecules where fluorine > 60% of heavy atoms.
        # Prevents mutation engine from replacing every hydrogen with fluorine
        # to artificially inflate scores (e.g., "CF" spam).
        n_f = sum(1 for a in ctx.mol.GetAtoms() if a.GetAtomicNum() == 9)
        if n_total_heavy > 0 and n_f / n_total_heavy > 0.6:
            return False

        # Electrochemical stability: reject molecules with unstable motifs
        for smarts, _name in _ELECTROCHEMICALLY_UNSTABLE_SMARTS:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None and ctx.mol.HasSubstructMatch(pattern):
                return False

        # Hydrolytic stability: reject molecules with hydrolytically unstable motifs
        for smarts, _name in _HYDROLYTICALLY_UNSTABLE_SMARTS:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None and ctx.mol.HasSubstructMatch(pattern):
                return False

        # Strained ring filter: reject molecules with 3- or 4-membered rings
        # (epoxides, azetidines, cyclopropanes — decompose at electrode potentials)
        ring_info = ctx.mol.GetRingInfo()
        if ring_info is not None and ring_info.NumRings() > 0:
            for ring in ring_info.BondRings():
                if len(ring) <= 4:
                    return False
            if ring_info.NumRings() > 3:
                return False

        # Anti-gaming: maximum conjugation path length
        # Prevents the mutation engine from creating infinitely conjugated
        # "Frankenstein" molecules that score well in a purely additive model.
        mol = ctx.mol
        max_conj = _find_max_conjugated_path(mol)
        if max_conj > 16:
            return False

        # Anti-gaming: minimum sp³ carbon fraction
        # Electrolytes should have 3D structural complexity, not flat
        # aromatic sheets. Reject molecules with < 20% sp³ carbons.
        n_sp3 = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and a.GetHybridization() == Chem.HybridizationType.SP3)
        n_c_total = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
        if n_c_total >= 4:
            sp3_frac = n_sp3 / n_c_total
            if sp3_frac < 0.20:
                return False

        return True

    def _brics_from_pool(self, ctx: MoleculeContext) -> list[str]:
        """BRICS decomposition + electrolyte-fragment-guided reassembly."""
        generated: list[str] = []

        try:
            frag_smiles = list(BRICS.BRICSDecompose(ctx.mol))
            frag_ctxs = [
                MoleculeContext.from_smiles(s) for s in frag_smiles
                if MoleculeContext.from_smiles(s) is not None
            ]
            frag_ctxs = [c for c in frag_ctxs if c is not None]

            electrolyte_ctxs = self._load_electrolyte_fragments()
            all_frags = frag_ctxs + electrolyte_ctxs

            if len(all_frags) < 2:
                return generated

            filtered: list[MoleculeContext] = []
            for f_ctx in all_frags:
                f_mw = rdMolDescriptors.CalcExactMolWt(f_ctx.mol)
                f_hbd = rdMolDescriptors.CalcNumHBD(f_ctx.mol)
                if f_mw > 250:
                    continue
                if f_hbd > 0:
                    continue
                filtered.append(f_ctx)
            if len(filtered) < 2:
                return generated
            all_frags = filtered

            n_seed = len(frag_ctxs)
            n_electrolyte = len(electrolyte_ctxs)

            for _ in range(40):
                rng = np.random.default_rng(self._rng.integers(0, 2**31))
                if n_electrolyte > 0 and rng.random() < 0.7:
                    idx1 = n_seed + rng.integers(0, n_electrolyte) if n_electrolyte > 0 else 0
                    idx2 = rng.integers(0, len(all_frags))
                    if idx1 == idx2:
                        idx2 = (idx2 + 1) % len(all_frags)
                    idx = [idx1, idx2]
                else:
                    idx = rng.choice(len(all_frags), size=min(2, len(all_frags)), replace=False)

                try:
                    for r_mol in BRICS.BRICSBuild([all_frags[i].mol for i in idx]):
                        if r_mol is None:
                            continue
                        try:
                            Chem.SanitizeMol(r_mol)
                            s = Chem.MolToSmiles(r_mol, isomericSmiles=True)
                            product_ctx = MoleculeContext(
                                smiles=s,
                                mol=r_mol,
                            )
                            if (
                                product_ctx.is_valid_electrolyte_mol()
                                and self._novelty_check(product_ctx)
                                and self._is_electrolyte_like(product_ctx)
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

        Args:
            smiles: SMILES string of the seed molecule.
            batch_size: Maximum number of variants to return.

        Returns:
            List of candidate SMILES strings.
        """
        ctx = MoleculeContext.from_smiles(smiles)
        if ctx is None:
            return []

        candidates: set[str] = set()

        smarts_results = self._apply_smarts_reactions(ctx)
        candidates.update(smarts_results)

        if len(candidates) < batch_size:
            brics_results = self._brics_from_pool(ctx)
            candidates.update(brics_results)

        result_list = list(candidates)
        if len(result_list) > batch_size:
            indices = self._rng.choice(len(result_list), size=batch_size, replace=False)
            result_list = [result_list[i] for i in indices]

        logger.info(
            "Mutation of %s: %d candidates (%d SMARTS, %d BRICS)",
            smiles, len(result_list), len(smarts_results),
            len(result_list) - len(smarts_results),
        )
        return result_list

    def mutate_batch(self, batch_smiles: list[str], batch_size: int = 50) -> list[str]:
        """Mutate a batch of seed molecules, returning all variants."""
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
        """Generate a large pool of candidate molecules from the seed pool."""
        all_variants: list[str] = []
        for smi in self.seed_pool:
            variants = self.mutate(smi, batch_size)
            all_variants.extend(variants)

        unique = list(dict.fromkeys(all_variants))
        if len(unique) > n_candidates:
            indices = self._rng.choice(len(unique), size=n_candidates, replace=False)
            unique = [unique[i] for i in indices]
        return unique
