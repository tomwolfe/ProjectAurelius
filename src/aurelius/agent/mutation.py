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
from collections.abc import Callable
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS, AllChem, rdMolDescriptors
from rdkit.DataStructs import BulkTanimotoSimilarity

from aurelius.constants import (
    ELECTROCHEMICALLY_UNSTABLE_PATTERNS as _EC_UNSTABLE_PATTERNS,
    ELECTROLYTE_MIN_HETEROATOM_RATIO,
    HYDROLYTICALLY_UNSTABLE_PATTERNS as _HYDRO_UNSTABLE_PATTERNS,
)
from aurelius.types import MoleculeContext
from aurelius.utils.chem_utils import _deserialize_fp

try:
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:
    MurckoScaffold = None

# Maximum number of dynamically harvested fragments to keep.
# Prevents memory bloat and combinatorial explosion over many generations.
_MAX_HARVESTED_FRAGMENTS = 200

# Universal BRICS linker fragments for fallback when complementary pairs are sparse.
# Each linker has multiple BRICS dummy-atom isotopes (e.g. {1, 2}) so it can bridge
# fragments that have non-overlapping isotope sets.
# Format: (SMILES, description)
_BRICS_LINKER_FRAGMENTS: list[tuple[str, str]] = [
    ("[1*]O[2*]", "ether linker"),
    ("[1*]C(=O)O[2*]", "ester linker"),
    ("[1*]C(=O)[2*]", "ketone linker"),
    ("[1*]C[2*]", "methylene linker"),
    ("[1*]CCO[2*]", "ethoxy linker"),
    ("[1*]S(=O)(=O)[2*]", "sulfone linker"),
]

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


def _get_brics_types(frag: Chem.Mol) -> set[int]:
    """Extract BRICS dummy-atom isotope types from a fragment.

    BRICS decomposition produces fragments with dummy atoms labelled by
    isotope (1–5).  For two fragments to be joined by BRICSBuild they
    must share at least one common dummy-atom type.
    """
    types: set[int] = set()
    for atom in frag.GetAtoms():
        if atom.GetAtomicNum() == 0:
            iso = atom.GetIsotope()
            if iso:
                types.add(iso)
    return types


# ---------------------------------------------------------------------------
# Data-driven electrolyte-likeness validators
# ---------------------------------------------------------------------------
# Each validator is a (name, predicate) tuple. The predicate receives a
# MoleculeContext and returns True if the check passes (electrolyte-like).
# This replaces a 60-line wall of sequential if-blocks with a declarative,
# composable rule set that is easy to audit, extend, or prune.
# ---------------------------------------------------------------------------

_ELECTROLYTE_CHECKS: list[tuple[str, Callable[[MoleculeContext], bool]]] = []


def _register(fn: Callable[[MoleculeContext], bool]) -> Callable[[MoleculeContext], bool]:
    _ELECTROLYTE_CHECKS.append((fn.__name__, fn))
    return fn


@_register
def aromatic_ring_limit(ctx: MoleculeContext) -> bool:
    return rdMolDescriptors.CalcNumAromaticRings(ctx.mol) <= 2


@_register
def has_heteroatom(ctx: MoleculeContext) -> bool:
    hetero_atoms = {8, 9, 15, 16}
    return sum(1 for a in ctx.mol.GetAtoms() if a.GetAtomicNum() in hetero_atoms) >= 1


@_register
def heteroatom_ratio_min(ctx: MoleculeContext) -> bool:
    n_total = sum(1 for a in ctx.mol.GetAtoms() if a.GetAtomicNum() > 1)
    if n_total == 0:
        return True
    o_f = sum(1 for a in ctx.mol.GetAtoms() if a.GetAtomicNum() in (8, 9))
    return o_f / n_total >= ELECTROLYTE_MIN_HETEROATOM_RATIO


@_register
def halogen_ratio_limit(ctx: MoleculeContext) -> bool:
    n_total = sum(1 for a in ctx.mol.GetAtoms() if a.GetAtomicNum() > 1)
    if n_total == 0:
        return True
    n_f = sum(1 for a in ctx.mol.GetAtoms() if a.GetAtomicNum() == 9)
    n_cl = sum(1 for a in ctx.mol.GetAtoms() if a.GetAtomicNum() == 17)
    n_br = sum(1 for a in ctx.mol.GetAtoms() if a.GetAtomicNum() == 35)
    n_halogen = n_f + n_cl + n_br
    n_heavy_halogen = n_cl + n_br
    if n_halogen / n_total > 0.9:
        return False
    if n_heavy_halogen / n_total > 0.5:
        return False
    if n_halogen > n_total * 0.6:
        n_o = sum(1 for a in ctx.mol.GetAtoms() if a.GetAtomicNum() == 8)
        n_n = sum(1 for a in ctx.mol.GetAtoms() if a.GetAtomicNum() == 7)
        if n_o + n_n == 0:
            return False
    return True


@_register
def electrochemically_stable(ctx: MoleculeContext) -> bool:
    for pattern, _name in _EC_UNSTABLE_PATTERNS:
        if pattern is not None and ctx.mol.HasSubstructMatch(pattern):
            return False
    return True


@_register
def hydrolytically_stable(ctx: MoleculeContext) -> bool:
    for pattern, _name, _severity in _HYDRO_UNSTABLE_PATTERNS:
        if pattern is not None and ctx.mol.HasSubstructMatch(pattern):
            return False
    return True


@_register
def ring_strain_limit(ctx: MoleculeContext) -> bool:
    ring_info = ctx.mol.GetRingInfo()
    if ring_info is not None and ring_info.NumRings() > 0:
        for ring in ring_info.BondRings():
            if len(ring) <= 4:
                return False
        if ring_info.NumRings() > 3:
            return False
    return True


@_register
def conjugation_limit(ctx: MoleculeContext) -> bool:
    return _find_max_conjugated_path(ctx.mol) <= 16


@_register
def sp3_fraction_min(ctx: MoleculeContext) -> bool:
    n_sp3 = sum(1 for a in ctx.mol.GetAtoms() if a.GetAtomicNum() == 6 and a.GetHybridization() == Chem.HybridizationType.SP3)
    n_c = sum(1 for a in ctx.mol.GetAtoms() if a.GetAtomicNum() == 6)
    if n_c >= 4:
        return n_sp3 / n_c >= 0.20
    return True


@_register
def valence_sanity(ctx: MoleculeContext) -> bool:
    max_valence: dict[int, int] = {6: 4, 7: 3, 8: 2, 9: 1, 15: 5, 16: 6, 17: 1, 35: 1}
    for atom in ctx.mol.GetAtoms():
        z = atom.GetAtomicNum()
        if z in max_valence and atom.GetExplicitValence() > max_valence[z]:
            return False
    return True


@_register
def polarity_ratio_min(ctx: MoleculeContext) -> bool:
    mw = ctx.mw
    tpsa = ctx.tpsa
    if mw > 200 and tpsa / mw < 0.05:
        return False
    return True


def _is_electrolyte_like(ctx: MoleculeContext) -> bool:
    for name, check in _ELECTROLYTE_CHECKS:
        if not check(ctx):
            return False
    return True


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

        # Static commercial electrolyte fingerprints (loaded once from JSON)
        self._commercial_fps: list[Any] = []
        for h in known_fps_hex or []:
            try:
                self._commercial_fps.append(_deserialize_fp(h))
            except Exception:
                continue

        # Canonical SMILES sets for O(1) exact duplicate prevention
        self._seed_smiles: set[str] = set()
        self._known_smiles: set[str] = set()
        self._generated_smiles: set[str] = set()
        self._seed_scaffolds: set[str] = set()
        for ctx in self.seed_contexts:
            try:
                canon = Chem.MolToSmiles(ctx.mol)
                self._seed_smiles.add(canon)
                if MurckoScaffold is not None:
                    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=ctx.mol)
                    if scaffold:
                        self._seed_scaffolds.add(scaffold)
            except Exception:
                continue

        self._load_known_electrolytes()
        # Dynamic fragment pool: harvested from high-scoring discoveries
        # Capped at _MAX_HARVESTED_FRAGMENTS to prevent memory bloat
        self._harvested_fragments: list[str] = []
        self._harvested_fragment_set: set[str] = set()
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
                    self._commercial_fps.append(ctx.get_ecfp4())
                    self._known_smiles.add(canon)

        logger.info(
            "Loaded %d known electrolyte fingerprints for global novelty checking.",
            len(self._commercial_fps),
        )

    def commercial_db_size(self) -> int:
        return len(self._commercial_fps)

    def add_to_db(self, smiles: str) -> None:
        """Add a generated SMILES to the exact-duplicate set (O(1) lookup)."""
        ctx = MoleculeContext.from_smiles(smiles)
        if ctx is not None:
            try:
                canon = Chem.MolToSmiles(ctx.mol)
                self._generated_smiles.add(canon)
            except Exception:
                pass

    def harvest_fragments(self, smiles: str) -> None:
        """Extract BRICS fragments from a high-scoring molecule for future reassembly.

        This enables the fragment pool to dynamically evolve based on what
        the engine is actually discovering, unlocking novel scaffold discovery
        rather than relying purely on the static ``ELECTROLYTE_FRAGMENT_POOL``.

        The pool is capped at ``_MAX_HARVESTED_FRAGMENTS`` (200) to prevent
        memory bloat and combinatorial explosion over many generations.
        When the cap is reached, the oldest fragments are evicted.
        """
        ctx = MoleculeContext.from_smiles(smiles)
        if ctx is None:
            return
        try:
            frag_smiles = list(BRICS.BRICSDecompose(ctx.mol))
            for frag_smi in frag_smiles:
                if frag_smi in self._harvested_fragment_set:
                    continue
                if frag_smi in ELECTROLYTE_FRAGMENT_POOL:
                    continue
                f_ctx = MoleculeContext.from_smiles(frag_smi)
                if f_ctx is None:
                    continue
                if f_ctx.mw > 250:
                    continue
                if f_ctx.hbd > 0:
                    continue
                if self._harvested_fragments and self._fragment_too_similar(frag_smi, self._harvested_fragments, threshold=0.85):
                    continue
                self._harvested_fragments.append(frag_smi)
                self._harvested_fragment_set.add(frag_smi)
                if len(self._harvested_fragments) > _MAX_HARVESTED_FRAGMENTS:
                    oldest = self._harvested_fragments.pop(0)
                    self._harvested_fragment_set.discard(oldest)
        except Exception:
            logger.debug("Failed to harvest fragments from %s", smiles, exc_info=True)

    @staticmethod
    def _fragment_too_similar(new_smi: str, existing_smis: list[str], threshold: float = 0.85) -> bool:
        new_ctx = MoleculeContext.from_smiles(new_smi)
        if new_ctx is None:
            return False
        new_fp = new_ctx.get_ecfp4()
        for old_smi in existing_smis:
            old_ctx = MoleculeContext.from_smiles(old_smi)
            if old_ctx is None:
                continue
            old_fp = old_ctx.get_ecfp4()
            from rdkit.DataStructs import TanimotoSimilarity
            sim = TanimotoSimilarity(new_fp, old_fp)
            if sim >= threshold:
                return True
        return False

    def fragment_pool_size(self) -> int:
        """Total fragments available for BRICS reassembly (static + harvested)."""
        return len(ELECTROLYTE_FRAGMENT_POOL) + len(self._harvested_fragments)

    def _novelty_check(self, ctx: MoleculeContext, check_scaffold: bool = True) -> bool:
        """Return True if molecule is novel.

        A molecule is considered novel if:
          1. Its canonical SMILES is not an exact match for any seed,
             known electrolyte, or previously generated molecule (O(1) set lookup).
          2. Its ECFP4 Tanimoto similarity is < 0.85 against all *known
             commercial electrolytes* only (static, small list).
          3. Optionally, its Murcko scaffold is not in the seed scaffold set
             (prevents rediscovery of known structural cores).

        The Tanimoto check against generated molecules has been intentionally
        removed — exact SMILES matching (O(1)) handles intra-run dedup.
        This prevents the O(N) performance crawl that occurred when every
        generated molecule was appended to the fingerprint database.
        """
        # O(1) canonical SMILES exact match (seeds, knowns, generated)
        try:
            canon = Chem.MolToSmiles(ctx.mol)
        except Exception:
            return False
        if canon in self._seed_smiles:
            return False
        if canon in self._known_smiles:
            return False
        if canon in self._generated_smiles:
            return False

        # Murcko scaffold novelty check
        if check_scaffold and MurckoScaffold is not None and self._seed_scaffolds:
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=ctx.mol)
                if scaffold and scaffold in self._seed_scaffolds:
                    return False
            except Exception:
                pass

        # Tanimoto check against *known commercial electrolytes only*
        # (static list, never grows — no O(N) bottleneck)
        fp = ctx.get_ecfp4()
        if self._commercial_fps:
            from rdkit.DataStructs import BulkTanimotoSimilarity
            sims = BulkTanimotoSimilarity(fp, self._commercial_fps)
            if any(s >= 0.85 for s in sims):
                return False

        return True

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

    def _load_brics_fragments(self, exploration_bias: bool = False) -> list[Chem.Mol]:
        """Load BRICS-decomposed fragments from the electrolyte pool.

        Pre-decomposes each pool molecule into BRICS fragments with dummy
        atoms so they can be directly used by BRICS.BRICSBuild for reassembly.
        This is essential for generating novel scaffolds — full molecules
        passed to BRICSBuild produce no products unless they already have
        broken BRICS bonds.

        When ``exploration_bias`` is True, harvested novel fragments are
        repeated in the pool to increase their selection probability,
        biasing reassembly toward scaffold novelty.
        """
        brics_frags: list[Chem.Mol] = []
        source_smiles: list[str] = list(ELECTROLYTE_FRAGMENT_POOL)

        # When in exploration mode, oversample harvested fragments to
        # bias BRICS reassembly toward novel scaffold discovery.
        if exploration_bias and self._harvested_fragments:
            repeat = max(1, len(ELECTROLYTE_FRAGMENT_POOL) // max(len(self._harvested_fragments), 1))
            for _ in range(repeat):
                source_smiles.extend(self._harvested_fragments)

        source_smiles.extend(self._harvested_fragments)

        for smi in source_smiles:
            ctx = MoleculeContext.from_smiles(smi)
            if ctx is None:
                continue
            if ctx.mw > 250 or ctx.hbd > 0:
                continue
            try:
                frag_smiles = list(BRICS.BRICSDecompose(ctx.mol))
                for fs in frag_smiles:
                    frag_ctx = MoleculeContext.from_brics_fragment(fs)
                    if frag_ctx is not None:
                        brics_frags.append(frag_ctx.mol)
            except Exception:
                continue
        return brics_frags

    def _is_electrolyte_like(self, ctx: MoleculeContext) -> bool:
        return _is_electrolyte_like(ctx)

    @staticmethod
    def _find_complementary_pairs(fragments: list[Chem.Mol]) -> list[tuple[int, int]]:
        """Find fragment pairs with complementary BRICS dummy-atom types.

        BRICSBuild connects fragments by matching dummy-atom isotope types.
        Random pairing almost always fails; this method finds all valid
        pairs upfront so that every BRICSBuild call has a chance of success.
        """
        frag_types: list[frozenset[int]] = []
        for frag in fragments:
            types = _get_brics_types(frag)
            frag_types.append(frozenset(types))

        pairs: list[tuple[int, int]] = []
        for i in range(len(fragments)):
            if not frag_types[i]:
                continue
            for j in range(i + 1, len(fragments)):
                if frag_types[i] & frag_types[j]:
                    pairs.append((i, j))

        # Fall back to all pairs if nothing matched (e.g. all seed fragments
        # that haven't been BRICS-decomposed yet)
        if not pairs and len(fragments) >= 2:
            pairs = [(i, j) for i in range(len(fragments)) for j in range(i + 1, len(fragments))]
        return pairs

    def _collect_brics_fragments(self, ctx: MoleculeContext, force_exploration: bool) -> list[Chem.Mol]:
        """Collect BRICS fragments from seed + pool, injecting linkers if needed."""
        try:
            seed_frag_smiles = list(BRICS.BRICSDecompose(ctx.mol))
        except Exception:
            return []
        seed_frags: list[Chem.Mol] = []
        for fs in seed_frag_smiles:
            frag_ctx = MoleculeContext.from_brics_fragment(fs)
            if frag_ctx is not None:
                seed_frags.append(frag_ctx.mol)

        pool_frags = self._load_brics_fragments(exploration_bias=force_exploration)
        all_frags = seed_frags + pool_frags
        return all_frags

    def _build_from_pairs(self, all_frags: list[Chem.Mol], valid_pairs: list[tuple[int, int]]) -> list[str]:
        """Sample complementary pairs and run BRICSBuild, returning valid products."""
        generated: list[str] = []
        for _ in range(500):
            rng = np.random.default_rng(self._rng.integers(0, 2**31))
            pair_idx = rng.integers(0, len(valid_pairs))
            i, j = valid_pairs[pair_idx]
            try:
                for r_mol in BRICS.BRICSBuild([all_frags[i], all_frags[j]]):
                    if r_mol is None:
                        continue
                    try:
                        Chem.SanitizeMol(r_mol)
                        s = Chem.MolToSmiles(r_mol)
                        product_ctx = MoleculeContext(smiles=s, mol=r_mol)
                        if (
                            product_ctx.is_valid_electrolyte_mol()
                            and self._novelty_check(product_ctx)
                            and _is_electrolyte_like(product_ctx)
                            and s
                        ):
                            generated.append(s)
                    except Exception:
                        continue
            except Exception:
                continue
        return generated

    @staticmethod
    def _inject_linkers(all_frags: list[Chem.Mol]) -> None:
        """Inject universal BRICS linker fragments when the pair matrix is too sparse."""
        for linker_smi, _desc in _BRICS_LINKER_FRAGMENTS:
            linker_ctx = MoleculeContext.from_brics_fragment(linker_smi)
            if linker_ctx is not None:
                all_frags.append(linker_ctx.mol)

    def _brics_from_pool(self, ctx: MoleculeContext, force_exploration: bool = False) -> list[str]:
        """BRICS decomposition + electrolyte-fragment-guided reassembly.

        Pre-decomposes both the seed molecule and the electrolyte pool into
        BRICS fragments (with dummy atom connection points), then reassembles
        them via BRICSBuild to generate novel scaffolds.

        Uses smart pairing based on complementary BRICS dummy-atom types
        instead of blind random sampling, dramatically increasing the yield
        of valid reassembly products.

        When ``force_exploration`` is True, harvested novel fragments are
        oversampled to bias toward scaffold novelty.
        """
        generated: list[str] = []

        all_frags = self._collect_brics_fragments(ctx, force_exploration)
        if len(all_frags) < 2:
            return generated

        # Build valid complementary pairs for BRICS reassembly
        valid_pairs = self._find_complementary_pairs(all_frags)
        if not valid_pairs:
            return generated

        # If the complementary pair matrix is too sparse (< 20% of fragments
        # participate in valid pairs), inject universal linker fragments that
        # have multiple BRICS isotopes to bridge incompatible fragment types.
        n_participating = len({i for p in valid_pairs for i in p})
        if n_participating < len(all_frags) * 0.2 and force_exploration:
            self._inject_linkers(all_frags)
            valid_pairs = self._find_complementary_pairs(all_frags)
            if not valid_pairs:
                return generated

        generated = self._build_from_pairs(all_frags, valid_pairs)
        return list(set(generated))

    # ------------------------------------------------------------------
    # Public mutation API
    # ------------------------------------------------------------------

    def mutate(self, smiles: str, batch_size: int = 50, force_exploration: bool = False) -> list[str]:
        """Generate up to batch_size mutated variants of a seed molecule.

        Args:
            smiles: SMILES string of the seed molecule.
            batch_size: Maximum number of variants to return.
            force_exploration: If True, skip local SMARTS edits and rely solely
                on global BRICS scaffold-hopping to escape local optima.

        Returns:
            List of candidate SMILES strings.
        """
        ctx = MoleculeContext.from_smiles(smiles)
        if ctx is None:
            return []

        candidates: set[str] = set()

        if not force_exploration:
            smarts_results = self._apply_smarts_reactions(ctx)
            candidates.update(smarts_results)

        if not candidates or len(candidates) < batch_size:
            brics_results = self._brics_from_pool(ctx, force_exploration=force_exploration)
            candidates.update(brics_results)

        result_list = list(candidates)
        if len(result_list) > batch_size:
            indices = self._rng.choice(len(result_list), size=batch_size, replace=False)
            result_list = [result_list[i] for i in indices]

        brics_count = sum(1 for s in result_list if s not in (smarts_results if not force_exploration else set()))
        logger.info(
            "Mutation of %s: %d candidates (%d SMARTS, %d BRICS) [force_exploration=%s]",
            smiles, len(result_list), len(result_list) - brics_count,
            brics_count, force_exploration,
        )
        return result_list

    def mutate_batch(self, batch_smiles: list[str], batch_size: int = 50, force_exploration: bool = False) -> list[str]:
        """Mutate a batch of seed molecules, returning all variants.

        Args:
            batch_smiles: List of seed SMILES strings.
            batch_size: Maximum number of variants per seed.
            force_exploration: If True, skip SMARTS edits and use BRICS only
                (used to escape scaffold stagnation).

        Returns:
            List of unique candidate SMILES strings.
        """
        all_variants: list[str] = []
        for smi in batch_smiles:
            variants = self.mutate(smi, batch_size, force_exploration=force_exploration)
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
