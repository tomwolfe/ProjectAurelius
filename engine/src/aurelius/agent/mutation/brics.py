"""BRICS fragmentation and reassembly helpers.

Contains:
  - BRICS linker fragments and type utilities
  - Complementary pair finding for BRICSBuild
  - Aliphatic chain anti-gaming check
  - Building-block grounding cross-referencing (BRICS + functional-group dual-mode)
"""

from __future__ import annotations

import re
from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import BRICS

from aurelius.constants import COMMERCIAL_BUILDING_BLOCK_SMILES
from aurelius.types import MoleculeContext

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
    ("[1*]COC(=O)O[2*]", "cyclic_carbonate_linker"),
    ("[1*]CCOCC[2*]", "thf_linker"),
    ("[1*]CS(=O)(=O)CC[2*]", "sulfolane_linker"),
    ("[1*]CO[2*]", "epoxide_linker"),
]


def get_brics_types(frag: Chem.Mol) -> set[int]:
    """Extract BRICS dummy-atom isotope types from a fragment.

    BRICS decomposition produces fragments with dummy atoms labelled by
    isotope (1-5).  For two fragments to be joined by BRICSBuild they
    must share at least one common dummy-atom type.
    """
    types: set[int] = set()
    for atom in frag.GetAtoms():
        if atom.GetAtomicNum() == 0:
            iso = atom.GetIsotope()
            if iso:
                types.add(iso)
    return types


def find_complementary_pairs(fragments: list[Chem.Mol]) -> list[tuple[int, int]]:
    """Find fragment pairs with complementary BRICS dummy-atom types.

    BRICSBuild connects fragments by matching dummy-atom isotope types.
    Random pairing almost always fails; this method finds all valid
    pairs upfront so that every BRICSBuild call has a chance of success.
    """
    frag_types: list[frozenset[int]] = []
    for frag in fragments:
        types = get_brics_types(frag)
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


def inject_linkers(all_frags: list[Chem.Mol]) -> None:
    """Inject universal BRICS linker fragments when the pair matrix is too sparse."""
    for linker_smi, _desc in _BRICS_LINKER_FRAGMENTS:
        linker_ctx = MoleculeContext.from_brics_fragment(linker_smi)
        if linker_ctx is not None:
            all_frags.append(linker_ctx.mol)


def has_excessive_aliphatic_chain(mol: Chem.Mol, max_chain: int = 12) -> bool:
    """Check if molecule has an aliphatic chain longer than max_chain.

    BRICS reassembly can produce "Frankenstein" molecules with
    unrealistically long continuous aliphatic chains that are
    synthetically inaccessible and lack the heteroatom density
    needed for electrolyte function.
    """
    visited: set[int] = set()
    longest = [0]

    def _dfs(idx: int, length: int) -> None:
        visited.add(idx)
        longest[0] = max(longest[0], length)
        atom = mol.GetAtomWithIdx(idx)
        for nb in atom.GetNeighbors():
            n_idx = nb.GetIdx()
            if n_idx not in visited:
                n_atom = mol.GetAtomWithIdx(n_idx)
                if n_atom.GetAtomicNum() == 6 and n_atom.GetHybridization() == Chem.HybridizationType.SP3:
                    _dfs(n_idx, length + 1)
        visited.discard(idx)

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6 and atom.GetHybridization() == Chem.HybridizationType.SP3 and atom.GetIdx() not in visited:
            _dfs(atom.GetIdx(), 1)

    return longest[0] > max_chain


# ---------------------------------------------------------------------------
# Building-Block Grounding — Commercial Precursor Cross-Referencing
# ---------------------------------------------------------------------------
# Pre-compile building block molecules at module load time.

_BB_MOLS: tuple[Chem.Mol, ...] = ()
_bb_temp: list[Chem.Mol] = []
for _smi in COMMERCIAL_BUILDING_BLOCK_SMILES:
    _ctx = MoleculeContext.from_smiles(_smi)
    if _ctx is not None:
        _bb_temp.append(_ctx.mol)
_BB_MOLS = tuple(_bb_temp)
_BB_DUMMY_RE: re.Pattern[str] = re.compile(r"\[\d*\*\]")


def _strip_brics_dummies(frag_smi: str) -> str | None:
    """Strip BRICS dummy-atom labels (e.g. [1*], [2*]) from a fragment SMILES.

    Uses RDKit to remove dummy atoms (atomic num 0) properly, handling cases
    where multiple dummy atoms neighbor a single heavy atom (e.g. [1*]C([1*])=O).
    """
    frag_ctx = MoleculeContext.from_brics_fragment(frag_smi)
    if frag_ctx is None:
        return None
    frag_mol = frag_ctx.mol

    rw = Chem.RWMol(frag_mol)
    dummy_ids = sorted(
        (a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0),
        reverse=True,
    )
    for idx in dummy_ids:
        rw.RemoveAtom(idx)

    try:
        rw.UpdatePropertyCache()
        Chem.SanitizeMol(rw)
    except Exception:
        return None
    return Chem.MolToSmiles(rw)


@lru_cache(maxsize=2048)
def _cached_coverage(smiles: str) -> float:
    """Cached building block coverage by SMILES string."""
    ctx = MoleculeContext.from_smiles(smiles)
    if ctx is None:
        return 0.5
    mol = ctx.mol
    try:
        frags = list(BRICS.BRICSDecompose(mol))
    except Exception:
        return 0.5
    if not frags:
        return 0.5
    matched = 0
    for fs in frags:
        core_smi = _strip_brics_dummies(fs)
        if core_smi is None:
            continue
        core_ctx = MoleculeContext.from_smiles(core_smi)
        if core_ctx is None:
            continue
        core_mol = core_ctx.mol
        for bb in _BB_MOLS:
            if core_mol.HasSubstructMatch(bb) or bb.HasSubstructMatch(core_mol):
                matched += 1
                break
    return matched / len(frags)


def brics_building_block_coverage(mol: Chem.Mol) -> float:
    """Fraction of BRICS fragments matching known commercial building blocks.

    For each BRICS fragment from the molecule, strips dummy-atom labels and
    checks substructure match against pre-compiled commercial building blocks.
    Returns 0.0 (poor) to 1.0 (excellent). Returns 0.5 if decomposition fails.
    """
    smiles = Chem.MolToSmiles(mol)
    return _cached_coverage(smiles)


# ---------------------------------------------------------------------------
# Functional-Group Grounding — 1-Step Synthetic Feasibility
# ---------------------------------------------------------------------------
# Physical justification: A molecule with a novel BRICS scaffold may still be
# synthesizable if ALL of its functional groups appear in commercial building
# blocks. This relaxes pure BRICS-fragment matching, which penalises molecules
# whose retrosynthetic cuts happen to produce non-commercial fragments even
# though each functional group is commercially available. The dual-mode
# grounding (BRICS + functional-group) increases novel scaffold yield without
# sacrificing synthetic feasibility.

_GROUNDING_FG_PATTERNS: list[tuple[Chem.Mol, str]] = [
    (Chem.MolFromSmarts("[CX3](=O)[OX2H0]"), "ester"),
    (Chem.MolFromSmarts("[CX3](=O)[OH]"), "carboxylic_acid"),
    (Chem.MolFromSmarts("[CX3](=O)[NX3]"), "amide"),
    (Chem.MolFromSmarts("[CX3](=O)[CX3]"), "ketone"),
    (Chem.MolFromSmarts("[CH](=O)"), "aldehyde"),
    (Chem.MolFromSmarts("O=C([OX2])[OX2]"), "carbonate"),
    (Chem.MolFromSmarts("[OD2]([CX4])[CX4]"), "ether"),
    (Chem.MolFromSmarts("[OH][CX4]"), "alcohol"),
    (Chem.MolFromSmarts("[C]#[N]"), "nitrile"),
    (Chem.MolFromSmarts("S(=O)(=O)"), "sulfone"),
    (Chem.MolFromSmarts("[PX4](=O)([OX2])([OX2])[OX2]"), "phosphate"),
    (Chem.MolFromSmarts("[C](F)(F)F"), "trifluoromethyl"),
    (Chem.MolFromSmarts("[F]"), "fluorine"),
]

_PRE_COMPUTED_FG_IN_BB: dict[str, bool] = {}
for _fg_pat, _fg_name in _GROUNDING_FG_PATTERNS:
    if _fg_pat is not None:
        for _bb in _BB_MOLS:
            if _bb.HasSubstructMatch(_fg_pat):
                _PRE_COMPUTED_FG_IN_BB[_fg_name] = True
                break
        if _fg_name not in _PRE_COMPUTED_FG_IN_BB:
            _PRE_COMPUTED_FG_IN_BB[_fg_name] = False


def functional_group_coverage(mol: Chem.Mol) -> float:
    """Fraction of functional groups in the molecule found in commercial building blocks.

    For each functional group pattern present in the molecule, checks whether
    that pattern also exists in at least one commercial building block SMILES.
    Returns 0.0 (no functional groups commercial) to 1.0 (all commercial).

    Physical justification: If a molecule's constituent functional groups are
    all commercially available precursors, the molecule is likely synthesizable
    via functionalisation of those precursors, even if the BRICS scaffold is
    novel. This is a weaker condition than BRICS-fragment matching but more
    permissive for scaffold-hopping.
    """
    present = 0
    commercial = 0
    for pat, name in _GROUNDING_FG_PATTERNS:
        if pat is None:
            continue
        n_matches = len(mol.GetSubstructMatches(pat))
        if n_matches > 0:
            present += 1
            if _PRE_COMPUTED_FG_IN_BB.get(name, False):
                commercial += 1
    if present == 0:
        return 0.5
    return commercial / present


# Maximum BRICS disconnection depth before a linear penalty kicks in.
# Physical basis: Each BRICS disconnection corresponds to a synthetic step.
# Beyond 2 steps, the synthetic cost grows linearly, making the molecule
# uneconomical for kilogram-scale manufacturing. This prevents the EA
# from favoring molecules requiring >2 synthetic steps from commercial
# precursors, which would be impractical for bulk electrolyte synthesis.
_MAX_BRICS_DEPTH: int = 2
_BRICS_DEPTH_PENALTY_PER_STEP: float = 0.1

# Minimum combined grounding score for BRICS products.
# Physical justification: At least 60% of the molecule's fragments or
# functional groups must map to commercial building blocks. This ensures
# novel-scaffold candidates remain synthesizable from catalog precursors
# while not strangling genuine scaffold hopping. Too high a threshold (0.7+)
# collapses the proposal space back to seed-like molecules.
MIN_GROUNDING_SCORE: float = 0.6


def _compute_brics_depth(mol: Chem.Mol, max_iter: int = 5) -> int:
    """Compute the maximum BRICS disconnection depth to reach commercial building blocks.

    Recursively decomposes the molecule using BRICS. At each level, fragments
    are checked against the commercial building block library. If any fragment
    at a given depth does NOT match a commercial building block, it is further
    decomposed (depth + 1). Returns the maximum depth needed for all fragments
    to resolve to commercial building blocks.

    A depth of 0 means the molecule itself is a commercial building block.
    """
    def _recurse(frag_smi: str, current_depth: int) -> int:
        if current_depth >= max_iter:
            return current_depth
        core_smi = _strip_brics_dummies(frag_smi)
        if core_smi is None:
            return current_depth
        core_ctx = MoleculeContext.from_smiles(core_smi)
        if core_ctx is None:
            return current_depth
        n_core = core_ctx.mol.GetNumHeavyAtoms()
        for bb in _BB_MOLS:
            if bb.HasSubstructMatch(core_ctx.mol):
                n_bb = bb.GetNumHeavyAtoms()
                if abs(n_core - n_bb) <= 2:
                    return current_depth
        try:
            sub_frags = list(BRICS.BRICSDecompose(core_ctx.mol))
        except Exception:
            return current_depth
        if not sub_frags:
            return current_depth
        max_d = current_depth
        for sf in sub_frags:
            d = _recurse(sf, current_depth + 1)
            if d > max_d:
                max_d = d
        return max_d

    for bb in _BB_MOLS:
        if bb.HasSubstructMatch(mol):
            return 0

    try:
        frags = list(BRICS.BRICSDecompose(mol))
    except Exception:
        return 0
    if not frags:
        return 0
    max_depth = 0
    for f in frags:
        d = _recurse(f, 1)
        if d > max_depth:
            max_depth = d
    return max_depth


def _estimate_synthetic_depth(mol: Chem.Mol) -> int:
    """Estimate the retrosynthetic depth of a molecule using recursive BRICS decomposition.

    Performs BRICS decomposition recursively until all fragments match commercial
    building blocks. Returns the maximum recursion depth required for all fragments
    to resolve to commercial building blocks.

    Depth 0 means the molecule itself is a commercial building block.
    """
    return _compute_brics_depth(mol)


def combined_grounding_score(mol: Chem.Mol) -> float:
    """Combined grounding score: max of BRICS coverage and functional-group coverage,
    with a linear penalty for excessive BRICS disconnection depth, adjusted by
    retrosynthetic reaction rule feasibility.

    Uses the maximum of the two coverage metrics so that a molecule with a
    novel BRICS scaffold but fully commercial functional groups is not unduly
    penalised. This is the minimal relaxation needed to enable scaffold hopping
    while maintaining synthetic feasibility.

    If the BRICS disconnection depth exceeds _MAX_BRICS_DEPTH, a multiplicative
    penalty of (1 - _BRICS_DEPTH_PENALTY_PER_STEP)^(depth - _MAX_BRICS_DEPTH)
    is applied to the base score. This ensures that economically viable
    molecules (2 or fewer synthetic steps from commercial precursors) are
    preferred over deeper retrosynthetic paths (e.g., depth 3 -> 0.9x, depth
    4 -> 0.81x).

    Additionally, a retrosynthetic reaction rule check is performed. If none of
    the molecule's bond-forming motifs match a known high-yield reaction SMARTS
    (weight >= 0.8), a mild penalty of 0.1x is applied. This prevents the EA
    from proposing molecules whose fragments are individually commercial but
    cannot be realistically coupled at scale.
    """
    brics_cov = brics_building_block_coverage(mol)
    fg_cov = functional_group_coverage(mol)
    base = max(brics_cov, fg_cov)

    depth = _estimate_synthetic_depth(mol)
    if depth > _MAX_BRICS_DEPTH:
        base *= (1.0 - _BRICS_DEPTH_PENALTY_PER_STEP) ** (depth - _MAX_BRICS_DEPTH)

    # Retrosynthetic reaction rule feasibility check
    from aurelius.agent.mutation.reaction_rules import check_retrosynthetic_feasibility
    rx_score = check_retrosynthetic_feasibility(mol)
    if rx_score < 0.8:
        base *= 0.9

    return base
