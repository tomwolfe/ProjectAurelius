"""BRICS fragmentation and reassembly helpers.

Contains:
  - BRICS linker fragments and type utilities
  - Complementary pair finding for BRICSBuild
  - Aliphatic chain anti-gaming check
  - Building-block grounding cross-referencing (BRICS + functional-group dual-mode)
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import BRICS

from aurelius.constants import COMMERCIAL_BUILDING_BLOCK_SMILES
from aurelius.types import MoleculeContext

# Maximum number of dynamically harvested fragments to keep.
# Prevents memory bloat and combinatorial explosion over many generations.
_MAX_HARVESTED_FRAGMENTS = 200


_COMMERCIAL_PRECURSORS_PATH = Path(files("aurelius.data")) / "commercial_precursors.json"


def _load_all_precursors():
    """Load commercial precursors from both constants and JSON file.

    Returns:
        tuple[Chem.Mol, ...]: Combined precursor molecules (constants + JSON)
    """
    import json

    # Start with constants
    precursors = []
    for smi in COMMERCIAL_BUILDING_BLOCK_SMILES:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            precursors.append(mol)

    # Load from JSON file
    try:
        with open(_COMMERCIAL_PRECURSORS_PATH) as f:
            json_precursors = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        json_precursors = []

    # Add valid SMILES from JSON
    for entry in json_precursors:
        smiles = entry["smiles"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            precursors.append(mol)

    # Remove duplicates while preserving order
    seen = set()
    unique_precursors = []
    for mol in precursors:
        smiles = Chem.MolToSmiles(mol)
        if smiles not in seen:
            seen.add(smiles)
            unique_precursors.append(mol)

    return tuple(unique_precursors)

# Load all precursors at module import time
BB_MOLS = _load_all_precursors()

# Keep backward compatibility
_BB_MOLS = BB_MOLS


@lru_cache(maxsize=2048)
def _cached_coverage(smiles: str) -> float:
    """Cached building block coverage by SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.5
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
        core_mol = Chem.MolFromSmiles(core_smi)
        if core_mol is None:
            continue
        for bb in BB_MOLS:
            if core_mol.HasSubstructMatch(bb) or bb.HasSubstructMatch(core_mol):
                matched += 1
                break
    return matched / len(frags)


def _strip_brics_dummies(frag_smi: str) -> str | None:
    """Strip BRICS dummy-atom labels (e.g. [1*], [2*]) from a fragment SMILES.

    Uses RDKit to remove dummy atoms (atomic num 0) properly, handling cases
    where multiple dummy atoms neighbor a single heavy atom (e.g. [1*]C([1*])=O).
    """
    frag_mol = Chem.MolFromSmiles(frag_smi)
    if frag_mol is None:
        return None

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


def brics_building_block_coverage(mol: Chem.Mol) -> float:
    """Fraction of BRICS fragments matching known commercial building blocks.

    For each BRICS fragment from the molecule, strips dummy-atom labels and
    checks substructure match against pre-compiled commercial building blocks.
    Returns 0.0 (poor) to 1.0 (excellent). Returns 0.5 if decomposition fails.
    """
    smiles = Chem.MolToSmiles(mol)
    return _cached_coverage(smiles)


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


def brics_retrosynthetic_depth(mol: Chem.Mol) -> int:
    """Calculate BRICS retrosynthetic depth for a molecule.

    Args:
        mol: RDKit molecule

    Returns:
        int: Retrosynthetic depth (1 = direct precursor, >1 = multi-step)
    """
    smiles = Chem.MolToSmiles(mol)
    return _cached_retrosynthetic_depth(smiles)


def _is_known_bb_precursor(mol: Chem.Mol) -> bool:
    """Check whether *mol* matches any commercial building-block precursor."""
    return any(mol.HasSubstructMatch(bb) or bb.HasSubstructMatch(mol) for bb in BB_MOLS)


def _decompose_brics_fragments(frag_smiles: list[str]) -> list[str]:
    """Decompose each BRICS fragment SMILES into sub-fragment SMILES."""
    next_fragments: list[str] = []
    for frag in frag_smiles:
        try:
            decomposed = list(BRICS.BRICSDecompose(frag))
            next_fragments.extend(decomposed)
        except Exception:
            next_fragments.append(frag)
    return next_fragments


def _count_brics_precursor_matches(frag_smiles: list[str]) -> int:
    """Count how many fragment cores match a known commercial precursor."""
    matched = 0
    for frag in frag_smiles:
        core_smi = _strip_brics_dummies(frag)
        if core_smi is None:
            continue
        core_mol = Chem.MolFromSmiles(core_smi)
        if core_mol is None:
            continue
        if _is_known_bb_precursor(core_mol):
            matched += 1
    return matched


def _cached_retrosynthetic_depth(smiles: str) -> int:
    """Cached retrosynthetic depth calculation by SMILES.

    Args:
        smiles: SMILES string of the target molecule

    Returns:
        int: Retrosynthetic depth (1 = direct precursor, >1 = multi-step)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 5  # Maximum depth for invalid molecules

    # Get initial BRICS decomposition
    try:
        fragments = list(BRICS.BRICSDecompose(mol))
    except Exception:
        return 5  # Maximum depth for decomposition failures

    if not fragments:
        return 5

    # Track depth
    current_depth = 0
    current_fragments = fragments
    max_iterations = 5

    while current_depth < max_iterations:
        current_depth += 1
        next_fragments = _decompose_brics_fragments(current_fragments)

        matched = _count_brics_precursor_matches(next_fragments)

        # If >80% of fragments match precursors, we're at acceptable depth
        if len(next_fragments) > 0 and (matched / len(next_fragments)) >= 0.8:
            return current_depth

        # Continue decomposing if depth < max_iterations
        current_fragments = next_fragments

    return current_depth  # Return max depth if not converged


def combined_grounding_score(mol: Chem.Mol) -> float:
    """Combined grounding score: weighted blend of BRICS coverage and template-based synthesis feasibility.

    Uses a weighted combination of BRICS building-block coverage
    and template-based synthesis feasibility:
      0.4 × BRICS_coverage + 0.6 × template_feasibility

    The template feasibility component checks the molecule's
    BRICS fragments against known reaction SMARTS templates for
    core electrolyte transformations (carbonate formation, ether
    alkylation, nitrile substitution, sulfone coupling,
    fluorination, etc.).

    Physical justification: BRICS fragment matching alone
    overestimates synthesizability for molecules whose fragments
    are not commercially available but whose synthetic routes are
    well-precedented. Template-based feasibility captures this
    by checking whether the molecule's disconnections match
    known reaction templates. The 0.4/0.6 weighting prioritizes
    template feasibility because a molecule with a novel BRICS
    scaffold but well-known synthetic routes is more likely to be
    practically synthesizable than one with commercial fragments
    but no clear route.

    ADR-2026-08-07-05: Depth-dependent penalty now follows the
    safe synthesibility form mandated by the Net Progress simplicity gate:
    ``score *= max(0.5, 1.0 - 0.1 * (depth - 1))``. This monotonically
    penalises deeper trees while keeping the floor at 0.5 so that a
    genuinely high-value novel scaffold is never fully rejected. The
    previous hardcoded {1:1.0, 2:0.95, 3:0.85, 4:0.75, 5:0.65} table is
    replaced by this closed form, halving the per-call branch count and
    reducing architectural surface for the Net Progress test.

    Returns: score in [0, 1] where 1.0 = perfect synthesizability.
    """
    brics_cov = brics_building_block_coverage(mol)
    from aurelius.agent.mutation.retrosynthetic import compute_synthesis_feasibility
    template_feas = compute_synthesis_feasibility(mol)
    depth = brics_retrosynthetic_depth(mol)

    depth_penalty_factor = max(0.5, 1.0 - 0.1 * (depth - 1))

    return (0.4 * brics_cov + 0.6 * template_feas) * depth_penalty_factor


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

# ---------------------------------------------------------------------------
# Building-Block Grounding — Commercial Precursor Cross-Referencing
# ---------------------------------------------------------------------------
# Pre-compile building block molecules at module load time.

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


def _load_brics_bond_rules() -> dict[str, frozenset[str]]:
    """Build the map of which BRICS dummy-atom types may bond to each other.

    Read directly from ``rdkit.Chem.BRICS.reactionDefs`` rather than
    hard-coded, so the rules cannot drift from the RDKit version in use.
    Types are kept as strings because BRICS uses '7a' and '7b' alongside
    the numeric labels.
    """
    partners: dict[str, set[str]] = {}
    for row in BRICS.reactionDefs:
        for type_a, type_b, _smarts in row:
            partners.setdefault(type_a, set()).add(type_b)
            partners.setdefault(type_b, set()).add(type_a)
    return {k: frozenset(v) for k, v in partners.items()}


_BRICS_BOND_RULES: dict[str, frozenset[str]] = _load_brics_bond_rules()


def get_brics_types(frag: Chem.Mol) -> set[int]:
    """Extract BRICS dummy-atom isotope types from a fragment.

    BRICS decomposition produces fragments with dummy atoms labelled by
    isotope. Note that two fragments are joinable when their types are
    *complementary* under the BRICS reaction rules, not when they are equal
    — see :func:`find_complementary_pairs`.
    """
    types: set[int] = set()
    for atom in frag.GetAtoms():
        if atom.GetAtomicNum() == 0:
            iso = atom.GetIsotope()
            if iso:
                types.add(iso)
    return types


def _fragment_bonding_profiles(
    fragments: list[Chem.Mol],
) -> tuple[list[frozenset[str]], list[frozenset[str]]]:
    """Return each fragment's own dummy types and the types it can bond to.

    Split out from :func:`find_complementary_pairs` to keep that function
    within the cyclomatic-complexity budget enforced by
    ``test_cyclomatic_complexity``.
    """
    own: list[frozenset[str]] = []
    partners: list[frozenset[str]] = []
    for frag in fragments:
        types = frozenset(str(t) for t in get_brics_types(frag))
        own.append(types)
        reachable: set[str] = set()
        for type_name in types:
            reachable |= _BRICS_BOND_RULES.get(type_name, frozenset())
        partners.append(frozenset(reachable))
    return own, partners


def find_complementary_pairs(fragments: list[Chem.Mol]) -> list[tuple[int, int]]:
    """Find fragment pairs whose BRICS dummy-atom types can actually bond.

    ADR-2026-08-07-10: this previously paired fragments that shared a dummy
    type (``frag_types[i] & frag_types[j]``), which is the wrong rule and
    silently disabled the entire BRICS pathway.

    BRICS bonds join *complementary* types, not identical ones. Reading
    ``BRICS.reactionDefs`` shows that L3 bonds to {1, 4, 13, 14, 15, 16} and
    to nothing else; only types 14 and 16 may bond to themselves. So the
    intersection rule selected pairs that BRICSBuild is definitionally unable
    to connect: decomposing dimethyl carbonate gives 40 fragments and 61
    "complementary" pairs, of which every single one produced zero products.
    Measured over eight seed molecules, the BRICS path yielded 0 candidates
    while SMARTS yielded 53 — the evolutionary algorithm was running on
    template reactions alone, and ``force_exploration=True``, which disables
    SMARTS and relies on BRICS only, produced nothing at all.

    This also explains why the grounding threshold appeared to have no
    effect: the gate it guards is on the BRICS path, which was never
    producing anything to reject.
    """
    frag_types, partners = _fragment_bonding_profiles(fragments)

    pairs = [
        (i, j)
        for i in range(len(fragments))
        for j in range(i + 1, len(fragments))
        if partners[i] & frag_types[j]
    ]

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
