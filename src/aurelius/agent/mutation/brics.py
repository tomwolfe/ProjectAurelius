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
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import BRICS

from aurelius.constants import COMMERCIAL_BUILDING_BLOCK_SMILES
from aurelius.types import MoleculeContext

# Maximum number of dynamically harvested fragments to keep.
# Prevents memory bloat and combinatorial explosion over many generations.
_MAX_HARVESTED_FRAGMENTS = 200

# Load commercial precursors from extended JSON file
_COMMERCIAL_PRECURSORS_PATH = Path("src/aurelius/data/commercial_precursors.json")


def _load_all_precursors():
    """Load commercial precursors from both constants and JSON file.
    
    Returns:
        tuple[Chem.Mol, ...]: Combined precursor molecules (constants + JSON)
    """
    # Start with constants
    precursors = []
    for smi in COMMERCIAL_BUILDING_BLOCK_SMILES:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            precursors.append(mol)
    
    # Load from JSON file
    if _COMMERCIAL_PRECURSORS_PATH.exists():
        import json
        with open(_COMMERCIAL_PRECURSORS_PATH, "r") as f:
            json_precursors = json.load(f)
        
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
        next_fragments = []

        # Decompose all current fragments
        for frag in current_fragments:
            try:
                decomposed = list(BRICS.BRICSDecompose(frag))
                next_fragments.extend(decomposed)
            except Exception:
                next_fragments.append(frag)

        # Check what fraction of fragments match known precursors
        matched = 0
        for frag in next_fragments:
            # frag is already a SMILES string from BRICS.BRICSDecompose
            core_smi = _strip_brics_dummies(frag)
            if core_smi is None:
                continue

            core_mol = Chem.MolFromSmiles(core_smi)
            if core_mol is None:
                continue

            # Check if this fragment matches any known precursor
            is_precursor = False
            for precursor in BB_MOLS:
                if core_mol.HasSubstructMatch(precursor) or precursor.HasSubstructMatch(core_mol):
                    is_precursor = True
                    break

            if is_precursor:
                matched += 1

        # If >80% of fragments match precursors, we're at acceptable depth
        if len(next_fragments) > 0 and (matched / len(next_fragments)) >= 0.8:
            return current_depth

        # Continue decomposing if depth < max_iterations
        current_fragments = next_fragments

    return current_depth  # Return max depth if not converged


def combined_grounding_score(mol: Chem.Mol) -> float:
    """Combined grounding score: max of BRICS coverage, functional-group coverage, and retrosynthetic depth.

    Uses the maximum of the three coverage metrics so that a molecule with a
    novel BRICS scaffold but fully commercial functional groups and direct precursors
    is not unduly penalised. Depth penalty reduces score for molecules requiring
    multi-step synthesis.

    Returns: score in [0, 1] where 1.0 = perfect synthesizability (depth=1, all commercial)
    """
    brics_cov = brics_building_block_coverage(mol)
    fg_cov = functional_group_coverage(mol)
    depth = brics_retrosynthetic_depth(mol)

    # Apply depth penalty
    depth_penalty = {
        1: 1.0,   # Direct precursor - no penalty
        2: 0.95,  # One-step synthesis - small penalty
        3: 0.85,  # Two-step synthesis - moderate penalty
        4: 0.75,  # Three-step synthesis - significant penalty
        5: 0.65,  # Four+ steps - high penalty
    }
    max_depth = max(depth_penalty.keys())
    depth_penalty_factor = depth_penalty.get(depth, depth_penalty[max_depth])

    # Combined score: max of all three, with depth penalty applied
    return max(brics_cov, fg_cov) * depth_penalty_factor


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
