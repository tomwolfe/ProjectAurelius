"""BRICS fragmentation and reassembly helpers.

Contains:
  - BRICS linker fragments and type utilities
  - Complementary pair finding for BRICSBuild
  - Aliphatic chain anti-gaming check
"""

from __future__ import annotations

from rdkit import Chem

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
