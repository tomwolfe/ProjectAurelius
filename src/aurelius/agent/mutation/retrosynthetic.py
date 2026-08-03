"""Precursor database and retrosynthetic depth estimation for Project Aurelius.

This module implements the precursor database expansion and retrosynthetic
depth estimation for Gap 2: Synthesizable outputs.
"""

import json
from functools import lru_cache
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import BRICS


COMMERCIAL_PRECURSORS_PATH = Path("src/aurelius/data/commercial_precursors.json")


def _load_precursors():
    """Load commercial precursors from JSON and pre-compile to Mol objects.
    
    Returns:
        tuple[Chem.Mol, ...]: Pre-compiled precursor molecules
    """
    if not COMMERCIAL_PRECURSORS_PATH.exists():
        raise FileNotFoundError(f"Commercial precursors file not found: {COMMERCIAL_PRECURSORS_PATH}")
    
    with open(COMMERCIAL_PRECURSORS_PATH, "r") as f:
        precursor_data = json.load(f)
    
    # Convert SMILES to Mol objects and filter invalid ones
    precursor_mols = []
    invalid_smiles = []
    
    for entry in precursor_data:
        smiles = entry["smiles"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            precursor_mols.append(mol)
        else:
            invalid_smiles.append(smiles)
    
    if invalid_smiles:
        print(f"Warning: {len(invalid_smiles)} invalid SMILES in commercial_precursors.json")
        print(f"Invalid SMILES: {invalid_smiles[:5]}...")  # Show first 5
    
    print(f"Loaded {len(precursor_mols)} valid commercial precursors from {len(precursor_data)} entries")
    return tuple(precursor_mols)


# Load precursors at module import time
try:
    _BB_MOLS = _load_precursors()
except Exception as e:
    print(f"Warning: Failed to load commercial precursors: {e}")
    _BB_MOLS = tuple()


def _strip_brics_dummies(frag_smi: str) -> str | None:
    """Strip BRICS dummy-atom labels (e.g. [1*], [2*]) from a fragment SMILES.
    
    Args:
        frag_smi: SMILES string with BRICS dummy atoms
        
    Returns:
        SMILES without BRICS dummy atoms, or None if parsing fails
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


def _decompose_fragments(frag_smiles: list[str]) -> list[str]:
    """Decompose each BRICS fragment SMILES into sub-fragment SMILES.

    Fragments that fail to parse or decompose fall back to their original SMILES.
    """
    next_fragments: list[str] = []
    for frag_smi in frag_smiles:
        try:
            frag_mol = Chem.MolFromSmiles(frag_smi)
            if frag_mol is None:
                next_fragments.append(frag_smi)
                continue
            decomposed = BRICS.BRICSDecompose(frag_mol)
            if decomposed is not None:
                for subfrag in decomposed:
                    subfrag_smi = Chem.MolToSmiles(subfrag) if subfrag else None
                    if subfrag_smi:
                        next_fragments.append(subfrag_smi)
                    else:
                        next_fragments.append(frag_smi)
            else:
                next_fragments.append(frag_smi)
        except Exception:
            next_fragments.append(frag_smi)
    return next_fragments


def _count_precursor_matches(frag_smiles: list[str]) -> int:
    """Count how many fragment cores match a known commercial precursor."""
    matched = 0
    for frag_smi in frag_smiles:
        core_smi = _strip_brics_dummies(frag_smi)
        if core_smi is None:
            continue
        core_mol = Chem.MolFromSmiles(core_smi)
        if core_mol is None:
            continue
        if _is_known_precursor(core_mol):
            matched += 1
    return matched


def _is_known_precursor(mol: Chem.Mol) -> bool:
    """Check whether *mol* matches any commercial building-block precursor."""
    for precursor in _BB_MOLS:
        if mol.HasSubstructMatch(precursor) or precursor.HasSubstructMatch(mol):
            return True
    return False


@lru_cache(maxsize=2048)
def _cached_retrosynthetic_depth(smiles: str) -> int:
    """Cached retrosynthetic depth calculation by SMILES.
    
    Args:
        smiles: SMILES string of the target molecule
        
    Returns:
        int: Retrosynthetic depth (1 for direct precursor, >1 for multi-step)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 5  # Maximum depth for invalid molecules
    
    # Get initial BRICS decomposition
    try:
        fragments = BRICS.BRICSDecompose(mol)
        if fragments is None:
            return 5
        fragments = list(fragments)
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
        next_fragments = _decompose_fragments(current_fragments)
        
        # If >80% of fragments match precursors, we're at acceptable depth
        matched = _count_precursor_matches(next_fragments)
        if len(next_fragments) > 0 and (matched / len(next_fragments)) >= 0.8:
            return current_depth
        
        # Continue decomposing if depth < max_iterations
        current_fragments = next_fragments
    
    return current_depth  # Return max depth if not converged


def brics_retrosynthetic_depth(mol: Chem.Mol) -> int:
    """Calculate BRICS retrosynthetic depth for a molecule.
    
    Args:
        mol: RDKit molecule
        
    Returns:
        int: Retrosynthetic depth (1 = direct precursor, >1 = multi-step)
    """
    smiles = Chem.MolToSmiles(mol)
    return _cached_retrosynthetic_depth(smiles)


def get_commercial_precursors() -> list[dict]:
    """Get the commercial precursors data with metadata.
    
    Returns:
        list[dict]: List of precursor dictionaries with smiles, name, and category
    """
    with open(COMMERCIAL_PRECURSORS_PATH, "r") as f:
        return json.load(f)


def get_commercial_precursor_count() -> int:
    """Get the number of commercial precursors.
    
    Returns:
        int: Number of commercial precursors
    """
    try:
        with open(COMMERCIAL_PRECURSORS_PATH, "r") as f:
            data = json.load(f)
        return len(data)
    except Exception:
        return 0


if __name__ == "__main__":
    # Test the module
    precursors = get_commercial_precursors()
    print(f"Total precursors: {len(precursors)}")
    
    # Show some examples
    print("\nFirst 5 precursors:")
    for entry in precursors[:5]:
        print(f"  {entry['name']}: {entry['smiles']} ({entry['category']})")
    
    # Test depth calculation
    test_mols = [
        ("DMC", "COC(=O)OC"),
        ("Anisole", "COc1ccccc1"),
        ("tert-butanol", "C(C)(C)(C)O"),
    ]
    
    print("\nRetrosynthetic depth examples:")
    for name, smiles in test_mols:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            depth = brics_retrosynthetic_depth(mol)
            print(f"  {name}: depth = {depth}")
