"""Precursor database and retrosynthetic depth estimation for Project Aurelius.

This module implements the precursor database expansion, retrosynthetic
depth estimation, and template-based synthesis feasibility assessment
for Gap 2: Synthesizable outputs.
"""

import json
from functools import lru_cache
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import BRICS


def _data_path(filename: str) -> Path:
    from importlib.resources import files
    return Path(files("aurelius.data")) / filename


COMMERCIAL_PRECURSORS_PATH = _data_path("commercial_precursors.json")
SYNTHESIS_TEMPLATES_PATH = _data_path("synthesis_templates.json")


def _load_synthesis_templates() -> list[dict[str, str]]:
    """Load synthesis reaction templates from JSON file.

    Returns:
        List of template dicts with keys: name, smarts, description, category.
        Returns empty list if file not found or invalid.
    """
    try:
        with open(SYNTHESIS_TEMPLATES_PATH) as f:
            data = json.load(f)
        return data.get("templates", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


@lru_cache(maxsize=512)
def _load_template_smarts() -> list[tuple[str, str]]:
    """Pre-compile synthesis template SMARTS patterns.

    Returns:
        List of (smarts_string, template_name) tuples for valid templates.
    """
    templates = _load_synthesis_templates()
    compiled = []
    for tmpl in templates:
        smarts = tmpl.get("smarts", "")
        name = tmpl.get("name", "")
        try:
            pat = Chem.MolFromSmarts(smarts)
            if pat is not None:
                compiled.append((smarts, name))
        except Exception:
            continue
    return compiled


def compute_synthesis_feasibility(mol: Chem.Mol) -> float:
    """Compute template-based synthesis feasibility score for a molecule.

    Checks the molecule for the presence of functional groups that
    correspond to known feasible synthesis routes for electrolyte
    molecules. Core electrolyte functional groups (carbonate, ether,
    sulfone) indicate high synthesizability via well-precedented
    synthetic routes. Supporting functional groups (nitrile, fluoride,
    hydroxyl, etc.) indicate moderate synthesizability. Molecules with
    no recognizable electrolyte functional groups are scored low.

    Scoring:
      - Any core group matched: 0.9 (directly synthesizable)
      - Any functional group matched but no core: 0.5 (moderate)
      - No groups matched: 0.1 (low, likely Frankenstein)
      - No templates available: 0.5 (neutral default)

    Physical justification: BRICS depth alone cannot distinguish
    between fragments that are commercially available and those
    that require multi-step synthesis. Template matching provides
    a direct check against known feasible reaction pathways that
    are standard in electrolyte synthesis. A molecule whose
    structure is built from common electrolyte functional groups
    (carbonates, ethers, sulfones) is more likely to be practically
    synthesizable than one containing exotic or unusual functional
    groups.

    Args:
        mol: RDKit molecule to evaluate.

    Returns:
        float: Synthesis feasibility score in [0.0, 1.0].
    """
    if mol is None:
        return 0.0

    templates = _load_template_smarts()
    if not templates:
        return 0.5

    matched_core = _has_core_group(mol, templates)
    matched_functional = _has_functional_group(mol, templates)

    if matched_core:
        return 0.9
    if matched_functional:
        return 0.5
    return 0.1


def _has_core_group(mol: Chem.Mol, templates: list[tuple[str, str]]) -> bool:
    """Check if molecule contains any core electrolyte functional group."""
    for smarts, _name in templates:
        try:
            pat = Chem.MolFromSmarts(smarts)
            if pat is not None and mol.HasSubstructMatch(pat):
                for tmpl in _load_synthesis_templates():
                    if tmpl.get("smarts") == smarts and tmpl.get("category") == "core":
                        return True
        except Exception:
            continue
    return False


def _has_functional_group(mol: Chem.Mol, templates: list[tuple[str, str]]) -> bool:
    """Check if molecule contains any supporting functional group."""
    for smarts, _name in templates:
        try:
            pat = Chem.MolFromSmarts(smarts)
            if pat is not None and mol.HasSubstructMatch(pat):
                for tmpl in _load_synthesis_templates():
                    if tmpl.get("smarts") == smarts and tmpl.get("category") == "functional":
                        return True
        except Exception:
            continue
    return False


def _load_precursors():
    """Load commercial precursors from JSON and pre-compile to Mol objects.

    Returns:
        tuple[Chem.Mol, ...]: Pre-compiled precursor molecules
    """
    try:
        with open(COMMERCIAL_PRECURSORS_PATH) as f:
            precursor_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return tuple()

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


def _precursor_match_score(mol: Chem.Mol) -> float:
    """Compute the best precursor coverage score for a fragment.

    Returns a float in [0, 1] representing the maximum fraction of the
    fragment's heavy atoms accounted for by a single commercial precursor.

    Two directions are considered:
      1. The fragment *contains* a precursor (precursor is a substructure of
         the fragment): coverage = precursor_atoms / fragment_atoms.
      2. The fragment *is contained in* a precursor (fragment is a substructure
         of the precursor): coverage = fragment_atoms / precursor_atoms.

    A trivial single-atom match (e.g., one C or O) yields a low score, so
    fragments are only "known" when a substantial portion is purchasable.
    This prevents the constant-depth pathology where every BRICS fragment
    matched something via a one-atom substructure.

    Returns:
        float: best coverage in [0, 1]. 0.0 means no meaningful match.
    """
    if mol is None or not _BB_MOLS:
        return 0.0

    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy == 0:
        return 0.0

    best_coverage = 0.0
    for precursor in _BB_MOLS:
        p_heavy = precursor.GetNumHeavyAtoms()
        if p_heavy == 0:
            continue

        # Direction 1: precursor is a substructure of the fragment
        if mol.HasSubstructMatch(precursor):
            coverage = p_heavy / n_heavy
            best_coverage = max(best_coverage, coverage)

        # Direction 2: fragment is a substructure of the precursor
        elif precursor.HasSubstructMatch(mol):
            coverage = n_heavy / p_heavy
            best_coverage = max(best_coverage, coverage)

    return min(best_coverage, 1.0)


def _is_known_precursor(mol: Chem.Mol) -> bool:
    """Check whether *mol* matches any commercial building-block precursor.

    A fragment is "known" only when a commercial precursor accounts for at
    least 50% of its heavy atoms. This is stricter than pure substructure
    matching (where one C or O would count) and prevents every BRICS fragment
    from being trivially grounded.
    """
    return _precursor_match_score(mol) >= 0.5


def _fragment_rarity_penalty(mol: Chem.Mol) -> float:
    """Penalize fragments containing exotic/rare functional groups.

    Returns a multiplier in [0.3, 1.0]:
      - 1.0 for fragments with only common atoms (C, H, O, N, S, F, P)
      - 0.3 for fragments containing exotic atoms (Se, Te, Si, B, etc.)
      - 0.6 for fragments with halogens beyond F (Cl, Br, I)

    Physical justification: even if an exotic fragment is technically
    "purchasable", the precursor is expensive, unstable, or requires
    air-free handling that makes synthesis impractical for a discovery
    campaign.
    """
    if mol is None:
        return 0.3

    exotic_atoms = {32, 34, 50, 52, 33, 14, 5, 31, 35, 53}  # Ge, Ge, Sn, Te, As, Si, B, P, Br, I
    heavy_halogens = {17, 35, 53}  # Cl, Br, I

    atoms = {a.GetAtomicNum() for a in mol.GetAtoms()}
    if atoms & exotic_atoms:
        return 0.3
    if atoms & heavy_halogens:
        return 0.6
    return 1.0


def _precursor_coverage(frag_smiles: list[str]) -> float:
    """Compute weighted precursor coverage for a list of BRICS fragments.

    For each fragment, computes precursor_match_score × rarity_penalty,
    weighted by fragment size (larger fragments matter more). Returns the
    weighted average coverage in [0, 1].

    This replaces the binary _count_precursor_matches with a continuous
    signal that differentiates "most fragments are purchasable" from
    "fragments are only trivially grounded".
    """
    if not frag_smiles:
        return 0.0

    total_weight = 0.0
    weighted_coverage = 0.0

    for frag_smi in frag_smiles:
        core_smi = _strip_brics_dummies(frag_smi)
        if core_smi is None:
            continue
        core_mol = Chem.MolFromSmiles(core_smi)
        if core_mol is None:
            continue

        n_heavy = core_mol.GetNumHeavyAtoms()
        if n_heavy == 0:
            continue

        match_score = _precursor_match_score(core_mol)
        rarity = _fragment_rarity_penalty(core_mol)
        coverage = match_score * rarity

        weighted_coverage += coverage * n_heavy
        total_weight += n_heavy

    if total_weight == 0:
        return 0.0
    return weighted_coverage / total_weight


@lru_cache(maxsize=2048)
def _cached_retrosynthetic_depth(smiles: str) -> int:
    """Cached retrosynthetic depth calculation by SMILES.

    Uses continuous precursor coverage (weighted by fragment size and rarity)
    instead of binary fragment counting. Depth semantics:

      depth 1 = the molecule itself is directly purchasable (exact or
                near-exact match to a commercial precursor)
      depth 2 = one BRICS disconnection yields purchasable fragments
      depth 3+ = multiple disconnections required
      depth 5 = exotic/unsynthesizable (fails all decomposition levels)

    Args:
        smiles: SMILES string of the target molecule

    Returns:
        int: Retrosynthetic depth (1 = direct precursor, >1 for multi-step)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 5  # Maximum depth for invalid molecules

    # Depth 1: check if the molecule itself is directly purchasable.
    # Only count direction 1 (a commercial precursor is a substructure of
    # the molecule, covering ≥85% of its atoms). Direction 2 (molecule is
    # a substructure of a larger precursor) does NOT count — being a piece
    # of something purchasable does not make the fragment itself purchasable.
    # This prevents benzene (substructure of biphenyl) from scoring depth 1.
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy > 0:
        for precursor in _BB_MOLS:
            p_heavy = precursor.GetNumHeavyAtoms()
            if p_heavy == 0:
                continue
            if mol.HasSubstructMatch(precursor):
                coverage = p_heavy / n_heavy
                if coverage >= 0.85:
                    return 1

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

    # Depth 2+: check if BRICS fragments are purchasable
    current_depth = 1
    current_fragments = fragments
    max_iterations = 5

    while current_depth < max_iterations:
        current_depth += 1
        next_fragments = _decompose_fragments(current_fragments)

        # Continuous coverage: weighted by fragment size × rarity penalty
        coverage = _precursor_coverage(next_fragments)
        if coverage >= 0.6:
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
    with open(COMMERCIAL_PRECURSORS_PATH) as f:
        return json.load(f)


def get_commercial_precursor_count() -> int:
    """Get the number of commercial precursors.

    Returns:
        int: Number of commercial precursors
    """
    try:
        with open(COMMERCIAL_PRECURSORS_PATH) as f:
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
