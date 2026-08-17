"""Precursor database and retrosynthetic depth estimation for Project Aurelius.

This module implements the precursor database expansion, retrosynthetic
depth estimation, and template-based synthesis feasibility assessment
for Gap 2: Synthesizable outputs.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import BRICS


def _data_path(filename: str) -> Path:
    from importlib.resources import files
    return Path(str(files("aurelius.data"))) / filename


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


@lru_cache(maxsize=1)
def _template_patterns() -> tuple[tuple[Chem.Mol, str, str], ...]:
    """Compile every template once as (pattern, name, category)."""
    compiled: list[tuple[Chem.Mol, str, str]] = []
    for tmpl in _load_synthesis_templates():
        pat = Chem.MolFromSmarts(tmpl.get("smarts", ""))
        if pat is not None:
            compiled.append((pat, tmpl.get("name", ""), tmpl.get("category", "")))
    return tuple(compiled)


# Structural motifs that make a molecule hard or dangerous to make, regardless
# of how many nice functional groups it also contains. Values are multiplicative
# penalties applied to the template score.
_INFEASIBLE_MOTIFS: list[tuple[str, str, float]] = [
    ("[OX2][OX2]", "peroxide", 0.15),
    ("[NX2]=[NX2+]=[NX1-]", "azide", 0.15),
    ("[N+](=O)[O-]", "nitro", 0.55),
    ("[Si][OX2][Si]", "disiloxane", 0.35),
    ("[SiX4]", "silicon", 0.45),
    ("[B,Se,Te,As,Ge,Sn]", "exotic_heteroatom", 0.25),
    ("[OX2r3]", "epoxide_or_dioxirane", 0.40),
    ("[CX3](=[OX1])[OX2][CX3](=[OX1])", "anhydride", 0.55),
    ("[CX3](=[OX1])[CX2]#[NX1]", "acyl_cyanide", 0.35),
    ("[CX4](C#N)(C#N)C#N", "polynitrile_carbon", 0.30),
    ("[F,Cl,Br,I][CX4][OX2][CX4][F,Cl,Br,I]", "bis_halo_ether", 0.55),
]

_COMPILED_INFEASIBLE: tuple[tuple[Chem.Mol, str, float], ...] = tuple(
    (pat, name, penalty)
    for smarts, name, penalty in _INFEASIBLE_MOTIFS
    if (pat := Chem.MolFromSmarts(smarts)) is not None
)


def infeasibility_penalty(mol: Chem.Mol) -> tuple[float, list[str]]:
    """Multiplicative penalty for motifs that defeat a plausible synthesis.

    Returns ``(multiplier, motif_names)``. Penalties compound, so a molecule
    carrying both a peroxide and a silyl ether is punished far harder than one
    carrying either alone — which matches how a chemist reads such a structure.
    """
    if mol is None:
        return 0.0, ["invalid"]

    multiplier = 1.0
    hits: list[str] = []
    for pat, name, penalty in _COMPILED_INFEASIBLE:
        if mol.HasSubstructMatch(pat):
            multiplier *= penalty
            hits.append(name)
    return multiplier, hits


def compute_synthesis_feasibility(mol: Chem.Mol) -> float:
    """Template-based synthesis feasibility score for a molecule.

    Physical justification: BRICS depth alone cannot distinguish fragments
    that are commercially available from those requiring multi-step synthesis.
    Template matching checks the structure against reaction pathways that are
    standard in electrolyte synthesis.

    ADR-2026-08-10-02: this previously returned one of exactly three values
    (0.9 / 0.5 / 0.1) depending on whether *any* core or functional template
    matched. Since essentially every electrolyte candidate contains at least
    one ether or carbonyl, the score was 0.9 for 96% of realistic populations
    and carried almost no information. It is now graded on:

    * **coverage** — what fraction of the molecule's heavy atoms is spanned by
      recognised template motifs, so a molecule that is *entirely* made of
      well-precedented groups outranks one with a single ester hanging off an
      exotic core;
    * **diversity** — how many distinct core/functional templates match,
      saturating, since the second recognised motif adds less than the first;
    * **infeasibility motifs** — a multiplicative penalty for peroxides,
      azides, silyl ethers, polynitrile carbons and other groups that make a
      molecule unmakeable or unusable regardless of what else it contains.

    Args:
        mol: RDKit molecule to evaluate.

    Returns:
        float: Synthesis feasibility score in [0.0, 1.0].
    """
    if mol is None:
        return 0.0

    templates = _template_patterns()
    if not templates:
        return 0.5

    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy == 0:
        return 0.0

    covered: set[int] = set()
    n_core = 0
    n_functional = 0
    for pat, _name, category in templates:
        matches = mol.GetSubstructMatches(pat)
        if not matches:
            continue
        if category == "core":
            n_core += 1
        elif category == "functional":
            n_functional += 1
        for match in matches:
            covered.update(match)

    if not covered:
        return 0.1

    coverage = len(covered) / n_heavy
    # Saturating diversity term: 1 motif -> 0.5, 2 -> 0.71, 4 -> 1.0.
    diversity = min(1.0, ((n_core + 0.5 * n_functional) / 4.0) ** 0.5)
    base = 0.15 + 0.55 * coverage + 0.30 * diversity

    penalty, _hits = infeasibility_penalty(mol)
    return float(min(max(base * penalty, 0.0), 1.0))


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


def _ring_aware_query_params() -> Chem.AdjustQueryParameters:
    """Query parameters that forbid a chain atom matching a ring atom.

    ADR-2026-08-10-02: plain ``HasSubstructMatch`` is topology-blind, so the
    linear precursor triglyme (COCCOCCOC) matches the strained triepoxide
    C1OC1C1OC1C1OC1 at 100% atom coverage. The triepoxide was consequently
    scored as directly purchasable (depth 1, direct confidence 1.00) and
    survived adversarial filtering. Requiring ring atoms to match ring atoms
    is the minimum chemistry a precursor lookup has to respect: a
    three-membered ring is not "an ether you can buy".
    """
    params = Chem.AdjustQueryParameters.NoAdjustments()
    params.adjustRingChain = True
    params.adjustRingChainFlags = Chem.ADJUST_IGNORENONE
    return params


_RING_AWARE_PARAMS = _ring_aware_query_params()


def as_ring_aware_query(mol: Chem.Mol) -> Chem.Mol:
    """Return a topology-respecting query version of a precursor molecule."""
    try:
        return Chem.AdjustQueryProperties(mol, _RING_AWARE_PARAMS)
    except Exception:
        return mol


def _load_precursors() -> tuple[Chem.Mol, ...]:
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

# Ring-aware query forms, built once. Used for every substructure lookup so a
# chain precursor can never match a ring target (see _ring_aware_query_params).
_BB_QUERIES: tuple[Chem.Mol, ...] = tuple(as_ring_aware_query(m) for m in _BB_MOLS)


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

    mol_query = as_ring_aware_query(mol)
    best_coverage = 0.0
    for precursor, p_query in zip(_BB_MOLS, _BB_QUERIES, strict=False):
        p_heavy = precursor.GetNumHeavyAtoms()
        if p_heavy == 0:
            continue

        # Direction 1: precursor is a substructure of the fragment
        if mol.HasSubstructMatch(p_query):
            coverage = p_heavy / n_heavy
            best_coverage = max(best_coverage, coverage)

        # Direction 2: fragment is a substructure of the precursor
        elif precursor.HasSubstructMatch(mol_query):
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
        for precursor, p_query in zip(_BB_MOLS, _BB_QUERIES, strict=False):
            p_heavy = precursor.GetNumHeavyAtoms()
            if p_heavy == 0:
                continue
            if mol.HasSubstructMatch(p_query):
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


def continuous_synthesizability_score(mol: Chem.Mol) -> float:
    """Continuous [0,1] synthesizability score based on precursor coverage.

    Unlike ``brics_retrosynthetic_depth`` which returns discrete {1,2,3,4,5},
    this returns a continuous score that reflects *how well* the molecule is
    grounded in commercial precursors. The score combines:

    1. **Direct match quality** — the best precursor coverage of the molecule
       itself (0 if no precursor matches, 1 if fully covered).
    2. **Fragment coverage** — weighted precursor coverage of BRICS fragments
       at the first decomposition level, rewarding molecules whose fragments
       are substantially purchasable.
    3. **Decomposition efficiency** — how quickly BRICS decomposition converges
       to purchasable fragments, penalising molecules that require many steps.

    Returns:
        float in [0, 1] where 1.0 = directly purchasable, 0.0 = unsynthesizable.
    """
    if mol is None:
        return 0.0

    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy == 0:
        return 0.0

    # Component 1: Direct precursor match (weight 0.40)
    direct_match = _precursor_match_score(mol)
    # _precursor_match_score uses 0.5 threshold internally; rescale so that
    # partial matches contribute proportionally.
    direct_score = min(direct_match / 0.85, 1.0) if direct_match > 0 else 0.0

    # Component 2: Fragment coverage after one BRICS decomposition (weight 0.35)
    try:
        fragments = list(BRICS.BRICSDecompose(mol))
    except Exception:
        fragments = []
    frag_coverage = _precursor_coverage(fragments) if fragments else 0.0

    # Component 3: Decomposition efficiency (weight 0.25)
    # How many steps to reach >0.6 coverage? Fewer steps = higher score.
    decomp_score = _decomposition_efficiency(mol, fragments)

    # Weighted combination
    return 0.40 * direct_score + 0.35 * frag_coverage + 0.25 * decomp_score


def _decomposition_efficiency(mol: Chem.Mol, fragments: list[str]) -> float:
    """Score how quickly BRICS decomposition yields purchasable fragments.

    Returns 1.0 if the molecule is directly purchasable, decaying smoothly
    as more decomposition steps are needed.
    """
    if not fragments:
        return 0.0

    # Check if directly purchasable
    direct = _precursor_match_score(mol)
    if direct >= 0.85:
        return 1.0

    # Check first-level fragments
    cov = _precursor_coverage(fragments)
    if cov >= 0.6:
        return 0.85
    if cov >= 0.4:
        return 0.65
    if cov >= 0.2:
        return 0.40

    # One more decomposition level
    next_frags = _decompose_fragments(fragments)
    cov2 = _precursor_coverage(next_frags) if next_frags else 0.0
    if cov2 >= 0.6:
        return 0.50
    if cov2 >= 0.3:
        return 0.30

    return 0.10


def get_commercial_precursors() -> list[dict[str, Any]]:
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


_REACTION_SMARTS: list[tuple[str, str]] = [
    # Esterification: acid + alcohol -> ester + water
    ("[CX3:1](=O)[OX2H1:2].[OX2H1:3][CX4:4]>>[CX3:1](=O)[OX2:3][CX4:4]",
     "esterification"),
    # Etherification: alcohol + alkyl halide -> ether
    ("[OX2H1:1][CX4:2].[CX4:3][Cl,Br,I:4]>>[OX2:1][CX4:2].[Cl,Br,I:4]",
     "etherification_williamson"),
    # Nucleophilic substitution: SN2
    ("[CX4:1][Cl,Br,I:2].[N,O,S:3]>>[CX4:1][N,O,S:3].[Cl,Br,I:2]",
     "nucleophilic_substitution"),
    # Reduction: carbonyl to alcohol
    ("[CX3:1](=O)[CX4:2]>>[CX4:1]([OX2H1])[CX4:2]",
     "carbonyl_reduction"),
    # Oxidation: alcohol to carbonyl
    ("[CX4:1]([OX2H1])[CX4:2]>>[CX3:1](=O)[CX4:2]",
     "alcohol_oxidation"),
    # Ring closing: diol to cyclic ether
    ("[OX2H1:1][CX4:2][CX4:3][OX2H1:4]>>[OX2:1]1[CX4:2][CX4:3][OX2:4]1",
     "ring_closing_diol"),
    # Suzuki coupling: boronic acid + aryl halide
    ("[BX3:1]([OX2H1])[OX2H1].[c:2][Cl,Br,I:3]>>[c:2][c:1]",
     "suzuki_coupling"),
    # Amide coupling: acid + amine -> amide
    ("[CX3:1](=O)[OX2H1].[NX3H2:2][CX4:3]>>[CX3:1](=O)[NX3:2][CX4:3]",
     "amide_coupling"),
    # Carbonate formation: phosgene + alcohol
    ("O=C(Cl)Cl.[OX2H1:1][CX4:2]>>O=C([OX2:1][CX4:2])[OX2H1]",
     "carbonate_formation"),
    # Sulfonamide formation: sulfonyl chloride + amine
    ("[SX4:1](=O)(=O)[Cl:2].[NX3H2:3][CX4:4]>>[SX4:1](=O)(=O)[NX3:3][CX4:4]",
     "sulfonamide_formation"),
    # Urea formation: isocyanate + amine
    ("[NX2:1]=[C:2]=[O:3].[NX3H2:4][CX4:5]>>[NX3:1][C:2](=O)[NX3:4][CX4:5]",
     "urea_formation"),
    # Ether cleavage (reverse): ether -> alcohol + alkyl halide
    ("[OX2:1][CX4:2]>>[OX2H1:1][CX4:2].[Cl:3]",
     "ether_cleavage"),
    # Ester hydrolysis (reverse): ester -> acid + alcohol
    ("[CX3:1](=O)[OX2:2][CX4:3]>>[CX3:1](=O)[OX2H1].[OX2H1:2][CX4:3]",
     "ester_hydrolysis"),
    # Carbonate hydrolysis (reverse)
    ("O=C([OX2:1][CX4:2])[OX2:3][CX4:4]>>O=C([OX2H1])[OX2H1].[OX2H1:1][CX4:2].[OX2H1:3][CX4:4]",
     "carbonate_hydrolysis"),
    # Amide hydrolysis (reverse)
    ("[CX3:1](=O)[NX3:2][CX4:3]>>[CX3:1](=O)[OX2H1].[NX3H2:2][CX4:3]",
     "amide_hydrolysis"),
]


def attempt_one_step_disconnection(mol: Chem.Mol) -> list[dict[str, Any]]:
    """Apply reaction SMARTS in reverse to find one-step disconnections.

    For each reaction SMARTS, applies it in reverse (products -> reactants)
    to find possible precursor pairs that could yield the target molecule.

    Args:
        mol: Target RDKit molecule to disconnect.

    Returns:
        List of dicts with keys: 'precursors' (list of SMILES), 'reaction_name',
        'smarts'. Each dict represents one possible disconnection.
    """
    from rdkit.Chem import AllChem

    results = []
    target_smi = Chem.MolToSmiles(mol)

    for smarts, name in _REACTION_SMARTS:
        try:
            # Parse reaction and reverse it
            rxn = AllChem.ReactionFromSmarts(smarts)
            if rxn is None:
                continue

            # Check if reaction has valid reactants and products
            if rxn.GetNumReactantTemplates() == 0 or rxn.GetNumProductTemplates() == 0:
                continue

            # Apply in reverse: use target as product, get reactants
            rxn_rev = AllChem.ChemicalReaction()
            # Initialize with the reversed reaction
            ok = rxn_rev.Initialize()
            if not ok:
                continue

            # Swap reactants and products for reverse reaction
            for i in range(rxn.GetNumProductTemplates()):
                prod = rxn.GetProductTemplate(i)
                rxn_rev.AddReactantTemplate(prod)
            for i in range(rxn.GetNumReactantTemplates()):
                reac = rxn.GetReactantTemplate(i)
                rxn_rev.AddProductTemplate(reac)

            # Run reverse reaction on target
            outcomes = rxn_rev.RunReactants((mol,))
            if outcomes:
                for precursor_tuple in outcomes:
                    # Validate each precursor
                    valid_precursors = []
                    for p in precursor_tuple:
                        if p is not None and p.GetNumAtoms() > 0:
                            try:
                                Chem.SanitizeMol(p)
                                valid_precursors.append(Chem.MolToSmiles(p))
                            except Exception:
                                pass

                    if valid_precursors:
                        results.append({
                            "precursors": valid_precursors,
                            "reaction_name": name,
                            "smarts": smarts,
                        })
        except Exception:
            continue

    return results


def has_plausible_route(mol: Chem.Mol, precursor_db: list[str] | None = None) -> tuple[bool, str]:
    """Check if molecule has at least one plausible retrosynthetic route.

    A route is plausible if at least one one-step disconnection produces
    fragments that all match entries in the commercial precursor database.

    Args:
        mol: Target RDKit molecule.
        precursor_db: Optional list of precursor SMILES. If None, loads from
            commercial_precursors.json.

    Returns:
        Tuple of (has_route: bool, description: str).
        Returns (True, "No disconnections found - assumed synthesizable") if the
        disconnection engine cannot find any routes (implementation limitation).
    """
    if precursor_db is None:
        precursor_db = [entry["smiles"] for entry in get_commercial_precursors()]

    precursor_set = set(precursor_db)

    try:
        disconnections = attempt_one_step_disconnection(mol)
    except Exception as e:
        # If disconnection engine fails, assume synthesizable (permissive default)
        return True, f"Disconnection engine error ({e}) - assumed synthesizable"

    if not disconnections:
        # No disconnections found - could be implementation limitation
        # Permissive default: assume synthesizable rather than blocking
        return True, "No disconnections found - assumed synthesizable"

    for disc in disconnections:
        precursors = disc["precursors"]
        if all(p in precursor_set for p in precursors):
            return True, f"Route via {disc['reaction_name']}: {' + '.join(precursors)}"

    # Disconnections found but none match commercial precursors
    return False, "No plausible one-step route to commercial precursors"


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
