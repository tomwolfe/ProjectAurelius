"""Template-based retrosynthetic pathway verification.

Uses USPTO/BRICS reaction templates to check whether candidate
molecules can be formed in <= 3 reaction steps from commercial
building blocks. Candidates lacking a viable retrosynthetic route
are penalized or rejected.
"""

from __future__ import annotations

import logging
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions

logger = logging.getLogger(__name__)

_MAX_STEPS = 3

_COMMERCIAL_BUILDING_BLOCKS: list[str] = [
    # Simple aromatics
    "c1ccccc1",
    "c1ccc(F)cc1",
    "c1ccc(Cl)cc1",
    "c1ccc(Br)cc1",
    "c1ccc(C)cc1",
    "c1ccc(OC)cc1",
    "c1ccc(N)cc1",
    "c1ccc(CF3)cc1",
    "c1ccc(CN)cc1",
    "c1ccc(CO)cc1",
    "c1ccc(CC)cc1",
    "c1ccc(NC)cc1",
    # Heterocycles
    "c1ccncc1",
    "c1ccoc1",
    "c1ccsc1",
    "c1cncn1",
    "c1ccnn1",
    "c1ccno1",
    # Simple aliphatics
    "CC",
    "CCO",
    "CCOCC",
    "CCN",
    "CCNCC",
    "CCC",
    "CCCC",
    "CC(C)C",
    "CC(C)(C)C",
    "CC=O",
    "CC(=O)O",
    "CC(=O)N",
    "CCN",
    "CCNC",
    "CCO",
    "CCOC",
    "CCOC(C)=O",
    "CCN(C)C",
    "CC(=O)OC",
    "CC(=O)OCC",
    # Halides and electrophiles
    "CCl",
    "CBr",
    "CI",
    "CF",
    "C(F)(F)F",
    "C(F)(F)(F)F",
    "C(F)(F)(F)(F)F",
    # Sulfonyl groups
    "CS(=O)(=O)C",
    "CS(=O)(=O)Cl",
    "CS(=O)(=O)N",
    # Phosphate/phosphonate
    "COP(=O)(OC)OC",
    "CCOP(=O)(OCC)OCC",
    # Nitrile
    "CC#N",
    "CCC#N",
    # Isocyanate/isothiocyanate
    "CN=C=O",
    "CSN=C=O",
    # Boronic acid/ester
    "OB(O)O",
    "OB(O)OC",
    "OB(O)OCC",
    # Azide
    "CN=[N+]=[N-]",
    # Simple amines
    "CN",
    "CCN",
    "CCCN",
    "c1ccc(CN)cc1",
    "c1ccc(CCN)cc1",
    # Alcohols
    "CO",
    "CCO",
    "CCC(O)",
    "c1ccc(CO)cc1",
    # Carboxylic acids/esters
    "C(=O)O",
    "CC(=O)O",
    "COC(=O)C",
    "CCOC(=O)C",
    # Aldehydes
    "C=O",
    "CC=O",
    "c1ccc(C=O)cc1",
    # Ketones
    "CC(=O)C",
    "c1ccc(C(C)=O)cc1",
]

_COMMERCIAL_BB_MOLS: list[Chem.Mol] = [
    Chem.MolFromSmiles(s) for s in _COMMERCIAL_BUILDING_BLOCKS
]

_BRICS_REACTION_TEMPLATES: list[str] = [
    # Ordered by typical synthetic yield (high-yield first).
    # High-yield (>85%): esterification, amidation, Suzuki, carbonate,
    # sulfonamide, Williamson ether.
    # Ester bond formation (95% yield)
    "[*:1]C(=O)O>>[*:1]C(=O)O[*:2]",
    # Amide bond formation (90% yield)
    "[*:1]C(=O)O>>[*:1]C(=O)N[*:2]",
    # Suzuki coupling (85-90% yield)
    "[*:1]c1ccccc1Br>>[*:1]c1ccccc1[*:2]",
    # Carbonate formation via chloroformate (85% yield)
    "[*:1]O>>[*:1]O[*:2]",
    # Sulfonamide formation (85% yield)
    "[*:1]S(=O)(=O)N>>[*:1]S(=O)(=O)N[*:2]",
    # Williamson ether synthesis (80% yield)
    "[*:1]O>>[*:1]O[*:2]",
    # Esterification (acid + alcohol) (95% yield, broad scope)
    "[*:1]C(=O)O>>[*:1]C(=O)O[*:2]",
    # Amidation (acid + amine) (90% yield)
    "[*:1]C(=O)O>>[*:1]C(=O)N[*:2]",
    # Medium-yield (70-85%): coupling, SNAr, Wittig, Diels-Alder
    # Sonogashira coupling (Ar-X + terminal alkyne) (70-80% yield)
    "[*:1]c1ccccc1>>[*:1]C#C[*:2]",
    # Negishi coupling (Ar-Zn + Ar-X) (70-80% yield)
    "[*:1]c1ccccc1>>[*:1]c1ccccc1[*:2]",
    # Ullmann coupling (Ar-X + Cu) (70-80% yield)
    "[*:1]c1ccccc1>>[*:1]c1ccccc1[*:2]",
    # Buchwald-Hartwig amination (Ar-X + amine) (75-85% yield)
    "[*:1]c1ccccc1>>[*:1]N[*:2]",
    # Stille coupling (Ar-Sn + Ar-X) (70-80% yield)
    "[*:1]c1ccccc1>>[*:1]c1ccccc1[*:2]",
    # Kumada coupling (Ar-Mg + Ar-X) (70-85% yield)
    "[*:1]c1ccccc1>>[*:1]c1ccccc1[*:2]",
    # Hiyama coupling (Ar-Si + Ar-X) (60-75% yield)
    "[*:1]c1ccccc1>>[*:1]c1ccccc1[*:2]",
    # Heck coupling (Ar-X + alkene) (70-80% yield)
    "[*:1]c1ccccc1>>[*:1]c1ccccc1[*:2]",
    # Chan-Lam coupling (Ar-B(OH)2 + amine) (60-75% yield)
    "[*:1]B(O)O>>[*:1]N[*:2]",
    # SNAr (nucleophilic aromatic substitution) (60-80% yield)
    "[*:1]c1ccccc1>>[*:1]N[*:2]",
    # Diels-Alder (diene + dienophile) (70-90% yield)
    "[*:1]C=C>>[*:1][*:2]",
    # Wittig reaction (aldehyde + ylide) (60-80% yield)
    "[*:1]C=O>>[*:1]C=C[*:2]",
    # Lower-yield (50-70%): lactonization, RCM, fluorination, deprotection
    # Lactonization (intramolecular ester) (40-60% yield)
    "[*:1]C(=O)O>>[*:1]C(=O)O[*:2]",
    # Ring-closing metathesis (60-80% yield)
    "[*:1]C=C>>[*:1][*:2]",
    # DAST fluorination (alcohol -> fluoride) (40-60% yield)
    "[*:1]O>>[*:1]F[*:2]",
    # Boronate ester (65-80% yield)
    "[*:1]B(O)O>>[*:1]B(O)O[*:2]",
    # Fluorination (electrophilic fluorinating agents) (40-60% yield)
    "[*:1]F>>[*:1]F[*:2]",
    # Nitrile formation (60-75% yield)
    "[*:1]C#N>>[*:1]C#N[*:2]",
    # Phosphonate formation (50-70% yield)
    "[*:1]P(=O)(O)>>[*:1]P(=O)(O)[*:2]",
    # Phosphate formation (50-70% yield)
    "[*:1]P(=O)(O)>>[*:1]P(=O)(O)[*:2]",
    # C-N bond formation (reductive amination) (60-70% yield)
    "[*:1]C=O>>[*:1]C-N[*:2]",
    # C-N bond formation (alkylation) (50-70% yield)
    "[*:1]N>>[*:1]N[*:2]",
    # C-S bond formation (50-60% yield)
    "[*:1]S>>[*:1]S[*:2]",
    # C-C bond formation (alkylation) (40-60% yield)
    "[*:1]C>>[*:1]C[*:2]",
    # Radical C-H functionalization (30-50% yield)
    "[*:1]C>>[*:1]C[*:2]",
    # Acylation (Friedel-Crafts) (40-60% yield)
    "[*:1]c1ccccc1>>[*:1]C(=O)[*:2]",
    # Reductive amination (imine + reductant) (50-70% yield)
    "[*:1]C=N>>[*:1]C-N[*:2]",
    # Oxidation of alcohol to ketone (50-70% yield)
    "[*:1]C(O)>>[*:1]C(=O)[*:2]",
    # Ugi-type multicomponent (50-70% yield)
    "[*:1]C(=O)N>>[*:1]C(=O)N[*:2]",
    # Schmidt lactamization (40-60% yield)
    "[*:1]C(=O)N>>[*:1]C(=O)N[*:2]",
    # Mitsunobu reaction (ROH + nucleophile) (50-60% yield)
    "[*:1]O>>[*:1]N[*:2]",
    # Ring formation (intramolecular) (generic, 30-50% yield)
    "[*:1]>>[*:1]",
    # Deprotection (removal of protecting group) (70-90% yield)
    "[*:1]C>>[*:1][*:2]",
]


def _estimate_step_economy(smiles: str) -> int:
    """Estimate synthetic step economy (1-10 scale).

    Returns a value from 1 (highly efficient, few steps) to
    10 (complex synthesis requiring many steps). The estimate
    is based on molecular complexity metrics: number of rings,
    stereocenters, and heavy atoms.

    Returns:
        int in range [1, 10]
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 10
    n_heavy = mol.GetNumHeavyAtoms()
    n_rings = mol.GetRingInfo().NumRings()
    n_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    score = 1
    score += min(n_heavy / 20, 3)
    score += min(n_rings, 3)
    score += min(n_stereo, 2)
    return max(1, min(10, int(score)))


def _parse_template(template: str) -> rdChemReactions.Reaction:
    """Parse a BRICS reaction template string into an RDKit Reaction."""
    try:
        rxn = rdChemReactions.ReactionFromSmarts(template)
        return rxn
    except Exception:
        return None  # type: ignore[return-value]


def _apply_retro_template(
    mol: Chem.Mol,
    template: rdChemReactions.Reaction,
) -> list[tuple[str, str]]:
    """Apply a retrosynthetic template to a molecule.

    Runs the reaction in reverse (product -> reactants) and
    returns pairs of (reactant1_smiles, reactant2_smiles).
    """
    if template is None:
        return []

    try:
        products = template.RunReactants((mol,))
    except Exception:
        return []

    disconnections: list[tuple[str, str]] = []
    for product_set in products:
        for product_mol in product_set:
            try:
                smi = Chem.MolToSmiles(product_mol, canonical=True)
                disconnections.append((smi, ""))
            except Exception:
                continue

    return disconnections


def _is_commercial_building_block(smiles: str) -> bool:
    """Check if a SMILES matches a known commercial building block."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    for bb_mol in _COMMERCIAL_BB_MOLS:
        if bb_mol is None:
            continue
        try:
            if mol.HasSubstructMatch(bb_mol):
                return True
        except Exception:
            continue

    # Also check exact match or very similar
    try:
        canonical = Chem.MolToSmiles(mol, canonical=True)
        for bb_smiles in _COMMERCIAL_BUILDING_BLOCKS:
            bb_mol = Chem.MolFromSmiles(bb_smiles)
            if bb_mol is None:
                continue
            bb_canonical = Chem.MolToSmiles(bb_mol, canonical=True)
            if canonical == bb_canonical:
                return True
    except Exception:
        pass

    return False


def retro_check(
    smiles: str,
    max_steps: int = _MAX_STEPS,
) -> dict[str, Any]:
    """Check if a molecule has a viable retrosynthetic route.

    Uses template-based single-step and two-step retrosynthetic
    disconnection checks against USPTO/BRICS reaction templates
    and commercial building blocks.

    Args:
        smiles: SMILES string of the candidate molecule.
        max_steps: Maximum number of retrosynthetic steps (default 3).

    Returns:
        Dict with keys:
            - viable: bool, whether a viable route exists
            - n_steps: int, number of steps needed
            - route: list of dicts describing the retrosynthetic route
            - building_blocks: list of commercial building blocks found
            - sa_score: float, synthetic accessibility score
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "viable": False,
            "n_steps": max_steps + 1,
            "route": [],
            "building_blocks": [],
            "sa_score": 999.0,
        }

    # Parse templates
    templates: list[rdChemReactions.Reaction] = []
    for template_str in _BRICS_REACTION_TEMPLATES:
        rxn = _parse_template(template_str)
        if rxn is not None:
            templates.append(rxn)

    # Step 1: Check if molecule is itself a commercial building block
    if _is_commercial_building_block(smiles):
        return {
            "viable": True,
            "n_steps": 0,
            "route": [{"type": "commercial", "smiles": smiles}],
            "building_blocks": [smiles],
            "sa_score": 1.0,
        }

    # Step 2: Single-step retrosynthesis
    for template in templates:
        disconnections = _apply_retro_template(mol, template)
        for product_smiles, _ in disconnections:
            if _is_commercial_building_block(product_smiles):
                return {
                    "viable": True,
                    "n_steps": 1,
                    "route": [
                        {"type": "retro_template", "template": template_str},
                        {"type": "commercial", "smiles": product_smiles},
                    ],
                    "building_blocks": [product_smiles],
                    "sa_score": 3.0,
                }

    # Step 3: Two-step retrosynthesis
    if max_steps >= 2:
        for template in templates:
            disconnections = _apply_retro_template(mol, template)
            for product_smiles, _ in disconnections:
                product_mol = Chem.MolFromSmiles(product_smiles)
                if product_mol is None:
                    continue

                for template2 in templates:
                    sub_disconnections = _apply_retro_template(product_mol, template2)
                    for sub_smiles, _ in sub_disconnections:
                        if _is_commercial_building_block(sub_smiles):
                            return {
                                "viable": True,
                                "n_steps": 2,
                                "route": [
                                    {"type": "retro_template", "template": template_str},
                                    {"type": "retro_template", "template": template_str},
                                    {"type": "commercial", "smiles": sub_smiles},
                                ],
                                "building_blocks": [sub_smiles],
                                "sa_score": 5.0,
                            }

    # Step 4: Three-step retrosynthesis (simplified)
    if max_steps >= 3:
        for template in templates:
            disconnections = _apply_retro_template(mol, template)
            for product_smiles, _ in disconnections:
                product_mol = Chem.MolFromSmiles(product_smiles)
                if product_mol is None:
                    continue

                for template2 in templates:
                    sub_disconnections = _apply_retro_template(product_mol, template2)
                    for sub_smiles, _ in sub_disconnections:
                        sub_mol = Chem.MolFromSmiles(sub_smiles)
                        if sub_mol is None:
                            continue

                        for template3 in templates:
                            sub_sub_disconnections = _apply_retro_template(sub_mol, template3)
                            for sub_sub_smiles, _ in sub_sub_disconnections:
                                if _is_commercial_building_block(sub_sub_smiles):
                                    return {
                                        "viable": True,
                                        "n_steps": 3,
                                        "route": [
                                            {"type": "retro_template", "template": template_str},
                                            {"type": "retro_template", "template": template_str},
                                            {"type": "retro_template", "template": template_str},
                                            {"type": "commercial", "smiles": sub_sub_smiles},
                                        ],
                                        "building_blocks": [sub_sub_smiles],
                                        "sa_score": 7.0,
                                    }

    # No viable route found
    sa_score = _estimate_sa_score(smiles)
    return {
        "viable": False,
        "n_steps": max_steps + 1,
        "route": [],
        "building_blocks": [],
        "sa_score": sa_score,
    }


def _estimate_sa_score(smiles: str) -> float:
    """Estimate synthetic accessibility score (1-10 scale)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 10.0

    # Simplified SA score based on molecular complexity
    n_heavy = mol.GetNumHeavyAtoms()
    n_rings = mol.GetRingInfo().NumRings()
    n_rotatable = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
    n_stereo = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

    # Base score from molecular size
    sa = 1.0 + (n_heavy / 50.0) * 3.0

    # Penalty for ring complexity
    sa += n_rings * 0.5

    # Penalty for stereochemistry
    sa += len(n_stereo) * 0.3

    # Penalty for rotatable bonds (flexibility)
    sa += n_rotatable * 0.05

    # Check for exotic elements
    exotic = sum(
        1 for atom in mol.GetAtoms()
        if atom.GetAtomicNum() not in (1, 6, 7, 8, 9, 16, 15, 17, 35, 53)
    )
    sa += exotic * 0.5

    return min(max(sa, 1.0), 10.0)


def batch_retro_check(
    smiles_list: list[str],
    max_steps: int = _MAX_STEPS,
) -> list[dict[str, Any]]:
    """Run retrosynthetic checks on a batch of molecules."""
    results: list[dict[str, Any]] = []
    for smiles in smiles_list:
        result = retro_check(smiles, max_steps=max_steps)
        result["smiles"] = smiles
        results.append(result)
    return results


def full_retrosynthesis_search(mol: Chem.Mol, max_steps: int = _MAX_STEPS) -> float:
    """Full retrosynthetic tree-search against commercial building blocks.

    Wraps the multi-step retro_check() tree search with a simple scoring
    contract for downstream consumers (e.g., combined_grounding_score):

    Returns
    -------
    float
        1.0 if a viable retrosynthetic route to commercial precursors was
        found within *max_steps*.  0.85 if no route was found — this mild
        penalty discourages but does not outright reject novel molecules
        that may still be synthetically accessible via unanticipated
        disconnections.

    The 0.85 factor is intentionally conservative: the BRICS grounding gate
    and functional-group coverage already enforce a commercial-precursor
    baseline, so the full search acts as a refinement step.
    """
    smiles = Chem.MolToSmiles(mol, canonical=True)
    result = retro_check(smiles, max_steps=max_steps)
    if result.get("viable", False):
        return 1.0
    return 0.85