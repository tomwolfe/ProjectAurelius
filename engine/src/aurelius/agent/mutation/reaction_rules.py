"""Reaction Rule Engine — Lightweight Retrosynthetic Feasibility Check.

Provides a small set of robust, high-yield reaction SMARTS patterns for
common electrolyte-relevant transformations (Suzuki, Ullmann, Esterification,
carbonate formation, nitrile hydration). Each rule includes a SMARTS pattern,
a human-readable name, and a confidence weight.

The engine checks whether a given molecule pair (two BRICS fragments mapped to
commercial precursors) can be recombined via a known reaction rule. If no rule
matches, a grounding penalty is applied.

KISS check: SMARTS-based reaction rules avoid heavy retrosynthesis AI
frameworks (e.g., AiZynthFinder). Each rule is 1-2 lines and interpretable.
"""

from __future__ import annotations

from rdkit import Chem

# Reaction rule tuples: (name, reaction_smarts, confidence_weight)
# confidence_weight in [0.0, 1.0] — how reliable/high-yield the reaction is
# for electrolyte-scale synthesis.
_REACTION_RULES: list[tuple[str, str, float]] = [
    # Suzuki coupling: aryl/vinyl boronic acid + aryl/vinyl halide -> biaryl
    # [c:1][B:2] >> [c:1][c:3]  — actually SMARTS for checking if a C-C bond
    # can be formed between two sp2 carbons (one with a boronic acid handle)
    ("suzuki", "[c:1][BX3]([OX2])[OX2].[c:2][Cl,Br,I]", 0.90),
    # Ullmann coupling: aryl halide + aryl halide -> biaryl (Cu-mediated)
    ("ullmann", "[c:1][Cl,Br,I].[c:2][Cl,Br,I]", 0.70),
    # Esterification: carboxylic acid + alcohol -> ester
    ("esterification", "[CX3](=O)[OX2H1].[OX2H1][CX4]", 0.95),
    # Carbonate formation: alcohol + chloroformate -> carbonate
    ("carbonate_formation", "[OX2H1][CX4].Cl[CX3](=O)[OX2][CX4]", 0.85),
    # Amidation: carboxylic acid + amine -> amide
    ("amidation", "[CX3](=O)[OX2H1].[NX3;H1,H2]", 0.90),
    # Williamson ether synthesis: alkoxide + alkyl halide -> ether
    ("williamson_ether", "[OX2H1][CX4].[Cl,Br,I][CX4]", 0.80),
    # Nitrile alkylation: nitrile + alkyl halide -> alkylated nitrile
    ("nitrile_alkylation", "[C:1]#[N:2].[Cl,Br,I][CX4]", 0.75),
    # Sulfonamide formation: sulfonyl chloride + amine -> sulfonamide
    ("sulfonamide", "S(=O)(=O)Cl.[NX3;H1,H2]", 0.85),
    # SN2 alkylation: alkyl halide + nucleophile -> substituted product
    ("sn2_alkylation", "[Cl,Br,I][CX4;!c].[NX3;H1,H2]", 0.80),
    # Alkylation of alcohol: alcohol + alkyl halide
    ("alcohol_alkylation", "[OX2H1][CX4;!c].[Cl,Br,I][CX4;!c]", 0.75),
]

# Cache pre-compiled rule patterns
_RULE_SMARTS: list[tuple[str, Chem.Mol, float]] = []
for name, smarts, weight in _REACTION_RULES:
    pat = Chem.MolFromSmarts(smarts)
    if pat is not None:
        _RULE_SMARTS.append((name, pat, weight))


def check_retrosynthetic_feasibility(mol: Chem.Mol) -> float:
    """Check if a molecule can be synthesised via known reaction rules.

    Scores the molecule based on whether its substructural features match
    any known high-yield reaction SMARTS. A score of 1.0 means at least one
    high-confidence (weight >= 0.8) rule matches. Lower values indicate that
    the molecule's connectivity may require exotic or multi-step chemistry.

    The check is a lightweight SMARTS substructure search against the
    pre-compiled rule library. Each matching rule contributes its confidence
    weight; the best match determines the final score.

    Args:
        mol: RDKit molecule.

    Returns:
        Score in [0.0, 1.0]. 1.0 = synthesizable via known high-yield reaction.
        0.0 = no matching reaction rule found.
    """
    best_score = 0.0
    for _name, pat, weight in _RULE_SMARTS:
        if mol.HasSubstructMatch(pat):
            best_score = max(best_score, weight)
    return best_score
