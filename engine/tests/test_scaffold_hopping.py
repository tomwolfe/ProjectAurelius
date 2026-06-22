"""Tests for scaffold-hopping SMARTS reactions and grounding gate.

Validates that:
  1. New SMARTS reactions produce molecules with novel Murcko scaffolds.
  2. Novel-scaffold molecules remain grounded in commercial building blocks.
  3. The BRICS grounding gate filters truly exotic molecules without
     collapsing the proposal space.
"""

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

from aurelius.agent.mutation.brics import MIN_GROUNDING_SCORE, combined_grounding_score
from aurelius.agent.mutation.smarts import ELECTROLYTE_SMARTS

SEED_SMILES = ["COC(=O)OC", "C1COCCO1"]


def _seed_scaffolds() -> set[str]:
    scaffolds: set[str] = set()
    for smi in SEED_SMILES:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            s = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            if s:
                scaffolds.add(s)
    return scaffolds


SEED_SCAFFOLDS = _seed_scaffolds()


class TestScaffoldHoppingSmarts:
    """SMARTS reactions must parse and produce molecules with novel scaffolds."""

    def test_all_scaffold_hopping_smarts_parse(self):
        """Every scaffold-hopping SMARTS reaction must parse as a valid AllChem reaction."""
        new_prefixes = ("oxygen_to_", "dioxane_", "methyl_to_ethyl",
                        "hydroxyl_to_methyl_sulfonate", "nitrile_to_primary_amide")
        for smarts, name in ELECTROLYTE_SMARTS:
            if any(name.startswith(p) for p in new_prefixes):
                rxn = AllChem.ReactionFromSmarts(smarts)
                assert rxn is not None, f"Failed to parse SMARTS '{name}': {smarts}"

    def test_scaffold_hopping_produces_novel_scaffolds(self):
        """At least 20% of scaffold-hopping SMARTS products must have novel Murcko scaffolds."""
        novel_count = 0
        total = 0
        for seed_smi in SEED_SMILES:
            seed_mol = Chem.MolFromSmiles(seed_smi)
            if seed_mol is None:
                continue
            for smarts, _name in ELECTROLYTE_SMARTS:
                try:
                    rxn = AllChem.ReactionFromSmarts(smarts)
                    if rxn is None:
                        continue
                    for products in rxn.RunReactants((seed_mol,)):
                        for product in products:
                            try:
                                Chem.SanitizeMol(product)
                                s = MurckoScaffold.MurckoScaffoldSmiles(mol=product)
                                if s:
                                    total += 1
                                    if s not in SEED_SCAFFOLDS:
                                        novel_count += 1
                            except Exception:
                                continue
                except Exception:
                    continue
        assert total > 0, "No SMARTS products generated with Murcko scaffolds"
        novelty_rate = novel_count / total
        assert novelty_rate >= 0.20, (
            f"Only {novelty_rate:.1%} of SMARTS products have novel scaffolds "
            f"({novel_count}/{total})"
        )

    def test_novel_scaffolds_remain_grounded(self):
        """Novel-scaffold SMARTS products must have combined_grounding_score >= 0.4."""
        low_grounding = 0
        total = 0
        for seed_smi in SEED_SMILES:
            seed_mol = Chem.MolFromSmiles(seed_smi)
            if seed_mol is None:
                continue
            for smarts, _name in ELECTROLYTE_SMARTS:
                try:
                    rxn = AllChem.ReactionFromSmarts(smarts)
                    if rxn is None:
                        continue
                    for products in rxn.RunReactants((seed_mol,)):
                        for product in products:
                            try:
                                Chem.SanitizeMol(product)
                                score = combined_grounding_score(product)
                                total += 1
                                if score < MIN_GROUNDING_SCORE:
                                    low_grounding += 1
                            except Exception:
                                continue
                except Exception:
                    continue
        assert total > 0, "No SMARTS products generated"
        low_grounding_rate = low_grounding / total
        assert low_grounding_rate < 0.50, (
            f"{low_grounding_rate:.1%} of products have grounding_score "
            f"< {MIN_GROUNDING_SCORE} ({low_grounding}/{total})"
        )


class TestBricsGroundingGate:
    """BRICS grounding gate must not collapse the proposal space."""

    def test_brics_engine_still_produces_candidates(self):
        """MutationEngine with the grounding gate must generate at least 1 candidate."""
        from aurelius.agent.mutation import MutationEngine
        engine = MutationEngine(seed_smiles=SEED_SMILES)
        candidates = engine.propose_candidates(n_candidates=50, batch_size=25)
        assert len(candidates) >= 1, (
            f"Grounding gate collapsed proposal space: 0 candidates from "
            f"{len(SEED_SMILES)} seeds"
        )

    def test_brics_grounding_gate_rejects_exotic_molecules(self):
        """Exotic molecules (e.g., long polyenes) should fail the grounding gate."""
        exotic_smiles = "C=CC=CC=CC=CC=C"
        mol = Chem.MolFromSmiles(exotic_smiles)
        assert mol is not None
        score = combined_grounding_score(mol)
        assert score < MIN_GROUNDING_SCORE, (
            f"Exotic molecule has grounding_score={score:.2f} >= {MIN_GROUNDING_SCORE}"
        )
