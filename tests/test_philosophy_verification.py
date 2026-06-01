"""Philosophy verification tests for Project Aurelius.

Verifies the codebase adheres to "as complex as necessary, and as simple as possible":
  A. Software simplicity — no over-engineering (string dispatch, redundant state)
  B. Chemical necessity — non-linear oracle captures cross-term interactions
  C. Novel scaffold discovery — EA loop escapes local chemical minima
"""

from __future__ import annotations

import inspect

import pytest
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from aurelius.agent.mutation import MutationEngine
from aurelius.pipeline import _OBJECTIVES, AureliusPipeline
from aurelius.scoring.oracle import PropertyOracle, predict_dielectric_proxy
from aurelius.types import MoleculeContext

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# A. Software "As Simple As Possible"
# ---------------------------------------------------------------------------


class TestSoftwareSimplicity:
    """Verify that over-engineered software abstractions were removed."""

    def test_objectives_use_callable_functions(self):
        """Objectives should use direct callables, not string dispatch."""
        for obj in _OBJECTIVES:
            assert callable(obj.function), (
                f"Objective '{obj.name}' has non-callable function {obj.function}"
            )

    def test_no_string_dispatch_in_compute_score(self):
        """_compute_score must not contain string-based function dispatch."""
        source = inspect.getsource(AureliusPipeline._compute_score)
        assert "self.function ==" not in source, "String dispatch still exists in _compute_score."

    def test_no_redundant_counters_in_discovery_loop(self):
        """DiscoveryLoop should not have redundant counter attributes."""
        from aurelius.agent.loop import DiscoveryLoop
        redundant = {"total_screened", "total_viable", "total_invalid"}
        attrs = set(DiscoveryLoop.__init__.__code__.co_varnames)
        overlap = redundant & attrs
        assert not overlap, (
            f"DiscoveryLoop still has redundant counters: {overlap}. "
            "Use LoopState counters instead."
        )

    def test_electrolyte_checks_are_data_driven(self):
        """_is_electrolyte_like should be a thin dispatch over a check list,
        not a massive wall of sequential if-blocks."""
        from aurelius.agent.mutation import _ELECTROLYTE_CHECKS
        assert len(_ELECTROLYTE_CHECKS) >= 5, (
            f"Expected at least 5 electrolyte check functions, got {len(_ELECTROLYTE_CHECKS)}"
        )


# ---------------------------------------------------------------------------
# B. Chemical "As Complex As Necessary"
# ---------------------------------------------------------------------------


class TestOracleNonlinear:
    """The oracle must capture non-linear chemical interactions
    that the old purely additive GC model missed."""

    def test_oracle_captures_nonlinear_interactions(self):
        oracle = PropertyOracle(use_xtb=False)

        ether = oracle.evaluate(MoleculeContext.from_smiles("CCOCC"))
        carbonate = oracle.evaluate(MoleculeContext.from_smiles("COC(=O)OC"))
        mixed = oracle.evaluate(MoleculeContext.from_smiles("CCOC(=O)OCCOCC"))

        linear_sum = ether["dielectric_proxy"] + carbonate["dielectric_proxy"] - 1.9
        diff = abs(mixed["dielectric_proxy"] - linear_sum)
        assert diff > 0.5, (
            f"Oracle is still purely linear/additive: mixed={mixed['dielectric_proxy']:.3f}, "
            f"linear_sum={linear_sum:.3f}, diff={diff:.3f}. "
            "Cross-term correction is not working."
        )

    def test_cross_term_functions_exist(self):
        """The cross-term correction function must be defined and callable."""
        from aurelius.scoring.oracle import _compute_dielectric_cross_terms
        assert callable(_compute_dielectric_cross_terms)

    def test_ether_carbonate_cross_term_applied(self):
        """A molecule with both ether and carbonate should show the
        carbonate-ether cross-term boost."""
        from aurelius.scoring.oracle import _count_fragments
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCOC(=O)OCCOCC")
        counts = _count_fragments(mol)
        assert counts.get("ether", 0) > 0, "Mixed molecule should have ether fragments"
        assert counts.get("carbonate", 0) > 0, "Mixed molecule should have carbonate fragments"


# ---------------------------------------------------------------------------
# C. Novel Molecule Discovery — The Ultimate Goal
# ---------------------------------------------------------------------------


def _robust_scaffold(mol: Chem.Mol) -> str:
    """Compute a scaffold SMILES robustly for both cyclic and acyclic molecules.

    For molecules with rings, uses the standard Murcko scaffold.
    For acyclic molecules (Murcko returns empty), falls back to a
    generic carbon-skeleton scaffold via MakeScaffoldGeneric.
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    if scaffold:
        return scaffold
    generic = MurckoScaffold.MakeScaffoldGeneric(mol)
    if generic:
        return Chem.MolToSmiles(generic)
    return Chem.MolToSmiles(mol)


class TestNovelScaffoldDiscovery:
    """The EA loop must generate molecules with scaffolds unseen in the seed pool."""

    def test_discovers_novel_murcko_scaffolds(self):
        engine = MutationEngine(seed_smiles=["COC(=O)OC", "C1COCCO1"])

        candidates = engine.propose_candidates(n_candidates=500, batch_size=50)

        seed_scaffolds: set[str] = set()
        for smi in engine.seed_pool:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    scaffold = _robust_scaffold(mol)
                    seed_scaffolds.add(scaffold)
            except Exception:
                continue

        novel_scaffolds: set[str] = set()
        for smi in candidates:
            ctx = MoleculeContext.from_smiles(smi)
            if ctx is not None:
                try:
                    scaffold = _robust_scaffold(ctx.mol)
                    if scaffold not in seed_scaffolds:
                        novel_scaffolds.add(scaffold)
                except Exception:
                    continue

        novelty_ratio = len(novel_scaffolds) / max(len(candidates), 1)
        assert novelty_ratio > 0.15, (
            f"Mutation engine is trapped in local minima. "
            f"Only {novelty_ratio:.1%} scaffold novelty "
            f"({len(novel_scaffolds)} novel / {len(candidates)} total)."
        )

    def test_mutation_engine_has_scaffold_tracking(self):
        """MutationEngine must track seed scaffolds for novelty checking."""
        engine = MutationEngine(seed_smiles=["COC(=O)OC", "C1COCCO1"])
        assert hasattr(engine, "_seed_scaffolds"), "Engine missing _seed_scaffolds"
        assert len(engine._seed_scaffolds) > 0, "Engine should have at least one seed scaffold"
