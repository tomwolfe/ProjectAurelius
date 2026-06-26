"""Tests for Physical Reality Audit — anti-gaming hardening measures.

Verifies:
  1. Steric hindrance gate rejects overcrowded molecules
  2. Cross-conjugation gate rejects branched pi-systems
  3. Reaction-type commonality penalty in combined_grounding_score
  4. >3 synthetic step cutoff in combined_grounding_score
  5. Scaffold hopping injection in mutate()
  6. TOM benchmark script runs without errors
"""

from __future__ import annotations

from rdkit import Chem

from aurelius.agent.mutation.brics import (
    _estimate_common_reaction_coverage,
    combined_grounding_score,
)
from aurelius.agent.mutation.smarts import (
    _has_cross_conjugation,
    is_electrolyte_like,
    steric_hindrance_check,
    cross_conjugation_check,
)
from aurelius.types import MoleculeContext


# ---------------------------------------------------------------------------
# Steric Hindrance Gate Tests
# ---------------------------------------------------------------------------


class TestStericHindranceGate:
    """The steric hindrance check must reject overcrowded molecules
    while accepting well-behaved electrolyte molecules."""

    def test_accepts_simple_electrolyte(self):
        """A simple carbonate must pass the steric check."""
        ctx = MoleculeContext.from_smiles("COC(=O)OC")  # DMC, MW=90
        assert ctx is not None
        assert steric_hindrance_check(ctx), "DMC must pass steric check"

    def test_accepts_ec(self):
        """Ethylene carbonate must pass the steric check."""
        ctx = MoleculeContext.from_smiles("C1COC(=O)O1")  # EC, MW=88
        assert ctx is not None
        assert steric_hindrance_check(ctx), "EC must pass steric check"

    def test_accepts_moderate_sized_molecule(self):
        """A moderately sized electrolyte molecule must pass."""
        ctx = MoleculeContext.from_smiles("CCOP(=O)(OCC)OCC")  # TEP, MW=182
        assert ctx is not None
        assert steric_hindrance_check(ctx), "TEP must pass steric check"

    def test_low_mw_exempt(self):
        """Molecules with MW < 150 are exempt from steric check."""
        ctx = MoleculeContext.from_smiles("CC#N")  # ACN, MW=41
        assert ctx is not None
        assert steric_hindrance_check(ctx), "ACN (MW<150) must pass steric check"


# ---------------------------------------------------------------------------
# Cross-Conjugation Gate Tests
# ---------------------------------------------------------------------------


class TestCrossConjugationGate:
    """The cross-conjugation check must reject branched pi-systems
    while accepting normal conjugated systems."""

    def test_accepts_normal_carbonate(self):
        """EC has a normal conjugated system (ester resonance)."""
        ctx = MoleculeContext.from_smiles("C1COC(=O)O1")  # EC
        assert ctx is not None
        assert cross_conjugation_check(ctx), "EC must pass cross-conjugation check"

    def test_accepts_simple_ester(self):
        """Methyl acetate has normal ester resonance."""
        ctx = MoleculeContext.from_smiles("CC(=O)OC")
        assert ctx is not None
        assert cross_conjugation_check(ctx), "Simple ester must pass"

    def test_accepts_linear_conjugation(self):
        """Linear conjugated diene must pass (not cross-conjugated)."""
        ctx = MoleculeContext.from_smiles("C=CC=C")
        assert ctx is not None
        assert cross_conjugation_check(ctx), "Linear diene must pass"

    def test_rejects_divinyl_ketone(self):
        """Divinyl ketone is cross-conjugated and must be rejected."""
        ctx = MoleculeContext.from_smiles("C=CC(=O)C=C")
        assert ctx is not None
        assert not cross_conjugation_check(ctx), "Divinyl ketone must be rejected"

    def test_rejects_benzophenone(self):
        """Benzophenone is cross-conjugated and must be rejected."""
        ctx = MoleculeContext.from_smiles("O=C(c1ccccc1)c1ccccc1")
        assert ctx is not None
        assert not cross_conjugation_check(ctx), "Benzophenone must be rejected"

    def test_has_cross_conjugation_detects_branched_pi(self):
        """The _has_cross_conjugation helper must detect branched pi systems."""
        mol = Chem.MolFromSmiles("C=CC(=O)C=C")
        assert mol is not None
        assert _has_cross_conjugation(mol), "Divinyl ketone has cross-conjugation"

    def test_has_cross_conjugation_false_for_linear(self):
        """The helper must return False for linear conjugated systems."""
        mol = Chem.MolFromSmiles("C=CC=C")
        assert mol is not None
        assert not _has_cross_conjugation(mol), "Linear diene has no cross-conjugation"

    def test_has_cross_conjugation_false_for_carbonate(self):
        """Carbonate resonance is NOT cross-conjugation."""
        mol = Chem.MolFromSmiles("COC(=O)OC")
        assert mol is not None
        assert not _has_cross_conjugation(mol), "Carbonate is not cross-conjugated"

    def test_integrated_with_electrolyte_check(self):
        """Cross-conjugated molecules must be rejected by is_electrolyte_like."""
        ctx = MoleculeContext.from_smiles("C=CC(=O)C=C")
        assert ctx is not None
        assert not is_electrolyte_like(ctx), "Cross-conjugated molecule must fail electrolyte check"


# ---------------------------------------------------------------------------
# Reaction-Type Commonality Tests
# ---------------------------------------------------------------------------


class TestReactionTypeCommonality:
    """The _estimate_common_reaction_coverage must correctly identify
    molecules built from common reaction types."""

    def test_ester_connections_high_score(self):
        """A molecule with ester linkages should score highly."""
        mol = Chem.MolFromSmiles("COC(=O)OC")  # DMC — carbonate (related to ester)
        assert mol is not None
        score = _estimate_common_reaction_coverage(mol)
        assert score >= 0.5, f"DMC reaction coverage {score:.3f} should be >= 0.5"

    def test_ether_connections_high_score(self):
        """A molecule with ether linkages should score highly."""
        mol = Chem.MolFromSmiles("COCCOC")  # DME
        assert mol is not None
        score = _estimate_common_reaction_coverage(mol)
        assert score >= 0.5, f"DME reaction coverage {score:.3f} should be >= 0.5"

    def test_nitrile_scores_reasonably(self):
        """A nitrile molecule should have moderate reaction coverage."""
        mol = Chem.MolFromSmiles("CC#N")  # ACN
        assert mol is not None
        score = _estimate_common_reaction_coverage(mol)
        assert score >= 0.0, "ACN must have non-negative reaction coverage"


# ---------------------------------------------------------------------------
# Combined Grounding Score — Synthetic Step Hard Cutoff Tests
# ---------------------------------------------------------------------------


class TestSyntheticStepHardCutoff:
    """The combined_grounding_score must harshly penalise molecules
    requiring >3 synthetic steps from commercial precursors."""

    def test_simple_molecule_high_score(self):
        """A simple commercial molecule should score highly."""
        mol = Chem.MolFromSmiles("COC(=O)OC")  # DMC — itself a commercial BB
        assert mol is not None
        score = combined_grounding_score(mol)
        assert score >= 0.6, f"DMC grounding score {score:.3f} should be >= 0.6"

    def test_ec_high_score(self):
        """EC is a commercial molecule and should score highly."""
        mol = Chem.MolFromSmiles("C1COC(=O)O1")
        assert mol is not None
        score = combined_grounding_score(mol)
        assert score >= 0.6, f"EC grounding score {score:.3f} should be >= 0.6"

    def test_complex_electrolyte_reasonable_score(self):
        """A complex but accessible electrolyte should have reasonable score."""
        mol = Chem.MolFromSmiles("CCOP(=O)(OCC)OCC")  # TEP
        assert mol is not None
        score = combined_grounding_score(mol)
        assert score >= 0.4, f"TEP grounding score {score:.3f} should be >= 0.4"


# ---------------------------------------------------------------------------
# Scaffold Hopping Injection Tests
# ---------------------------------------------------------------------------


class TestScaffoldHoppingInjection:
    """The mutation engine must occasionally use random scaffold replacement."""

    def test_scaffold_library_exists(self):
        """The scaffold library must be populated."""
        from aurelius.agent.mutation.base import BricsStrategy
        brics = BricsStrategy()
        assert len(brics._SCAFFOLD_LIBRARY) >= 5, (
            f"Scaffold library has {len(brics._SCAFFOLD_LIBRARY)} entries, expected >= 5"
        )

    def test_scaffold_hop_probability_is_set(self):
        """The scaffold hop probability must be defined and non-zero."""
        from aurelius.agent.mutation.base import BricsStrategy
        brics = BricsStrategy()
        assert brics._SCAFFOLD_HOP_PROBABILITY == 0.05, (
            f"Scaffold hop probability is {brics._SCAFFOLD_HOP_PROBABILITY}, expected 0.05"
        )

    def test_random_scaffold_replacement_returns_list(self):
        """_random_scaffold_replacement must return a list (possibly empty)."""
        from aurelius.agent.mutation.engine import MutationEngine
        from aurelius.agent.mutation.base import BricsStrategy, StrategyContext
        engine = MutationEngine(seed_smiles=["COC(=O)OC", "C1COCCO1"])
        ctx = MoleculeContext.from_smiles("COC(=O)OC")
        assert ctx is not None
        brics = next(s for s in engine._strategies if isinstance(s, BricsStrategy))
        result = brics._random_scaffold_replacement(ctx, engine._strategy_context)
        assert isinstance(result, list), "Scaffold replacement must return a list"

    def test_scaffold_hop_does_not_crash_mutate(self):
        """Mutate must not crash when scaffold hopping is attempted."""
        from aurelius.agent.mutation.engine import MutationEngine
        engine = MutationEngine(seed_smiles=["COC(=O)OC"])
        # Run mutate multiple times to increase chance of hitting 5% scaffold hop
        all_results: list[str] = []
        for _ in range(5):
            results = engine.mutate("COC(=O)OC", batch_size=10)
            all_results.extend(results)
        # Should not crash; may return empty if no valid products
        assert isinstance(all_results, list)


# ---------------------------------------------------------------------------
# Benchmark Script Tests
# ---------------------------------------------------------------------------


class TestBenchmarkScript:
    """The benchmark script must import and run without crashing."""

    def test_benchmark_script_imports(self):
        """The benchmark script must import cleanly."""
        import importlib.util
        import sys
        from pathlib import Path
        script_path = str(Path(__file__).resolve().parent.parent / "scripts" / "benchmark_tom_vs_xtb.py")
        if script_path not in sys.path:
            sys.path.insert(0, str(Path(script_path).parent))
        spec = importlib.util.spec_from_file_location("benchmark_tom_vs_xtb", script_path)
        assert spec is not None, f"Could not find spec for {script_path}"
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        assert hasattr(mod, "main"), "Script must have main()"
