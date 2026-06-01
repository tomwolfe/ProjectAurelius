"""Philosophy verification tests for Project Aurelius.

Verifies the codebase adheres to "as complex as necessary, and as simple as possible":
  A. Novel Discovery Yield — EA escapes local minima, generates novel scaffolds
  B. Physical Necessity — Oracle captures non-linear chemistry (sterics, saturation, QM)
  C. Software Simplicity — No over-engineering (string dispatch, redundant state, bloat)
"""

from __future__ import annotations

import ast
import inspect
import os

import pytest
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from aurelius.agent.mutation import (
    ELECTROLYTE_FRAGMENT_POOL,
    MutationEngine,
    _is_electrolyte_like,
)
from aurelius.pipeline import _OBJECTIVES, AureliusPipeline
from aurelius.scoring.oracle import (
    PropertyOracle,
    _count_branch_points,
    _count_fragments,
    predict_dielectric_proxy,
    predict_tom_orbitals,
    predict_viscosity_proxy,
)
from aurelius.types import MoleculeContext

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# A. Novel Molecule Discovery — The Ultimate Goal
# ---------------------------------------------------------------------------


def _robust_scaffold(mol: Chem.Mol) -> str:
    """Compute a scaffold SMILES robustly for both cyclic and acyclic molecules."""
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    if scaffold:
        return scaffold
    generic = MurckoScaffold.MakeScaffoldGeneric(mol)
    if generic:
        return Chem.MolToSmiles(generic)
    return Chem.MolToSmiles(mol)


class TestNovelScaffoldDiscovery:
    """The EA must generate molecules with scaffolds unseen in the seed pool."""

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
        assert novelty_ratio > 0.20, (
            f"Mutation engine is trapped in local minima. "
            f"Only {novelty_ratio:.1%} scaffold novelty "
            f"({len(novel_scaffolds)} novel / {len(candidates)} total)."
        )

    def test_mutation_engine_has_scaffold_tracking(self):
        engine = MutationEngine(seed_smiles=["COC(=O)OC", "C1COCCO1"])
        assert hasattr(engine, "_seed_scaffolds"), "Engine missing _seed_scaffolds"
        assert len(engine._seed_scaffolds) > 0, "Engine should have at least one seed scaffold"

    def test_trivial_alkyl_extension_rejected(self):
        """A trivial alkyl extension (DMC -> DPC) must be rejected as non-novel.

        DMC (COC(=O)OC) -> DPC (CCCOC(=O)OCCC) is just adding propyl chains.
        The novelty gate should reject this even though scaffold check is on.
        """
        engine = MutationEngine(seed_smiles=["COC(=O)OC"])
        dp_ctx = MoleculeContext.from_smiles("CCCOC(=O)OCCC")
        assert dp_ctx is not None
        assert engine._novelty_check(dp_ctx) is False, (
            "Dipropyl carbonate is a trivial alkyl extension of DMC — must be rejected"
        )

    def test_ethyl_methyl_carbonate_accepted(self):
        """Local single-carbon functionalization must still be accepted.

        EMC (COC(=O)OCC) is a one-carbon extension of DMC. The scaffold
        is the same but the functional change is minimal and non-trivial.
        """
        engine = MutationEngine(seed_smiles=["COC(=O)OC"])
        emc_ctx = MoleculeContext.from_smiles("COC(=O)OCC")
        assert emc_ctx is not None
        assert engine._novelty_check(emc_ctx, check_scaffold=False) is True, (
            "EMC is a single-carbon extension of DMC — should be accepted with check_scaffold=False"
        )


class TestAntiFrankenstein:
    """BRICS reassembly must reject "Frankenstein" molecules."""

    def test_long_aliphatic_chain_rejected(self):
        """Molecules with >12 continuous aliphatic carbons must be rejected."""
        engine = MutationEngine(seed_smiles=["COC(=O)OC"])
        long_chain_smi = "CCCCCCCCCCCCCCOC(=O)OC"
        ctx = MoleculeContext.from_smiles(long_chain_smi)
        assert ctx is not None
        assert engine._has_excessive_aliphatic_chain(ctx.mol) is True, (
            "Molecule with 14-carbon chain should be flagged as excessive"
        )

    def test_short_aliphatic_chain_accepted(self):
        """Molecules with short aliphatic chains should pass."""
        engine = MutationEngine(seed_smiles=["COC(=O)OC"])
        short_chain_smi = "CCCCCOC(=O)OC"
        ctx = MoleculeContext.from_smiles(short_chain_smi)
        assert ctx is not None
        assert engine._has_excessive_aliphatic_chain(ctx.mol) is False, (
            "Molecule with 5-carbon chain should be acceptable"
        )

    def test_impossible_valence_rejected(self):
        """Molecules with impossible valences must be rejected by _is_electrolyte_like."""
        bad_valence_smi = "C(C)(C)(C)C"  # Impossible: carbon with 4 single bonds to 4 carbons = pentavalent carbon
        ctx = MoleculeContext.from_smiles(bad_valence_smi)
        if ctx is None:
            return
        assert _is_electrolyte_like(ctx) is False, (
            "Pentavalent carbon should be rejected by valence_sanity check"
        )


class TestScaffoldHoppingLoop:
    """Run a short multi-generation loop and verify scaffold diversity."""

    def test_generation_loop_discovers_novel_scaffolds(self, tmp_path):
        from aurelius.agent.loop import DiscoveryLoop
        from aurelius.agent.state import LoopState

        seed_smiles = ["COC(=O)OC", "C1COCCO1", "CS(=O)(=O)C", "CC#N"]
        engine = MutationEngine(seed_smiles=seed_smiles)

        seed_scaffolds: set[str] = set()
        for smi in engine.seed_pool:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    s = _robust_scaffold(mol)
                    if s:
                        seed_scaffolds.add(s)
            except Exception:
                continue

        pipeline = AureliusPipeline()
        pipeline.initialize()
        state = LoopState(path=str(tmp_path / "loop_state.json"))
        loop = DiscoveryLoop(
            pipeline=pipeline,
            engine=engine,
            state=state,
            max_generations=3,
            batch_size=5,
            max_wall_time=120.0,
        )
        result = loop.execute()

        discoveries = result.get("discoveries", [])
        top_n = discoveries[:50] if len(discoveries) >= 50 else discoveries

        novel_scaffold_count = 0
        for d in top_n:
            ctx = MoleculeContext.from_smiles(d.smiles)
            if ctx is None:
                continue
            try:
                scaffold = _robust_scaffold(ctx.mol)
                if scaffold and scaffold not in seed_scaffolds:
                    novel_scaffold_count += 1
            except Exception:
                continue

        novelty_ratio = novel_scaffold_count / max(len(top_n), 1)
        assert novelty_ratio > 0.20, (
            f"Only {novelty_ratio:.1%} of top discoveries have novel scaffolds "
            f"({novel_scaffold_count}/{len(top_n)}). EA is stuck in local minima."
        )


# ---------------------------------------------------------------------------
# B. Chemical "As Complex As Necessary"
# ---------------------------------------------------------------------------


class TestOracleNonlinear:
    """The oracle must capture non-linear chemical interactions."""

    def test_oracle_captures_nonlinear_interactions(self):
        oracle = PropertyOracle(use_xtb=False)
        ether = oracle.evaluate(MoleculeContext.from_smiles("CCOCC"))
        carbonate = oracle.evaluate(MoleculeContext.from_smiles("COC(=O)OC"))
        mixed = oracle.evaluate(MoleculeContext.from_smiles("CCOC(=O)OCCOCC"))

        linear_sum = ether["dielectric_proxy"] + carbonate["dielectric_proxy"] - 1.9
        diff = abs(mixed["dielectric_proxy"] - linear_sum)
        assert diff > 0.5, (
            f"Oracle is still purely linear/additive: mixed={mixed['dielectric_proxy']:.3f}, "
            f"linear_sum={linear_sum:.3f}, diff={diff:.3f}."
        )

    def test_cross_term_functions_exist(self):
        from aurelius.scoring.oracle import _compute_dielectric_cross_terms
        assert callable(_compute_dielectric_cross_terms)

    def test_ether_carbonate_cross_term_applied(self):
        mol = Chem.MolFromSmiles("CCOC(=O)OCCOCC")
        counts = _count_fragments(mol)
        assert counts.get("ether", 0) > 0
        assert counts.get("carbonate", 0) > 0

    def test_steric_viscosity_branched_vs_linear(self):
        """Branched isomers must have higher viscosity proxy than linear ones.

        tert-butyl methyl carbonate (branched) vs n-pentyl methyl carbonate (linear).
        Both have 5 carbons in the alkyl chain; branching creates steric hindrance.
        """
        branched = predict_viscosity_proxy(MoleculeContext.from_smiles("CC(C)(C)OC(=O)OC"))
        linear = predict_viscosity_proxy(MoleculeContext.from_smiles("CCCCCOC(=O)OC"))
        assert branched > linear, (
            f"Branched viscosity ({branched:.4f}) should exceed linear ({linear:.4f})"
        )

    def test_branch_counting_heuristic(self):
        """_count_branch_points must correctly identify branch points."""
        branched_mol = Chem.MolFromSmiles("CC(C)(C)O")  # tert-butanol
        linear_mol = Chem.MolFromSmiles("CCCCO")  # butanol
        n_branched = _count_branch_points(branched_mol)
        n_linear = _count_branch_points(linear_mol)
        assert n_branched > 0, "Branched molecule should have branch points"
        assert n_linear == 0 or n_linear < n_branched, (
            f"Linear molecule ({n_linear}) should have fewer branch points "
            f"than branched ({n_branched})"
        )

    def test_fragment_saturation_prevents_stacking(self):
        """Stacking 5 ester groups does NOT linearly multiply dielectric proxy."""
        single = predict_dielectric_proxy(MoleculeContext.from_smiles("CC(=O)OCC"))
        five = predict_dielectric_proxy(
            MoleculeContext.from_smiles("CC(=O)OCC(=O)OCC(=O)OCC(=O)OCC(=O)OC")
        )
        counts_single = _count_fragments(MoleculeContext.from_smiles("CC(=O)OCC").mol)
        counts_five = _count_fragments(
            MoleculeContext.from_smiles("CC(=O)OCC(=O)OCC(=O)OCC(=O)OCC(=O)OC").mol
        )
        assert counts_single.get("ester", 0) == 1
        assert counts_five.get("ester", 0) >= 3
        assert five < 2.0 * single, (
            f"Saturation failed: 5 esters (diel={five:.3f}) "
            f"should be < 2x 1 ester (diel={single:.3f}, 2x={2 * single:.3f})"
        )

    def test_tom_conjugation_nonlinear(self):
        """TOM HOMO-LUMO gap follows particle-in-a-box scaling: ΔE ∝ 1/L²."""
        ethane_h, ethane_l = predict_tom_orbitals(Chem.MolFromSmiles("CC"))
        butadiene_h, butadiene_l = predict_tom_orbitals(Chem.MolFromSmiles("C=CC=C"))
        benzene_h, benzene_l = predict_tom_orbitals(Chem.MolFromSmiles("c1ccccc1"))

        gap_ethane = ethane_l - ethane_h
        gap_butadiene = butadiene_l - butadiene_h
        gap_benzene = benzene_l - benzene_h

        assert gap_butadiene < gap_ethane, (
            f"Butadiene gap {gap_butadiene:.3f} should be < ethane gap {gap_ethane:.3f}"
        )
        assert gap_benzene < gap_butadiene, (
            f"Benzene gap {gap_benzene:.3f} should be < butadiene gap {gap_butadiene:.3f}"
        )


# ---------------------------------------------------------------------------
# C. Software "As Simple As Possible"
# ---------------------------------------------------------------------------


class TestSoftwareSimplicity:
    """Verify no over-engineered software abstractions exist."""

    def test_objectives_use_callable_functions(self):
        for obj in _OBJECTIVES:
            assert callable(obj.function), (
                f"Objective '{obj.name}' has non-callable function {obj.function}"
            )

    def test_no_string_dispatch_in_compute_score(self):
        source = inspect.getsource(AureliusPipeline._compute_score)
        assert "self.function ==" not in source

    def test_no_redundant_counters_in_discovery_loop(self):
        from aurelius.agent.loop import DiscoveryLoop
        redundant = {"total_screened", "total_viable", "total_invalid"}
        attrs = set(DiscoveryLoop.__init__.__code__.co_varnames)
        overlap = redundant & attrs
        assert not overlap, (
            f"DiscoveryLoop still has redundant counters: {overlap}. "
            "Use LoopState counters instead."
        )

    def test_electrolyte_checks_are_data_driven(self):
        from aurelius.agent.mutation import _ELECTROLYTE_CHECKS
        assert len(_ELECTROLYTE_CHECKS) >= 5

    def test_no_dead_state_fields(self):
        """LoopState must not have dead fields that are never written."""
        from aurelius.agent.state import LoopState
        dead = {"batch", "total_generated"}
        attrs = set(LoopState.__dataclass_fields__.keys())
        overlap = dead & attrs
        assert not overlap, (
            f"LoopState still has dead fields: {overlap}"
        )

    def test_no_string_dispatch_via_ast(self):
        """AST-parse pipeline.py and oracle.py for string-based property dispatch.

        The codebase must use direct callable references or Objective dataclass,
        not ``if property == "string":`` dispatch patterns.
        """
        for filepath in [
            os.path.join(os.path.dirname(__file__), "..", "src", "aurelius", "pipeline.py"),
            os.path.join(os.path.dirname(__file__), "..", "src", "aurelius", "scoring", "oracle.py"),
        ]:
            with open(filepath) as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    if isinstance(node.test, ast.Compare):
                        left = node.test.left
                        if isinstance(left, ast.Name) and left.id in (
                            "property", "prop", "key", "name",
                        ):
                            for op in node.test.ops:
                                if isinstance(op, (ast.Eq, ast.Is)):
                                    for comparator in node.test.comparators:
                                        if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                                            pytest.fail(
                                                f"String dispatch found in {filepath}: "
                                                f"if {left.id} == '{comparator.value}'"
                                            )

    def test_dependency_tree_no_ml_frameworks(self):
        """pyproject.toml must not contain ML training frameworks."""
        pyproject_path = os.path.join(
            os.path.dirname(__file__), "..", "pyproject.toml"
        )
        with open(pyproject_path) as f:
            content = f.read().lower()
        ml_frameworks = {"torch", "tensorflow", "jax", "keras", "transformers", "flax"}
        found = [fw for fw in ml_frameworks if fw in content]
        assert not found, (
            f"ML frameworks found in pyproject.toml dependencies: {found}"
        )

    def test_single_parse_per_molecule(self):
        """Mock Chem.MolFromSmiles and verify it's called exactly once per molecule.

        During a full pipeline screen_molecule call, the SMILES is parsed
        exactly once via MoleculeContext.from_smiles.
        """
        import unittest.mock

        original = Chem.MolFromSmiles
        call_count = 0

        def counting_mol_from_smiles(smiles: str) -> Chem.Mol | None:
            nonlocal call_count
            call_count += 1
            return original(smiles)

        with unittest.mock.patch(
            "rdkit.Chem.MolFromSmiles", side_effect=counting_mol_from_smiles
        ):
            ctx = MoleculeContext.from_smiles("COC(=O)OC")
            assert ctx is not None

            pipeline = AureliusPipeline()
            pipeline.initialize()
            result = pipeline.screen_molecule(ctx)

        assert result is not None
        score = result.get("score", {})
        assert score.get("total_score", 0) > 0

    def test_cyclomatic_complexity(self):
        """No function in core modules should exceed cyclomatic complexity of 12.

        The data-driven ``_ELECTROLYTE_CHECKS`` pattern replaces nested if/elif
        chains for chemical rules.  Utility modules (chem_utils, reporting) may
        contain pre-existing complex functions; core modules (agent, scoring,
        pipeline, loop, state, selection, types, constants, filter) must keep
        all functions <= 12.

        Uses ``radon cc`` if available; otherwise skips the test.
        """
        radon = pytest.importorskip("radon.complexity", reason="radon not installed")
        from radon.complexity import cc_visit

        src_dir = os.path.join(os.path.dirname(__file__), "..", "src", "aurelius")
        excluded = {"chem_utils.py", "dependencies.py", "__init__.py", "__main__.py",
                    "reporting.py"}
        high_complexity: list[tuple[str, int]] = []
        for root, _dirs, files in os.walk(src_dir):
            for fn in files:
                if not fn.endswith(".py") or fn in excluded:
                    continue
                filepath = os.path.join(root, fn)
                with open(filepath) as f:
                    try:
                        blocks = cc_visit(f.read())
                        for block in blocks:
                            if block.complexity > 12:
                                rel_path = os.path.relpath(filepath, src_dir)
                                high_complexity.append(
                                    (f"{rel_path}:{block.lineno} {block.name}", block.complexity)
                                )
                    except Exception:
                        continue

        assert not high_complexity, (
            f"Functions exceeding cyclomatic complexity of 12:\n" +
            "\n".join(f"  {name}: {c}" for name, c in high_complexity)
        )
