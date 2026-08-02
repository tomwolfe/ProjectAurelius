"""Tests for diagnostics collection in mutation engine.

Verifies that rejection reasons are captured when a ``diagnostics`` list
is provided, and silently ignored when ``diagnostics=None`` (default).
"""

from __future__ import annotations

from rdkit import Chem

from aurelius.agent.mutation import BricsStrategy, MutationEngine, SmartsStrategy
from aurelius.agent.mutation.base import StrategyContext

# ---------------------------------------------------------------------------
# Engine-level: diagnostics plumbing through public API
# ---------------------------------------------------------------------------


class TestEngineDiagnostics:
    """``diagnostics=None`` (default) must not crash; ``diagnostics=[]`` must
    be accepted and may be populated."""

    def test_diagnostics_none_default(self) -> None:
        engine = MutationEngine(seed_smiles=["COC(=O)OC"])
        result = engine.mutate("COC(=O)OC", batch_size=10)
        assert isinstance(result, list)

    def test_diagnostics_list_accepted(self) -> None:
        engine = MutationEngine(seed_smiles=["COC(=O)OC"])
        diag: list[str] = []
        result = engine.mutate("COC(=O)OC", batch_size=10, diagnostics=diag)
        assert isinstance(result, list)

    def test_diagnostics_mutate_batch(self) -> None:
        engine = MutationEngine(seed_smiles=["COC(=O)OC", "C1COCCO1"])
        diag: list[str] = []
        result = engine.mutate_batch(
            ["COC(=O)OC", "C1COCCO1"], batch_size=10, diagnostics=diag,
        )
        assert isinstance(result, list)

    def test_diagnostics_propose_candidates(self) -> None:
        engine = MutationEngine(seed_smiles=["COC(=O)OC"])
        diag: list[str] = []
        result = engine.propose_candidates(n_candidates=10, diagnostics=diag)
        assert isinstance(result, list)

    def test_diagnostics_mutate_by_concept(self) -> None:
        engine = MutationEngine(seed_smiles=["COC(=O)OC"])
        diag: list[str] = []
        result = engine.mutate_by_concept(
            "COC(=O)OC", batch_size=10, diagnostics=diag,
        )
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Strategy-level: specific diagnostic messages with controlled inputs
# ---------------------------------------------------------------------------


class TestSmartsStrategyDiagnostics:
    """SmartsStrategy._process_smarts_product must record rejection reasons."""

    def test_product_none(self) -> None:
        strategy = SmartsStrategy()
        ctx = StrategyContext()
        diag: list[str] = []
        result = strategy._process_smarts_product(
            None, "COC(=O)OC", ctx, _diagnostics=diag,
        )
        assert result is None
        assert diag == ["SMARTS: product is None"]

    def test_identical_to_seed(self) -> None:
        strategy = SmartsStrategy()
        ctx = StrategyContext()
        diag: list[str] = []
        mol = Chem.MolFromSmiles("COC(=O)OC")
        assert mol is not None
        result = strategy._process_smarts_product(
            mol, "COC(=O)OC", ctx, _diagnostics=diag,
        )
        assert result is None
        assert any("identical to seed" in m for m in diag)

    def test_not_electrolyte_like(self) -> None:
        strategy = SmartsStrategy()
        ctx = StrategyContext()
        diag: list[str] = []
        mol = Chem.MolFromSmiles("c1ccncc1")  # pyridine: HBA=1 but <20% sp3
        assert mol is not None
        result = strategy._process_smarts_product(
            mol, "COC(=O)OC", ctx, _diagnostics=diag,
        )
        assert result is None
        assert any("not electrolyte-like" in m for m in diag)

    def test_none_diagnostics_no_collection(self) -> None:
        """When _diagnostics=None, rejection must not raise or collect."""
        strategy = SmartsStrategy()
        ctx = StrategyContext()
        result = strategy._process_smarts_product(
            None, "COC(=O)OC", ctx, _diagnostics=None,
        )
        assert result is None


class TestBricsStrategyDiagnostics:
    """BricsStrategy._validate_brics_product must record rejection reasons."""

    def test_product_none(self) -> None:
        strategy = BricsStrategy()
        ctx = StrategyContext()
        diag: list[str] = []
        result = strategy._validate_brics_product(
            None, ctx, _diagnostics=diag,
        )
        assert result is None
        assert diag == ["BRICS: product is None"]

    def test_not_electrolyte_like(self) -> None:
        strategy = BricsStrategy()
        ctx = StrategyContext()
        diag: list[str] = []
        mol = Chem.MolFromSmiles("c1ccncc1")  # pyridine: HBA=1 but <20% sp3
        assert mol is not None
        result = strategy._validate_brics_product(
            mol, ctx, _diagnostics=diag,
        )
        assert result is None
        assert any("not electrolyte-like" in m for m in diag)

    def test_none_diagnostics_no_collection(self) -> None:
        strategy = BricsStrategy()
        ctx = StrategyContext()
        result = strategy._validate_brics_product(
            None, ctx, _diagnostics=None,
        )
        assert result is None
