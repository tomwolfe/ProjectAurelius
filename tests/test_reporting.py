"""Tests for the ReportingEngine wet-lab handoff module (Gap 3 integration debt)."""

from __future__ import annotations

from pathlib import Path

from rdkit import Chem

from aurelius.agent.loop import ScreeningResult
from aurelius.reporting import (
    ReportingEngine,
    _confidence_interval,
    _passes_stage,
)
from aurelius.types import MoleculeContext


def _make_result(
    smiles: str = "COC(=O)OC",
    total_score: float = 80.0,
    grounding: float = 0.9,
    domain: float = 0.98,
    depth: int = 1,
    novelty: float = 0.5,
    viable: bool = True,
) -> ScreeningResult:
    ctx = MoleculeContext.from_smiles(smiles)
    ctx.mol if ctx else Chem.MolFromSmiles(smiles)
    return ScreeningResult(
        smiles=smiles,
        total_score=total_score,
        is_viable=viable,
        rejection_reasons=[],
        novelty_to_seed=novelty,
        homo_eV=-7.0,
        lumo_eV=0.0,
        dielectric_proxy=20.0,
        viscosity_proxy=1.5,
        sa_score=3.0,
        synthesis_depth=depth,
        sub_scores={"grounding": grounding, "domain": domain, "confidence": 0.9},
        combined_grounding_score=grounding,
    )


class TestReportingEngine:
    def test_candidate_builder_passes_filters(self, tmp_path: Path) -> None:
        """A well-scored molecule becomes a complete candidate dict."""
        engine = ReportingEngine()
        result = _make_result()
        cand = engine._build_candidate(result, [], skip_dft=True)
        assert cand is not None
        assert cand["is_viable"] is True
        # Output fields the CSV writer needs to serialize.
        for field in ["smiles", "total_score", "adjusted_score", "combined_grounding_score",
                       "domain_penalty", "synthesis_feasibility", "homo_eV", "lumo_eV"]:
            assert field in cand

    def test_candidate_builder_rejects_low_score(self) -> None:
        """Molecules below the 65-point threshold are rejected."""
        engine = ReportingEngine()
        result = _make_result(total_score=40.0)
        assert engine._build_candidate(result, [], skip_dft=True) is None

    def test_candidate_builder_rejects_low_novelty(self) -> None:
        engine = ReportingEngine()
        result = _make_result(novelty=0.1)
        assert engine._build_candidate(result, [], skip_dft=True) is None

    def test_cascade_selects_passing_molecules(self) -> None:
        engine = ReportingEngine()
        result = _make_result(grounding=0.8, domain=0.96, depth=1, novelty=0.4)
        cand = engine._build_candidate(result, [], skip_dft=True)
        assert cand is not None
        selected, rej = engine._apply_cascade([cand], skip_dft=True)
        assert len(selected) == 1
        assert rej["is_viable"] == 0

    def test_cascade_rejects_low_grounding(self) -> None:
        engine = ReportingEngine()
        result = _make_result(grounding=0.3, domain=0.96, depth=1, novelty=0.4)
        cand = engine._build_candidate(result, [], skip_dft=True)
        assert cand is not None
        selected, rej = engine._apply_cascade([cand], skip_dft=True)
        assert len(selected) == 0
        assert rej["combined_grounding_score"] == 1

    def test_report_markdown_has_sections(self) -> None:
        engine = ReportingEngine()
        result = _make_result()
        cand = engine._build_candidate(result, [], skip_dft=True)
        md = engine._render_markdown([cand], [cand], {"is_viable": 0, "combined_grounding_score": 0, "synthesis_depth": 0, "domain_penalty": 0, "novelty_to_seed": 0})
        assert "# Prospective Candidates Report" in md
        assert "Cascade Funnel Visualization" in md
        assert "Selection Rationale" in md

    def test_csv_written_with_legacy_fields(self, tmp_path: Path) -> None:
        engine = ReportingEngine()
        result = _make_result()
        cand = engine._build_candidate(result, [], skip_dft=True)
        out = tmp_path / "c.csv"
        engine._render_csv([cand], str(out))
        assert out.exists()
        text = out.read_text()
        assert "smiles" in text
        assert "adjusted_score" in text


class TestScoreHelpers:
    def test_confidence_interval_handles_none(self) -> None:
        lo, hi = _confidence_interval(None, None, 1.0)
        assert lo == 0.0 and hi == 0.0

    def test_confidence_interval_spread(self) -> None:
        lo, hi = _confidence_interval(-7.0, 1.0, 0.5)
        assert lo < hi

    def test_passes_stage_is_viable(self) -> None:
        assert _passes_stage({"is_viable": True}, "is_viable", True) is True
        assert _passes_stage({"is_viable": False}, "is_viable", True) is False

    def test_passes_stage_depth(self) -> None:
        assert _passes_stage({"synthesis_depth": 1}, "synthesis_depth", 2) is True
        assert _passes_stage({"synthesis_depth": 3}, "synthesis_depth", 2) is False

    def test_passes_stage_grounding(self) -> None:
        assert _passes_stage({"combined_grounding_score": 0.8}, "combined_grounding_score", 0.75) is True
        assert _passes_stage({"combined_grounding_score": 0.5}, "combined_grounding_score", 0.75) is False


class TestReportingCLI:
    def test_report_command_help(self) -> None:
        from click.testing import CliRunner

        from aurelius.__main__ import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--help"])
        assert result.exit_code == 0
        assert "--top" in result.output
        assert "--dft" in result.output
        assert "--summary" in result.output

    def test_report_command_has_skip_dft_flag(self) -> None:
        from click.testing import CliRunner

        from aurelius.__main__ import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["report", "--help"])
        assert result.exit_code == 0
        assert "--skip-dft" in result.output


class TestDFTCascadeGate:
    """Tests for the mandatory DFT geometry-optimization cascade gate."""

    def test_dft_grounding_score_defaulted_in_build_candidate(self) -> None:
        """_build_candidate with skip_dft=True gives dft_grounding_score=1.0."""
        engine = ReportingEngine()
        result = _make_result()
        cand = engine._build_candidate(result, [], skip_dft=True)
        assert cand is not None
        assert cand["dft_grounding_score"] == 1.0
        assert cand["dft_method"] == "skipped (--skip-dft)"

    def test_dft_gate_rejects_low_grounding_score(self) -> None:
        """Candidates with dft_grounding_score below 0.80 are rejected by cascade."""
        engine = ReportingEngine()
        result = _make_result()
        cand = engine._build_candidate(result, [], skip_dft=True)
        assert cand is not None
        # Simulate a failed DFT optimization
        cand["dft_grounding_score"] = 0.5
        selected, rej = engine._apply_cascade([cand], skip_dft=False)
        assert len(selected) == 0
        assert rej["dft_grounding_score"] == 1

    def test_dft_gate_passes_high_grounding_score(self) -> None:
        """Candidates with dft_grounding_score >= 0.80 pass the DFT gate."""
        engine = ReportingEngine()
        result = _make_result()
        cand = engine._build_candidate(result, [], skip_dft=True)
        assert cand is not None
        cand["dft_grounding_score"] = 1.0
        selected, rej = engine._apply_cascade([cand], skip_dft=False)
        assert len(selected) == 1
        assert rej.get("dft_grounding_score", 0) == 0

    def test_skip_dft_excludes_dft_stage(self) -> None:
        """When skip_dft=True, the DFT stage is not in the rejection log."""
        engine = ReportingEngine()
        result = _make_result()
        cand = engine._build_candidate(result, [], skip_dft=True)
        assert cand is not None
        selected, rej = engine._apply_cascade([cand], skip_dft=True)
        assert len(selected) == 1
        assert "dft_grounding_score" not in rej

    def test_cascade_with_skip_dft_rejects_low_grounding_even_if_dft_skipped(self) -> None:
        """When skip_dft=True but grounding is low, candidate is still rejected at grounding stage."""
        engine = ReportingEngine()
        result = _make_result(grounding=0.3, domain=0.96, depth=1, novelty=0.4)
        cand = engine._build_candidate(result, [], skip_dft=True)
        assert cand is not None
        selected, _rej = engine._apply_cascade([cand], skip_dft=True)
        assert len(selected) == 0

    def test_dft_geometry_optimize_returns_grounding_score(self) -> None:
        """dft_geometry_optimize returns a dict with dft_grounding_score."""
        from rdkit import Chem

        from aurelius.scoring.oracle.dft_validator import dft_geometry_optimize

        mol = Chem.MolFromSmiles("C1COC(=O)O1")
        assert mol is not None
        result = dft_geometry_optimize(mol, cache_path="/tmp/test_dft_geom_cache.json")
        assert "dft_grounding_score" in result
        assert isinstance(result["dft_grounding_score"], float)
        assert 0.0 <= result["dft_grounding_score"] <= 1.0
        assert "dft_final_energy_eV" in result
        assert "dft_method" in result

    def test_dft_geometry_optimize_cached(self, monkeypatch, tmp_path) -> None:
        """dft_geometry_optimize caches results by SMILES and does not recompute."""
        from rdkit import Chem

        from aurelius.scoring.oracle.dft_validator import dft_geometry_optimize

        cache_file = str(tmp_path / "dft_cache.json")
        # Clear any in-memory cache
        from aurelius.scoring.oracle.dft_validator import _DFT_GEOM_CACHE
        _DFT_GEOM_CACHE.clear()

        mol = Chem.MolFromSmiles("CO")
        assert mol is not None

        calls = {"n": 0}

        dft_geometry_optimize.__globals__.get("_find_xtb_binary", lambda: None)

        def counting_optimize(m, cache_path=cache_file, timeout=300):
            calls["n"] += 1
            # Simulate: first call computes, subsequent calls use cache
            smiles = Chem.MolToSmiles(m)
            if smiles not in _DFT_GEOM_CACHE:
                return {
                    "dft_grounding_score": 1.0,
                    "dft_final_energy_eV": -15.5,
                    "dft_method": "xTB GFN2-xTB geometry optimization",
                }
            return dict(_DFT_GEOM_CACHE[smiles])

        monkeypatch.setattr(
            "aurelius.scoring.oracle.dft_validator._find_xtb_binary", lambda: None
        )

        r1 = dft_geometry_optimize(mol, cache_path=cache_file)
        r2 = dft_geometry_optimize(mol, cache_path=cache_file)
        assert r1 == r2

    def test_parse_xtb_opt_convergence_converged(self) -> None:
        """_parse_xtb_opt_convergence detects convergence from xTB output."""
        from aurelius.scoring.oracle.dft_validator import _parse_xtb_opt_convergence

        output = "Some header\n   ***       Fully optimized       ***\n"
        converged, energy = _parse_xtb_opt_convergence(output)
        assert converged is True

    def test_parse_xtb_opt_convergence_not_converged(self) -> None:
        """_parse_xtb_opt_convergence returns False when no convergence marker."""
        from aurelius.scoring.oracle.dft_validator import _parse_xtb_opt_convergence

        output = "Some header\nOptimization failed\n"
        converged, _energy = _parse_xtb_opt_convergence(output)
        assert converged is False



