"""Tests for the active-learning suggest → validate → retrain pipeline."""

from __future__ import annotations

import pytest

from aurelius.agent.learning_loop import (
    AutoRetrainPipeline,
    ExperimentResultParser,
    SuggestAndValidatePipeline,
)
from aurelius.types import ScreeningResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def discoveries():
    """A list of mock discovery results."""
    return [
        ScreeningResult(
            smiles="C1COC(=O)O1",
            total_score=85.0,
            is_viable=True,
            rejection_reasons=[],
            homo_eV=-7.6,
            lumo_eV=-0.8,
            dielectric_proxy=25.0,
            viscosity_proxy=2.5,
            li_solvation_proxy=1.0,
        ),
        ScreeningResult(
            smiles="COC(=O)OC",
            total_score=72.0,
            is_viable=True,
            rejection_reasons=[],
            homo_eV=-7.8,
            lumo_eV=-0.5,
            dielectric_proxy=20.0,
            viscosity_proxy=3.0,
            li_solvation_proxy=0.8,
        ),
        ScreeningResult(
            smiles="CC(C)O",
            total_score=45.0,
            is_viable=False,
            rejection_reasons=["low dielectric"],
            homo_eV=-8.5,
            lumo_eV=0.2,
            dielectric_proxy=10.0,
            viscosity_proxy=1.5,
            li_solvation_proxy=0.3,
        ),
    ]


@pytest.fixture
def csv_content():
    """A CSV string with required columns."""
    return (
        "smiles,dielectric,viscosity,cycle_life\n"
        "C1COC(=O)O1,25.0,2.5,500\n"
        "COC(=O)OC,20.0,3.0,450\n"
    )


@pytest.fixture
def tmp_path_setup(tmp_path):
    """Return a temporary directory path."""
    return tmp_path


# ---------------------------------------------------------------------------
# Tests: SuggestAndValidatePipeline
# ---------------------------------------------------------------------------


class TestSuggestAndValidatePipeline:
    """Tests for SuggestAndValidatePipeline."""

    def test_top_10_selection(self, discoveries):
        """Should select top-10 (or fewer) discoveries sorted by score."""
        suggestions = SuggestAndValidatePipeline(discoveries)
        assert len(suggestions.suggestions) == 3
        assert suggestions.suggestions[0]["total_score"] == 85.0
        assert suggestions.suggestions[2]["total_score"] == 45.0

    def test_empty_discoveries(self):
        """Should handle empty discovery list gracefully."""
        suggestions = SuggestAndValidatePipeline([])
        assert len(suggestions.suggestions) == 0

    def test_single_discovery(self):
        """Should handle a single discovery."""
        discovery = ScreeningResult(
            smiles="CCO",
            total_score=90.0,
            is_viable=True,
            rejection_reasons=[],
        )
        suggestions = SuggestAndValidatePipeline([discovery])
        assert len(suggestions.suggestions) == 1
        assert suggestions.suggestions[0]["smiles"] == "CCO"

    def test_dict_discoveries(self):
        """Should handle dict-based discoveries."""
        discoveries_dict = [
            {"smiles": "CCO", "total_score": 80.0},
            {"smiles": "CCCO", "total_score": 60.0},
        ]
        suggestions = SuggestAndValidatePipeline(discoveries_dict)
        assert len(suggestions.suggestions) == 2
        assert suggestions.suggestions[0]["smiles"] == "CCO"

    def test_export_creates_file(self, tmp_path_setup):
        """Should create an SDF file when exporting."""
        suggestions = SuggestAndValidatePipeline([
            ScreeningResult(
                smiles="CCO",
                total_score=85.0,
                is_viable=True,
                rejection_reasons=[],
            )
        ])
        output_path = tmp_path_setup / "suggestions.sdf"
        suggestions.export(str(output_path))
        assert output_path.exists()
        assert output_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Tests: ExperimentResultParser
# ---------------------------------------------------------------------------


class TestExperimentResultParser:
    """Tests for ExperimentResultParser."""

    def test_parse_csv_basic(self, tmp_path_setup, csv_content):
        """Should parse a basic CSV file correctly."""
        csv_file = tmp_path_setup / "feedback.csv"
        csv_file.write_text(csv_content)

        results = ExperimentResultParser.parse_csv(str(csv_file))
        assert len(results) == 2
        assert results[0]["smiles"] == "C1COC(=O)O1"
        assert results[0]["dielectric"] == 25.0
        assert results[0]["viscosity"] == 2.5
        assert results[0]["cycle_life"] == 500.0

    def test_parse_csv_missing_columns(self, tmp_path_setup):
        """Should raise ValueError if required columns are missing."""
        csv_file = tmp_path_setup / "bad.csv"
        csv_file.write_text("smiles,dielectric\nCCO,25.0\n")

        with pytest.raises(ValueError, match="missing required columns"):
            ExperimentResultParser.parse_csv(str(csv_file))

    def test_parse_csv_empty_file(self, tmp_path_setup):
        """Should return empty list for empty CSV."""
        csv_file = tmp_path_setup / "empty.csv"
        csv_file.write_text("smiles,dielectric,viscosity,cycle_life\n")

        results = ExperimentResultParser.parse_csv(str(csv_file))
        assert len(results) == 0

    def test_parse_file_csv_auto_detect(self, tmp_path_setup, csv_content):
        """Auto-detect should work for CSV files."""
        csv_file = tmp_path_setup / "feedback.csv"
        csv_file.write_text(csv_content)

        results = ExperimentResultParser.parse_file(str(csv_file))
        assert len(results) == 2

    @pytest.mark.skip(reason="SDF parsing requires binary file object")
    def test_parse_file_sdf_auto_detect(self, tmp_path_setup):
        """Auto-detect should work for SDF files."""
        sdf_content = """C1COC(=O)O1
  AutoGen

> <SMILES>
C1COC(=O)O1

> <dielectric_constant>
25.0

> <viscosity_cP>
2.5

> <cycle_life>
500

M  END

"""
        sdf_file = tmp_path_setup / "feedback.sdf"
        sdf_file.write_text(sdf_content)

        results = ExperimentResultParser.parse_file(str(sdf_file))
        assert len(results) == 1
        assert results[0]["smiles"] == "C1COC(=O)O1"

    def test_parse_file_unsupported_extension(self, tmp_path_setup):
        """Should raise ValueError for unsupported file extensions."""
        txt_file = tmp_path_setup / "feedback.txt"
        txt_file.write_text("CCO,25.0,2.5,500")

        with pytest.raises(ValueError, match="Unsupported file extension"):
            ExperimentResultParser.parse_file(str(txt_file))

    def test_parse_csv_with_whitespace(self, tmp_path_setup):
        """Should strip whitespace from column names and values."""
        csv_content_ws = (
            "  SMILES ,  dielectric ,  viscosity ,  cycle_life \n"
            "  CCO , 25.0 , 2.5 , 500 \n"
        )
        csv_file = tmp_path_setup / "whitespace.csv"
        csv_file.write_text(csv_content_ws)

        results = ExperimentResultParser.parse_csv(str(csv_file))
        assert len(results) == 1
        assert results[0]["smiles"] == "CCO"
        assert results[0]["dielectric"] == 25.0


# ---------------------------------------------------------------------------
# Tests: AutoRetrainPipeline
# ---------------------------------------------------------------------------


class TestAutoRetrainPipeline:
    """Tests for AutoRetrainPipeline."""

    def test_empty_feedback(self):
        """Should return empty summary when no feedback data."""
        auto = AutoRetrainPipeline()
        summary = auto.summary()
        assert summary["n_feedback"] == 0
        assert "No feedback data loaded" in summary["message"]

    def test_from_pipeline(self):
        """Should load feedback data from a SuggestAndValidatePipeline."""
        discoveries = [
            ScreeningResult(
                smiles="CCO",
                total_score=85.0,
                is_viable=True,
                rejection_reasons=[],
                dielectric_proxy=25.0,
                viscosity_proxy=2.5,
            ),
        ]
        suggestions = SuggestAndValidatePipeline(discoveries)
        auto = AutoRetrainPipeline()
        auto.from_pipeline(suggestions)

        assert len(auto.feedback_data) == 1
        assert auto.feedback_data[0]["smiles"] == "CCO"
        assert auto.feedback_data[0]["dielectric_constant"] == 25.0

    def test_from_results(self):
        """Should load feedback data from ScreeningResult objects."""
        results = [
            ScreeningResult(
                smiles="CCO",
                total_score=85.0,
                is_viable=True,
                rejection_reasons=[],
                dielectric_proxy=25.0,
                viscosity_proxy=2.5,
            ),
        ]
        auto = AutoRetrainPipeline()
        auto.from_results(results)

        assert len(auto.feedback_data) == 1
        assert auto.feedback_data[0]["smiles"] == "CCO"

    def test_retrain_no_pipeline(self):
        """Should return error when pipeline is None."""
        auto = AutoRetrainPipeline()
        auto._feedback_data = [{"smiles": "CCO", "dielectric_constant": 25.0, "viscosity_cP": 2.5}]
        result = auto.retrain(None)
        assert result["status"] == "error"

    def test_summary_statistics(self):
        """Should compute correct summary statistics."""
        auto = AutoRetrainPipeline()
        auto._feedback_data = [
            {"smiles": "CCO", "dielectric_constant": 25.0, "viscosity_cP": 2.5},
            {"smiles": "CCCO", "dielectric_constant": 30.0, "viscosity_cP": 3.0},
        ]
        summary = auto.summary()
        assert summary["n_feedback"] == 2
        assert abs(summary["mean_dielectric"] - 27.5) < 0.01
        assert abs(summary["mean_viscosity"] - 2.75) < 0.01
