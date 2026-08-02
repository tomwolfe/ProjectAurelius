"""Tests for the experimental feedback ingestion pipeline."""

from __future__ import annotations

import pytest

from aurelius.feedback.parser import (
    ingest_feedback,
    parse_experimental_csv,
    parse_experimental_sdf,
    parse_feedback_file,
    validate_feedback_schema,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def csv_content():
    """A CSV string with required columns."""
    return (
        "smiles,dielectric,viscosity,cycle_life\n"
        "C1COC(=O)O1,25.0,2.5,500\n"
        "COC(=O)OC,20.0,3.0,450\n"
    )


@pytest.fixture
def valid_entries():
    """List of valid feedback entries."""
    return [
        {"smiles": "C1COC(=O)O1", "dielectric": 25.0, "viscosity": 2.5, "cycle_life": 500.0},
        {"smiles": "COC(=O)OC", "dielectric": 20.0, "viscosity": 3.0, "cycle_life": 450.0},
    ]


# ---------------------------------------------------------------------------
# Tests: parse_experimental_csv
# ---------------------------------------------------------------------------


class TestParseExperimentalCsv:
    """Tests for parse_experimental_csv."""

    def test_parse_basic(self, tmp_path, csv_content):
        """Should parse a basic CSV file correctly."""
        csv_file = tmp_path / "feedback.csv"
        csv_file.write_text(csv_content)

        results = parse_experimental_csv(str(csv_file))
        assert len(results) == 2
        assert results[0]["smiles"] == "C1COC(=O)O1"
        assert results[0]["dielectric"] == 25.0
        assert results[0]["viscosity"] == 2.5
        assert results[0]["cycle_life"] == 500.0

    def test_parse_missing_columns(self, tmp_path):
        """Should raise ValueError if required columns are missing."""
        csv_file = tmp_path / "bad.csv"
        csv_file.write_text("smiles,dielectric\nCCO,25.0\n")

        with pytest.raises(ValueError, match="missing required columns"):
            parse_experimental_csv(str(csv_file))

    def test_parse_empty_file(self, tmp_path):
        """Should return empty list for empty CSV."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("smiles,dielectric,viscosity,cycle_life\n")

        results = parse_experimental_csv(str(csv_file))
        assert len(results) == 0

    def test_parse_with_whitespace(self, tmp_path):
        """Should strip whitespace from column names and values."""
        csv_content_ws = (
            "  SMILES ,  dielectric ,  viscosity ,  cycle_life \n"
            "  CCO , 25.0 , 2.5 , 500 \n"
        )
        csv_file = tmp_path / "whitespace.csv"
        csv_file.write_text(csv_content_ws)

        results = parse_experimental_csv(str(csv_file))
        assert len(results) == 1
        assert results[0]["smiles"] == "CCO"
        assert results[0]["dielectric"] == 25.0

    def test_parse_invalid_values(self, tmp_path):
        """Should convert non-numeric values to 0.0."""
        csv_file = tmp_path / "invalid.csv"
        csv_file.write_text(
            "smiles,dielectric,viscosity,cycle_life\n"
            "CCO,invalid,2.5,500\n"
        )

        results = parse_experimental_csv(str(csv_file))
        assert len(results) == 1
        assert results[0]["dielectric"] == 0.0
        assert results[0]["viscosity"] == 2.5

    def test_parse_empty_smiles(self, tmp_path):
        """Should skip entries with empty SMILES."""
        csv_file = tmp_path / "empty_smiles.csv"
        csv_file.write_text(
            "smiles,dielectric,viscosity,cycle_life\n"
            ",25.0,2.5,500\n"
        )

        results = parse_experimental_csv(str(csv_file))
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Tests: validate_feedback_schema
# ---------------------------------------------------------------------------


class TestValidateFeedbackSchema:
    """Tests for validate_feedback_schema."""

    def test_valid_entry(self):
        """Should return True for a valid entry."""
        entry = {
            "smiles": "C1COC(=O)O1",
            "dielectric": 25.0,
            "viscosity": 2.5,
            "cycle_life": 500.0,
        }
        is_valid, errors = validate_feedback_schema(entry)
        assert is_valid
        assert len(errors) == 0

    def test_missing_smiles(self):
        """Should report missing 'smiles' field."""
        entry = {
            "dielectric": 25.0,
            "viscosity": 2.5,
            "cycle_life": 500.0,
        }
        is_valid, errors = validate_feedback_schema(entry)
        assert not is_valid
        assert any("smiles" in err for err in errors)

    def test_negative_dielectric(self):
        """Should report negative dielectric value."""
        entry = {
            "smiles": "CCO",
            "dielectric": -1.0,
            "viscosity": 2.5,
            "cycle_life": 500.0,
        }
        is_valid, errors = validate_feedback_schema(entry)
        assert not is_valid
        assert any("Negative value" in err for err in errors)

    def test_non_numeric_dielectric(self):
        """Should report non-numeric dielectric value."""
        entry = {
            "smiles": "CCO",
            "dielectric": "high",
            "viscosity": 2.5,
            "cycle_life": 500.0,
        }
        is_valid, errors = validate_feedback_schema(entry)
        assert not is_valid
        assert any("Non-numeric" in err for err in errors)

    def test_empty_smiles(self):
        """Should report empty SMILES."""
        entry = {
            "smiles": "",
            "dielectric": 25.0,
            "viscosity": 2.5,
            "cycle_life": 500.0,
        }
        is_valid, errors = validate_feedback_schema(entry)
        assert not is_valid


# ---------------------------------------------------------------------------
# Tests: parse_feedback_file
# ---------------------------------------------------------------------------


class TestParseFeedbackFile:
    """Tests for parse_feedback_file auto-detection."""

    def test_auto_detect_csv(self, tmp_path, csv_content):
        """Should auto-detect CSV files."""
        csv_file = tmp_path / "feedback.csv"
        csv_file.write_text(csv_content)

        results = parse_feedback_file(str(csv_file))
        assert len(results) == 2

    def test_auto_detect_sdf(self, tmp_path):
        """Should auto-detect SDF files."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        sdf_file = tmp_path / "feedback.sdf"
        mol = Chem.MolFromSmiles("C1COC(=O)O1")
        AllChem.Compute2DCoords(mol)
        mol.SetProp("SMILES", "C1COC(=O)O1")
        mol.SetProp("dielectric_constant", "25.0")
        mol.SetProp("viscosity_cP", "2.5")
        mol.SetProp("cycle_life", "500")

        writer = Chem.SDWriter(str(sdf_file))
        writer.write(mol)
        writer.close()

        results = parse_feedback_file(str(sdf_file))
        assert len(results) == 1
        assert results[0]["smiles"] == "C1COC(=O)O1"

    def test_unsupported_extension(self, tmp_path):
        """Should raise ValueError for unsupported file extensions."""
        txt_file = tmp_path / "feedback.txt"
        txt_file.write_text("CCO,25.0,2.5,500")

        with pytest.raises(ValueError, match="Unsupported file extension"):
            parse_feedback_file(str(txt_file))


# ---------------------------------------------------------------------------
# Tests: ingest_feedback
# ---------------------------------------------------------------------------


class TestIngestFeedback:
    """Tests for ingest_feedback."""

    def test_ingest_valid_entries(self, valid_entries):
        """Should return summary statistics for valid entries."""
        summary = ingest_feedback(valid_entries)
        assert summary["n_valid"] == 2
        assert summary["n_invalid"] == 0
        assert summary["mean_dielectric"] == 22.5
        assert summary["mean_viscosity"] == 2.75

    def test_ingest_invalid_entries(self):
        """Should report invalid entries and skip them."""
        invalid = [
            {"dielectric": 25.0, "viscosity": 2.5, "cycle_life": 500.0},
        ]
        summary = ingest_feedback(invalid)
        assert summary["n_valid"] == 0
        assert summary["n_invalid"] == 1

    def test_ingest_empty_list(self):
        """Should return empty summary for empty list."""
        summary = ingest_feedback([])
        assert summary["n_valid"] == 0
        assert summary["n_invalid"] == 0
        assert summary["mean_dielectric"] == 0.0
        assert summary["mean_viscosity"] == 0.0

    def test_ingest_with_pipeline(self):
        """Should call pipeline retrain when pipeline is provided."""
        class MockOracle:
            def __init__(self):
                self.received_data: list = []
            def append_empirical_data(self, data):
                self.received_data.extend(data)

        class MockPipeline:
            _oracle = MockOracle()

        feedback = [
            {"smiles": "CCO", "dielectric": 25.0, "viscosity": 2.5, "cycle_life": 500.0},
        ]
        summary = ingest_feedback(feedback, pipeline=MockPipeline())
        assert summary["n_valid"] == 1
        assert summary["retrained"] is True
