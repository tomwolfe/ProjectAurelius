"""Wet-lab experimental result ingestion (ADR-2026-08-07-11).

Covers the schema contract, the validation rules that protect the
calibration set, and the round trip from file to FeedbackController.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from aurelius.agent.experimental_ingestion import (
    SCHEMA_PATH,
    IngestionReport,
    ingest_experimental_results,
    load_records,
    validate_record,
)


def _record(**overrides) -> dict:
    base = {
        "smiles": "COC(=O)OC",
        "name": "DMC",
        "measured_property": "viscosity_cP",
        "value": 0.59,
        "units": "cP",
        "temperature_K": 298.15,
        "method": "falling-ball viscometry",
    }
    base.update(overrides)
    return base


class TestSchema:
    def test_schema_file_is_valid_json(self):
        with open(SCHEMA_PATH) as f:
            schema = json.load(f)
        assert schema["type"] == "object"
        assert "measurements" in schema["properties"]

    def test_schema_requires_provenance_fields(self):
        with open(SCHEMA_PATH) as f:
            schema = json.load(f)
        required = schema["definitions"]["measurement"]["required"]
        for field in ("smiles", "measured_property", "value", "units",
                      "temperature_K", "method"):
            assert field in required, f"{field} must be required by the schema"


class TestValidation:
    def test_valid_record_accepted(self):
        canonical, reason = validate_record(_record())
        assert canonical is not None, reason
        assert canonical["smiles"] == "COC(=O)OC"

    def test_smiles_is_canonicalised(self):
        canonical, _ = validate_record(_record(smiles="O=C(OC)OC"))
        assert canonical is not None
        assert canonical["smiles"] == "COC(=O)OC"

    def test_unparseable_smiles_rejected(self):
        canonical, reason = validate_record(_record(smiles="not_a_molecule["))
        assert canonical is None
        assert "unparseable" in reason.lower()

    def test_missing_required_field_rejected(self):
        record = _record()
        del record["temperature_K"]
        canonical, reason = validate_record(record)
        assert canonical is None
        assert "temperature_K" in reason

    def test_wrong_units_rejected_not_converted(self):
        """A Pa.s viscosity must be rejected, never silently converted.

        Accepting it as cP would introduce a 1000x error into the
        calibration set, which is invisible downstream.
        """
        canonical, reason = validate_record(_record(units="Pa.s", value=0.00059))
        assert canonical is None
        assert "units" in reason.lower()

    def test_implausible_value_rejected(self):
        canonical, reason = validate_record(_record(value=1e6))
        assert canonical is None
        assert "plausible" in reason.lower()

    def test_unknown_property_rejected(self):
        canonical, reason = validate_record(_record(measured_property="vibes"))
        assert canonical is None
        assert "unknown measured_property" in reason

    def test_nonpositive_temperature_rejected(self):
        canonical, reason = validate_record(_record(temperature_K=0))
        assert canonical is None
        assert "positive" in reason.lower()

    def test_dimensionless_dielectric_accepted(self):
        canonical, reason = validate_record(_record(
            measured_property="dielectric_constant", value=3.11, units="",
        ))
        assert canonical is not None, reason


class TestFileLoading:
    def test_load_json_envelope(self, tmp_path: Path):
        path = tmp_path / "results.json"
        path.write_text(json.dumps({"source": "lab", "measurements": [_record()]}))
        assert len(load_records(str(path))) == 1

    def test_load_bare_json_list(self, tmp_path: Path):
        path = tmp_path / "results.json"
        path.write_text(json.dumps([_record(), _record()]))
        assert len(load_records(str(path))) == 2

    def test_load_csv(self, tmp_path: Path):
        path = tmp_path / "results.csv"
        record = _record()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(record))
            writer.writeheader()
            writer.writerow(record)
        loaded = load_records(str(path))
        assert len(loaded) == 1
        assert loaded[0]["smiles"] == "COC(=O)OC"

    def test_malformed_json_rejected(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"nope": []}))
        with pytest.raises(ValueError, match="measurements"):
            load_records(str(path))


class TestIngestion:
    def test_round_trip_accept_and_reject(self, tmp_path: Path):
        path = tmp_path / "mixed.json"
        path.write_text(json.dumps({"measurements": [
            _record(),
            _record(smiles="!!!bad!!!"),
            _record(units="Pa.s"),
        ]}))
        report = ingest_experimental_results(str(path), trigger_refit=False)
        assert report.n_accepted == 1
        assert report.n_rejected == 2
        assert all(reason for _rec, reason in report.rejected)

    def test_off_reference_temperature_warns_but_accepts(self, tmp_path: Path):
        path = tmp_path / "hot.json"
        path.write_text(json.dumps({"measurements": [_record(temperature_K=333.15)]}))
        report = ingest_experimental_results(str(path), trigger_refit=False)
        assert report.n_accepted == 1
        assert any("K from the" in w for w in report.warnings)

    def test_bulk_only_measurements_warn_that_no_model_changes(self, tmp_path: Path):
        path = tmp_path / "bulk.json"
        path.write_text(json.dumps({"measurements": [_record()]}))
        report = ingest_experimental_results(str(path), trigger_refit=False)
        assert any("do not yet change any model" in w for w in report.warnings)

    def test_orbital_pair_reaches_feedback_controller(self, tmp_path: Path):
        from aurelius.agent.feedback import FeedbackController

        path = tmp_path / "orbitals.json"
        path.write_text(json.dumps({"measurements": [
            _record(measured_property="homo_eV", value=-7.6, units="eV",
                    method="photoelectron spectroscopy"),
            _record(measured_property="lumo_eV", value=-0.8, units="eV",
                    method="inverse photoemission"),
        ]}))
        controller = FeedbackController()
        report = ingest_experimental_results(
            str(path), controller=controller, trigger_refit=False
        )
        assert report.n_accepted == 2
        records = controller._state.records
        assert len(records) == 1
        assert records[0].experimental_homo == pytest.approx(-7.6)
        assert records[0].experimental_lumo == pytest.approx(-0.8)

    def test_unpaired_orbital_warns(self, tmp_path: Path):
        path = tmp_path / "homo_only.json"
        path.write_text(json.dumps({"measurements": [
            _record(measured_property="homo_eV", value=-7.6, units="eV"),
        ]}))
        report = ingest_experimental_results(str(path), trigger_refit=False)
        assert any("only one of HOMO/LUMO" in w for w in report.warnings)

    def test_report_serialises(self, tmp_path: Path):
        path = tmp_path / "r.json"
        path.write_text(json.dumps({"measurements": [_record()]}))
        report = ingest_experimental_results(str(path), trigger_refit=False)
        payload = json.loads(json.dumps(report.to_dict()))
        assert payload["n_accepted"] == 1


class TestCLI:
    def test_cli_command_registered(self):
        from click.testing import CliRunner

        from aurelius.__main__ import cli

        result = CliRunner().invoke(cli, ["ingest-experiment", "--help"])
        assert result.exit_code == 0
        assert "wet-lab" in result.output.lower()

    def test_cli_ingests_file(self, tmp_path: Path):
        from click.testing import CliRunner

        from aurelius.__main__ import cli

        path = tmp_path / "results.json"
        path.write_text(json.dumps({"measurements": [_record()]}))
        out = tmp_path / "report.json"
        result = CliRunner().invoke(
            cli, ["ingest-experiment", str(path), "--no-refit", "--output", str(out)]
        )
        assert result.exit_code == 0, result.output
        assert "Accepted: 1" in result.output
        assert json.loads(out.read_text())["n_accepted"] == 1

    def test_cli_exits_nonzero_when_everything_rejected(self, tmp_path: Path):
        from click.testing import CliRunner

        from aurelius.__main__ import cli

        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"measurements": [_record(units="Pa.s")]}))
        result = CliRunner().invoke(cli, ["ingest-experiment", str(path), "--no-refit"])
        assert result.exit_code == 1
        assert "REJECTED" in result.output


class TestIngestionReport:
    def test_counts_match_lists(self):
        report = IngestionReport()
        report.accepted.append(_record())
        report.rejected.append((_record(), "reason"))
        assert report.n_accepted == 1
        assert report.n_rejected == 1
