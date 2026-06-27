"""Tests for the ``aurelius tune`` CLI command.

Verifies that the ``tune`` command:
- Accepts CSV input and produces valid JSON output
- Contains ``tom_parameters`` in the output kernel
- Handles invalid or empty CSV files gracefully
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from aurelius.__main__ import cli


def _make_csv(rows: list[list[str]]) -> str:
    """Create a temporary CSV file with the given rows (including header)."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    try:
        with open(fd, "w", newline="") as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)
        return path
    except Exception:
        import os
        os.close(fd)
        os.unlink(path)
        raise


def _make_kernel_output(tmp_path: Path, command_output: str) -> dict[str, Any]:
    """Parse the JSON output from the CLI command."""
    return json.loads(command_output)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_CSV_HEADER = ["smiles", "property", "value"]
MINIMAL_CSV_ROWS = [
    ["CCO", "homo", "-9.8"],
    ["CCO", "lumo", "-2.1"],
    ["CCC", "homo", "-9.5"],
    ["CCC", "lumo", "-1.8"],
    ["CCCO", "homo", "-9.2"],
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_tune_cmd_produces_valid_json(tmp_path: Path) -> None:
    """The ``tune`` command must write a valid JSON file with ``tom_parameters``."""
    csv_path = tmp_path / "training.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(MINIMAL_CSV_HEADER)
        for row in MINIMAL_CSV_ROWS:
            writer.writerow(row)

    output_path = tmp_path / "kernel.json"
    runner = CliRunner()
    result = runner.invoke(cli, ["tune", str(csv_path), "--output", str(output_path)])
    assert result.exit_code == 0, f"CLI exited {result.exit_code}: {result.output}"

    assert output_path.exists(), "Output file was not created"
    data = json.loads(output_path.read_text())
    assert "tom_parameters" in data
    assert "homo_offset" in data["tom_parameters"]
    assert "lumo_offset" in data["tom_parameters"]
    assert "gc_scale" in data["tom_parameters"]


def test_tune_cmd_with_explicit_output_path(tmp_path: Path) -> None:
    """The ``tune`` command must write output to the specified path."""
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(MINIMAL_CSV_HEADER)
        for row in MINIMAL_CSV_ROWS:
            writer.writerow(row)

    output_path = tmp_path / "out.json"
    runner = CliRunner()
    result = runner.invoke(cli, ["tune", str(csv_path), "--output", str(output_path)])
    assert result.exit_code == 0
    assert output_path.exists()


def test_tune_cmd_with_custom_max_iter(tmp_path: Path) -> None:
    """The ``tune`` command must respect the ``--max-iter`` option."""
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(MINIMAL_CSV_HEADER)
        for row in MINIMAL_CSV_ROWS:
            writer.writerow(row)

    output_path = tmp_path / "out.json"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["tune", str(csv_path), "--output", str(output_path), "--max-iter", "50"],
    )
    assert result.exit_code == 0
    data = json.loads(output_path.read_text())
    assert "tom_parameters" in data


def test_tune_cmd_output_contains_validation_metrics(tmp_path: Path) -> None:
    """The output kernel must contain ``validation_metrics`` with expected keys."""
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(MINIMAL_CSV_HEADER)
        for row in MINIMAL_CSV_ROWS:
            writer.writerow(row)

    output_path = tmp_path / "out.json"
    runner = CliRunner()
    result = runner.invoke(cli, ["tune", str(csv_path), "--output", str(output_path)])
    assert result.exit_code == 0

    data = json.loads(output_path.read_text())
    metrics = data.get("validation_metrics", {})
    assert "spearman_rho" in metrics
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "n_training" in metrics


def test_tune_cmd_output_contains_domain_boundary(tmp_path: Path) -> None:
    """The output kernel must contain ``domain_boundary``."""
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(MINIMAL_CSV_HEADER)
        for row in MINIMAL_CSV_ROWS:
            writer.writerow(row)

    output_path = tmp_path / "out.json"
    runner = CliRunner()
    result = runner.invoke(cli, ["tune", str(csv_path), "--output", str(output_path)])
    assert result.exit_code == 0

    data = json.loads(output_path.read_text())
    assert "domain_boundary" in data


def test_tune_cmd_output_contains_gc_fragments(tmp_path: Path) -> None:
    """The output kernel must contain ``gc_fragments`` as a non-empty list."""
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(MINIMAL_CSV_HEADER)
        for row in MINIMAL_CSV_ROWS:
            writer.writerow(row)

    output_path = tmp_path / "out.json"
    runner = CliRunner()
    result = runner.invoke(cli, ["tune", str(csv_path), "--output", str(output_path)])
    assert result.exit_code == 0

    data = json.loads(output_path.read_text())
    fragments = data.get("gc_fragments", [])
    assert isinstance(fragments, list)
    assert len(fragments) > 0


def test_tune_cmd_output_contains_version(tmp_path: Path) -> None:
    """The output kernel must contain a ``version`` field."""
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(MINIMAL_CSV_HEADER)
        for row in MINIMAL_CSV_ROWS:
            writer.writerow(row)

    output_path = tmp_path / "out.json"
    runner = CliRunner()
    result = runner.invoke(cli, ["tune", str(csv_path), "--output", str(output_path)])
    assert result.exit_code == 0

    data = json.loads(output_path.read_text())
    assert "version" in data
    assert isinstance(data["version"], str)


def test_tune_cmd_requires_minimum_data(tmp_path: Path) -> None:
    """The ``tune`` command must fail with fewer than 3 data points."""
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(MINIMAL_CSV_HEADER)
        writer.writerow(["CCO", "homo", "-9.8"])
        writer.writerow(["CCC", "lumo", "-1.8"])

    output_path = tmp_path / "out.json"
    runner = CliRunner()
    result = runner.invoke(cli, ["tune", str(csv_path), "--output", str(output_path)])
    assert result.exit_code != 0
    assert "Error" in result.output or "error" in result.output.lower()


def test_tune_cmd_nonexistent_csv() -> None:
    """The ``tune`` command must fail when the CSV path does not exist."""
    runner = CliRunner()
    result = runner.invoke(cli, ["tune", "/nonexistent/path.csv"])
    assert result.exit_code != 0
    assert "Error" in result.output or "error" in result.output.lower()


def test_tune_cmd_default_output_path(tmp_path: Path) -> None:
    """Without --output, the CLI must write to the default ``aurelius_kernel.json``."""
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(MINIMAL_CSV_HEADER)
        for row in MINIMAL_CSV_ROWS:
            writer.writerow(row)

    runner = CliRunner()
    # Change CWD temporarily so the default output lands in tmp_path
    import os
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = runner.invoke(cli, ["tune", str(csv_path)])
        assert result.exit_code == 0
        default_path = tmp_path / "aurelius_kernel.json"
        assert default_path.exists()
    finally:
        os.chdir(old_cwd)
