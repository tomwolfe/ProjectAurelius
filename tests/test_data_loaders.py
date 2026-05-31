"""Tests for the QM9 data loader.

Verifies that the bundled qm9_subset.csv is correctly parsed
and returns properly typed molecular data.
"""

from __future__ import annotations

import os

import pytest

from aurelius.data.loaders import load_qm9_homo_lumo_data


def test_load_returns_list_of_tuples() -> None:
    """The loader must return a list of (smiles, homo, lumo) tuples."""
    data = load_qm9_homo_lumo_data()
    assert isinstance(data, list)
    assert len(data) > 0
    for entry in data:
        assert isinstance(entry, tuple)
        assert len(entry) == 3
        smi, homo, lumo = entry
        assert isinstance(smi, str) and len(smi) > 0
        assert isinstance(homo, float)
        assert isinstance(lumo, float)


def test_load_returns_at_least_100_entries() -> None:
    """The bundled QM9 subset must contain at least 100 molecules."""
    data = load_qm9_homo_lumo_data()
    assert len(data) >= 100


def test_load_homo_lumo_ranges_physically_plausible() -> None:
    """HOMO and LUMO values must be physically meaningful."""
    data = load_qm9_homo_lumo_data()
    for _, homo, lumo in data:
        assert -20.0 <= homo <= 0.0, f"HOMO {homo} out of range"
        assert -10.0 <= lumo <= 10.0, f"LUMO {lumo} out of range"
        assert lumo > homo, f"LUMO {lumo} <= HOMO {homo}"


def test_load_skips_invalid_rows_gracefully(tmp_path) -> None:
    """Rows with missing or unparseable values must be skipped, not crash."""
    csv_content = (
        "smiles,homo,lumo,gap\n"
        "CCO,-7.22,2.13,9.35\n"
        ",,,\n"
        "CC=O,-6.91,,6.37\n"
        ",abc,def,\n"
        "CC,-9.21,2.83,12.04\n"
    )
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text(csv_content)
    data = load_qm9_homo_lumo_data(str(bad_csv))
    assert len(data) == 2
    assert data[0] == ("CCO", -7.22, 2.13)
    assert data[1] == ("CC", -9.21, 2.83)


def test_load_raises_on_missing_file() -> None:
    """A missing CSV file must raise RuntimeError."""
    with pytest.raises(RuntimeError, match="not found"):
        load_qm9_homo_lumo_data("/nonexistent/path.csv")


def test_load_raises_on_empty_data(tmp_path) -> None:
    """A CSV with only headers and no data rows must raise RuntimeError."""
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("smiles,homo,lumo,gap\n")
    with pytest.raises(RuntimeError, match="No valid HOMO/LUMO"):
        load_qm9_homo_lumo_data(str(empty_csv))


def test_load_from_custom_path(tmp_path) -> None:
    """The loader must accept an explicit csv_path argument."""
    csv_content = (
        "smiles,homo,lumo,gap\n"
        "CCO,-7.22,2.13,9.35\n"
    )
    custom_csv = tmp_path / "custom.csv"
    custom_csv.write_text(csv_content)
    data = load_qm9_homo_lumo_data(str(custom_csv))
    assert len(data) == 1
    assert data[0] == ("CCO", -7.22, 2.13)


def test_load_bundled_csv_is_deterministic() -> None:
    """Loading the bundled CSV twice must return identical results."""
    d1 = load_qm9_homo_lumo_data()
    d2 = load_qm9_homo_lumo_data()
    assert d1 == d2
