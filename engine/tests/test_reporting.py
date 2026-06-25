"""Tests for reporting module — Pareto front export and run summary generation."""

from __future__ import annotations

import json
from dataclasses import dataclass

from aurelius.agent.reporting import _export_pareto_front, _pareto_entries, _pareto_entry
from aurelius.agent.selection import extract_pareto_front
from aurelius.types import ScreeningResult


@dataclass
class MockResult:
    """Minimal mock for extract_pareto_front compatibility."""
    lumo_eV: float
    dielectric_proxy: float
    viscosity_proxy: float


class TestParetoEntry:
    """Unit tests for _pareto_entry."""

    def test_basic_entry(self):
        result = ScreeningResult(
            smiles="CCO",
            total_score=85.0,
            is_viable=True,
            rejection_reasons=[],
            lumo_eV=-1.5,
            dielectric_proxy=12.0,
            viscosity_proxy=0.5,
        )
        entry = _pareto_entry(result, {})
        assert entry["smiles"] == "CCO"
        assert entry["total_score"] == 85.0
        assert entry["lumo_eV"] == -1.5
        assert entry["dielectric_proxy"] == 12.0
        assert entry["viscosity_proxy"] == 0.5

    def test_entry_with_uq_data(self):
        result = ScreeningResult(
            smiles="CCO",
            total_score=85.0,
            is_viable=True,
            rejection_reasons=[],
            lumo_eV=-1.5,
            dielectric_proxy=12.0,
            viscosity_proxy=0.5,
        )
        uq_data = {
            "CCO": {
                "diel_std": 0.1,
                "visc_std": 0.2,
                "uncertainty_weighted_score": 95.0,
            }
        }
        entry = _pareto_entry(result, uq_data)
        assert entry["diel_std"] == 0.1
        assert entry["visc_std"] == 0.2
        assert entry["uncertainty_weighted_score"] == 95.0

    def test_entry_missing_properties(self):
        result = ScreeningResult(
            smiles="CCO",
            total_score=85.0,
            is_viable=True,
            rejection_reasons=[],
            lumo_eV=None,
            dielectric_proxy=None,
            viscosity_proxy=None,
        )
        entry = _pareto_entry(result, {})
        assert entry["lumo_eV"] is None
        assert entry["dielectric_proxy"] is None
        assert entry["viscosity_proxy"] is None


class TestParetoEntries:
    """Unit tests for _pareto_entries."""

    def test_multiple_entries(self):
        results = [
            ScreeningResult(
                smiles=f"CCO{i}",
                total_score=float(80 + i),
                is_viable=True,
                rejection_reasons=[],
                lumo_eV=-1.5,
                dielectric_proxy=10.0 + i,
                viscosity_proxy=0.5 + i * 0.1,
            )
            for i in range(3)
        ]
        entries = _pareto_entries(results, {})
        assert len(entries) == 3
        assert entries[0]["smiles"] == "CCO0"
        assert entries[1]["smiles"] == "CCO1"
        assert entries[2]["smiles"] == "CCO2"

    def test_empty_list(self):
        assert _pareto_entries([], {}) == []


class TestExportParetoFront:
    """Unit tests for _export_pareto_front."""

    def test_writes_json_file(self, tmp_path):
        entries = [
            {
                "smiles": "CCO",
                "total_score": 85.0,
                "lumo_eV": -1.5,
                "dielectric_proxy": 12.0,
                "viscosity_proxy": 0.5,
            }
        ]
        _export_pareto_front(entries, output_dir=str(tmp_path))
        output_file = tmp_path / "pareto_front.json"
        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["smiles"] == "CCO"

    def test_empty_pareto(self, tmp_path):
        _export_pareto_front([], output_dir=str(tmp_path))
        output_file = tmp_path / "pareto_front.json"
        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
        assert data == []


class TestExtractParetoFrontIntegration:
    """Integration tests verifying Pareto dominance logic with ScreeningResult."""

    def test_dominated_solutions_removed(self):
        """A solution dominated on all objectives should be excluded."""
        results = [
            ScreeningResult(
                smiles="dominator",
                total_score=90.0,
                is_viable=True,
                rejection_reasons=[],
                lumo_eV=-1.0,
                dielectric_proxy=15.0,
                viscosity_proxy=0.3,
            ),
            ScreeningResult(
                smiles="dominated",
                total_score=70.0,
                is_viable=True,
                rejection_reasons=[],
                lumo_eV=-2.0,
                dielectric_proxy=5.0,
                viscosity_proxy=2.0,
            ),
        ]
        front = extract_pareto_front(results)
        smiles_in_front = {r.smiles for r in front}
        assert "dominator" in smiles_in_front
        assert "dominated" not in smiles_in_front

    def test_non_dominated_both_retained(self):
        """Solutions with trade-offs should both be Pareto-optimal."""
        results = [
            ScreeningResult(
                smiles="high_lumo",
                total_score=85.0,
                is_viable=True,
                rejection_reasons=[],
                lumo_eV=-1.0,
                dielectric_proxy=5.0,
                viscosity_proxy=2.0,
            ),
            ScreeningResult(
                smiles="low_viscosity",
                total_score=80.0,
                is_viable=True,
                rejection_reasons=[],
                lumo_eV=-3.0,
                dielectric_proxy=12.0,
                viscosity_proxy=0.3,
            ),
        ]
        front = extract_pareto_front(results)
        smiles_in_front = {r.smiles for r in front}
        assert len(front) == 2
        assert "high_lumo" in smiles_in_front
        assert "low_viscosity" in smiles_in_front

    def test_pareto_entries_via_extract(self):
        """End-to-end: extract_pareto_front -> _pareto_entries."""
        results = [
            ScreeningResult(
                smiles="cco",
                total_score=85.0,
                is_viable=True,
                rejection_reasons=[],
                lumo_eV=-1.5,
                dielectric_proxy=12.0,
                viscosity_proxy=0.5,
            ),
        ]
        front = extract_pareto_front(results)
        entries = _pareto_entries(front, {})
        assert len(entries) == 1
        assert entries[0]["smiles"] == "cco"
        assert entries[0]["lumo_eV"] == -1.5
        assert entries[0]["dielectric_proxy"] == 12.0
