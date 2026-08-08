"""Tests for the DFTValidator re-ranking gate (Feature 1: v11.0).

Verifies that:
1. ORCA output parsing extracts HOMO/LUMO correctly
2. ORCA input file is well-formed for the wB97X-D3/def2-SVP method
3. Spearman correlation handles degenerate and short inputs
4. The validator degrades gracefully when ORCA is unavailable
5. On-disk caching avoids recomputation
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from aurelius.scoring.oracle.dft_validator import (
    DFTValidator,
    _build_orca_input,
    _parse_orca_output,
    has_orca,
    spearman_correlation,
)

_SAMPLE_OUTPUT = """\
--------------------------------------------
ORBITAL ENERGIES
--------------------------------------------
   NO   OCC          E(Eh)            E(eV)
   0 :  2.0000    -20.075       -546.270
   1 :  2.0000    -11.342       -308.628
   2 :  2.0000     -1.025        -27.897
   3 :  0.0000      0.250          6.803
   4 :  0.0000      0.450         12.245
--------------------------------------------
"""


def test_parse_orca_output_homo_lumo() -> None:
    result = _parse_orca_output(_SAMPLE_OUTPUT)
    assert result is not None
    assert result["homo_eV"] == pytest.approx(-27.897, abs=1e-3)
    assert result["lumo_eV"] == pytest.approx(6.803, abs=1e-3)


def test_parse_orca_output_missing_block() -> None:
    assert _parse_orca_output("no orbital energies here") is None


def test_build_orca_input_method() -> None:
    inp = _build_orca_input()
    assert "wB97X-D3" in inp
    assert "def2-SVP" in inp
    assert "SP" in inp
    assert "xyzfile" in inp


def test_spearman_correlation_monotonic() -> None:
    rho, p = spearman_correlation([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
    assert rho == pytest.approx(1.0, abs=1e-6)


def test_spearman_correlation_degenerate() -> None:
    rho, _p = spearman_correlation([5.0, 5.0, 5.0], [1.0, 2.0, 3.0])
    assert rho == 0.0


def test_spearman_correlation_short() -> None:
    rho, p = spearman_correlation([1.0, 2.0], [3.0, 4.0])
    assert rho == 0.0 and p == 1.0


def test_validator_degrades_when_orca_unavailable() -> None:
    if has_orca():
        pytest.skip("ORCA installed — graceful-degradation path not testable")
    validator = DFTValidator(cache_path="does-not-exist/dft_cache.json")
    mol = Chem.MolFromSmiles("C1COC(=O)O1")
    assert mol is not None
    assert validator.compute(mol) is None


def test_validator_caches_by_smiles(tmp_path, monkeypatch) -> None:
    cache_file = tmp_path / "dft_cache.json"
    validator = DFTValidator(cache_path=str(cache_file))
    mol = Chem.MolFromSmiles("C1COC(=O)O1")
    assert mol is not None

    calls = {"n": 0}

    def fake_run(mol):
        calls["n"] += 1
        return {"homo_eV": -7.5, "lumo_eV": 0.5}

    monkeypatch.setattr(validator, "_run_orca", fake_run)
    r1 = validator.compute(mol)
    r2 = validator.compute(mol)
    assert r1 == r2
    assert calls["n"] == 1, "Cache must avoid recomputation"
    assert cache_file.exists(), "Cache must be written to disk"


def test_validate_ranking_metrics(monkeypatch) -> None:
    validator = DFTValidator(cache_path="/tmp/unused_dft_cache.json")
    mols = [Chem.MolFromSmiles(s) for s in
            ["C1COC(=O)O1", "COCCOC", "CS(=O)(=O)C", "CC#N"]]
    mols = [m for m in mols if m is not None]

    fake_results = {
        Chem.MolToSmiles(Chem.MolFromSmiles("C1COC(=O)O1")): {"homo_eV": -7.5, "lumo_eV": 0.5},
        Chem.MolToSmiles(Chem.MolFromSmiles("COCCOC")): {"homo_eV": -8.0, "lumo_eV": 1.0},
        Chem.MolToSmiles(Chem.MolFromSmiles("CS(=O)(=O)C")): {"homo_eV": -8.5, "lumo_eV": -1.0},
        Chem.MolToSmiles(Chem.MolFromSmiles("CC#N")): {"homo_eV": -9.0, "lumo_eV": -0.5},
    }

    def fake_compute(mol):
        smi = Chem.MolToSmiles(mol)
        return dict(fake_results[smi])

    monkeypatch.setattr(validator, "compute", fake_compute)
    metrics = validator.validate_ranking([90.0, 80.0, 70.0, 60.0], mols)
    assert metrics["n_validated"] == 4
    assert -1.0 <= metrics["rho_composite"] <= 1.0


def test_orca_method_exposed() -> None:
    assert "def2-SVP" in DFTValidator.METHOD
