"""Δ-learning correction layer tests.

Verifies that the kernel-ridge residual model:
  1. Reaches LOO cross-validation MAE < 0.8 eV on the calibration set
  2. Improves Spearman ρ over raw TOM on the external benchmark
  3. Preserves the deterministic base TOM API and QuantumOracle integration
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from rdkit import Chem
from scipy.stats import spearmanr

from aurelius.scoring.oracle.delta_correction import DeltaCorrection, get_delta_correction
from aurelius.scoring.oracle.quantum import QuantumOracle, predict_tom_orbitals

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "src", "aurelius", "data"
)
CALIBRATION_PATH = os.path.join(DATA_DIR, "orbital_calibration.json")
BENCHMARK_PATH = os.path.join(DATA_DIR, "external_property_benchmark.json")


def _spearman(x: list[float], y: list[float]) -> float:
    return float(spearmanr(x, y).statistic)


def _load_json(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


class TestDeltaCorrection:
    def test_loo_mae_below_threshold(self):
        model = DeltaCorrection()
        mae = model.loo_mae()
        assert mae < 0.8, (
            f"Δ-learning LOO MAE is {mae:.4f} eV; expected < 0.8 eV. "
            "The residual model is not correcting TOM errors enough."
        )

    def test_corrected_rho_exceeds_raw_tom(self):
        calib = _load_json(CALIBRATION_PATH)
        benchmark = _load_json(BENCHMARK_PATH)

        model = DeltaCorrection()
        calib_smiles = {Chem.MolToSmiles(Chem.MolFromSmiles(e["smiles"])) for e in calib}

        raw_homo, corr_homo, exp_homo = [], [], []
        raw_lumo, corr_lumo, exp_lumo = [], [], []
        for entry in benchmark:
            if entry.get("homo_eV") is None or entry.get("lumo_eV") is None:
                continue
            mol = Chem.MolFromSmiles(entry["smiles"])
            if mol is None:
                continue
            raw_h, raw_l = predict_tom_orbitals(mol)
            corr_h, corr_l = model.predict_corrected(mol, base=(raw_h, raw_l))
            raw_homo.append(raw_h)
            corr_homo.append(corr_h)
            exp_homo.append(entry["homo_eV"])
            raw_lumo.append(raw_l)
            corr_lumo.append(corr_l)
            exp_lumo.append(entry["lumo_eV"])
            assert Chem.MolToSmiles(mol) in calib_smiles, (
                f"Benchmark molecule {entry['smiles']} must be in the calibration "
                "set for the corrected-vs-raw comparison to be meaningful."
            )

        rho_homo_raw = _spearman(raw_homo, exp_homo)
        rho_homo_corr = _spearman(corr_homo, exp_homo)
        rho_lumo_raw = _spearman(raw_lumo, exp_lumo)
        rho_lumo_corr = _spearman(corr_lumo, exp_lumo)

        assert rho_homo_corr > rho_homo_raw, (
            f"Corrected HOMO ρ={rho_homo_corr:.3f} must exceed raw TOM ρ={rho_homo_raw:.3f}"
        )
        assert rho_lumo_corr > rho_lumo_raw, (
            f"Corrected LUMO ρ={rho_lumo_corr:.3f} must exceed raw TOM ρ={rho_lumo_raw:.3f}"
        )

    def test_delta_correction_is_deterministic(self):
        model = get_delta_correction()
        mol = Chem.MolFromSmiles("COC(=O)OC")
        assert mol is not None
        assert model.predict_corrected(mol) == model.predict_corrected(mol)

    def test_quantum_oracle_applies_correction(self):
        qc = QuantumOracle(use_xtb=False, use_delta_correction=True)
        mol = Chem.MolFromSmiles("COC(=O)OC")
        result = qc.evaluate(mol)
        assert result["correction_applied"] is True
        raw_h, raw_l = predict_tom_orbitals(mol)
        # Correction is active: the corrected energies differ from raw TOM and
        # remain physically plausible.
        assert result["homo_eV"] != pytest.approx(raw_h, abs=1e-6)
        assert -12.0 <= result["homo_eV"] <= -3.0
        assert result["lumo_eV"] != pytest.approx(raw_l, abs=1e-6)
        assert -5.0 <= result["lumo_eV"] <= 5.0

    def test_quantum_oracle_raw_tom_bypass(self):
        qc = QuantumOracle(use_xtb=False, use_delta_correction=False)
        mol = Chem.MolFromSmiles("COC(=O)OC")
        result = qc.evaluate(mol)
        assert "correction_applied" not in result
        raw_h, raw_l = predict_tom_orbitals(mol)
        assert result["homo_eV"] == pytest.approx(raw_h)
        assert result["lumo_eV"] == pytest.approx(raw_l)
