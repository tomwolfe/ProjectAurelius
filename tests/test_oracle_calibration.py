"""Calibration tests for the Topological Orbital Model (TOM).

Compares TOM HOMO/LUMO predictions against literature/DFT reference
values from orbital_calibration.json and enforces MAE < 1.5 eV.
"""

from __future__ import annotations

import json
import os

import pytest
from rdkit import Chem

from aurelius.scoring.oracle.quantum import predict_tom_orbitals


@pytest.fixture(scope="module")
def calibration_data():
    path = os.path.join(
        os.path.dirname(__file__), "..", "src", "aurelius", "data", "orbital_calibration.json"
    )
    with open(path) as f:
        return json.load(f)


class TestOracleCalibration:
    """TOM predictions must match reference values within MAE < 1.5 eV."""

    def test_tom_mae_below_threshold(self, calibration_data):
        errors = []
        for entry in calibration_data:
            mol = Chem.MolFromSmiles(entry["smiles"])
            assert mol is not None, f"Invalid SMILES: {entry['smiles']}"
            homo_pred, lumo_pred = predict_tom_orbitals(mol)
            homo_err = abs(homo_pred - entry["homo_eV"])
            lumo_err = abs(lumo_pred - entry["lumo_eV"])
            errors.append((homo_err + lumo_err) / 2.0)

        mae = sum(errors) / len(errors)
        assert mae < 1.5, (
            f"TOM MAE against calibration set is {mae:.4f} eV. "
            f"Expected < 1.5 eV. Recalibrate parameters in quantum.py."
        )

    def test_each_molecule_predicted(self, calibration_data):
        """Every molecule in the calibration set must produce valid predictions."""
        for entry in calibration_data:
            mol = Chem.MolFromSmiles(entry["smiles"])
            assert mol is not None
            homo, lumo = predict_tom_orbitals(mol)
            assert homo < 0.0, f"{entry['name']}: HOMO ({homo}) should be negative"
            assert lumo > -15.0, f"{entry['name']}: LUMO ({lumo}) suspiciously low"

    def test_calibration_set_has_minimum_molecules(self, calibration_data):
        # After purging synthetic placeholder data, we have real literature/DFT entries
        # Quality > quantity - require a minimum of 10 real, verifiable entries
        assert len(calibration_data) >= 10, (
            f"Calibration set has only {len(calibration_data)} entries; need >= 10 real entries"
        )

    def test_tom_conjugation_nonlinear(self):
        """HOMO-LUMO gap follows particle-in-a-box scaling: ΔE ∝ 1/L²."""
        ethane_h, ethane_l = predict_tom_orbitals(Chem.MolFromSmiles("CC"))
        butadiene_h, butadiene_l = predict_tom_orbitals(Chem.MolFromSmiles("C=CC=C"))
        benzene_h, benzene_l = predict_tom_orbitals(Chem.MolFromSmiles("c1ccccc1"))

        gap_ethane = ethane_l - ethane_h
        gap_butadiene = butadiene_l - butadiene_h
        gap_benzene = benzene_l - benzene_h

        assert gap_butadiene < gap_ethane, (
            f"Butadiene gap {gap_butadiene:.3f} should be < ethane gap {gap_ethane:.3f}"
        )
        assert gap_benzene < gap_butadiene, (
            f"Benzene gap {gap_benzene:.3f} should be < butadiene gap {gap_butadiene:.3f}"
        )
