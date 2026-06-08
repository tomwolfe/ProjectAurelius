"""Calibration tests for the Topological Orbital Model (TOM).

Compares TOM HOMO/LUMO predictions against literature/DFT reference
values from orbital_calibration.json and enforces MAE < 1.5 eV.
"""

from __future__ import annotations

import json
import math
import os

import pytest
from rdkit import Chem

from aurelius.scoring.oracle.quantum import (
    _apply_cross_conjugation_penalty,
    _apply_torsional_strain_penalty,
    _is_cross_conjugated,
    _longest_conjugation_path,
    predict_tom_orbitals,
)


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
        assert len(calibration_data) >= 10, (
            f"Calibration set has only {len(calibration_data)} entries; need >= 10"
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

    def test_tom_cross_conjugation_widens_gap(self):
        """Cross-conjugated systems should have wider gaps than linear systems
        of similar atom count due to disrupted pi-delocalisation."""
        # Divinyl ketone (cross-conjugated): central C=O has 3 conjugated neighbors
        divinyl_ketone_h, divinyl_ketone_l = predict_tom_orbitals(
            Chem.MolFromSmiles("C=CC(=O)C=C")
        )
        # 1,3,5-hexatriene (linear conjugated): same number of pi-electrons
        hexatriene_h, hexatriene_l = predict_tom_orbitals(
            Chem.MolFromSmiles("C=CC=CC=C")
        )

        gap_cross = divinyl_ketone_l - divinyl_ketone_h
        gap_linear = hexatriene_l - hexatriene_h

        assert gap_cross > gap_linear, (
            f"Cross-conjugated divinyl ketone gap ({gap_cross:.3f}) should be "
            f"> linear hexatriene gap ({gap_linear:.3f})"
        )

    def test_tom_torsional_strain_penalty(self):
        """The torsional strain penalty should reduce effective conjugation length
        for sterically hindered conjugated molecules.

        A sterically hindered biphenyl derivative (2,2',6,6'-tetramethylbiphenyl)
        has large dihedral angles between rings, reducing pi-orbital overlap.
        Its effective L should be reduced by the penalty, while a planar molecule
        like butadiene should have minimal reduction.
        """
        # Butadiene (planar, short conjugated diene)
        butadiene = Chem.MolFromSmiles("C=CC=C")
        assert butadiene is not None
        b_L_raw = _longest_conjugation_path(butadiene)
        b_L_penalized = _apply_torsional_strain_penalty(butadiene, b_L_raw)
        # Butadiene is planar — penalty should not reduce L (or minimal)
        assert b_L_penalized == b_L_raw, (
            f"Butadiene L should not change with penalty: "
            f"{b_L_penalized} != {b_L_raw}"
        )

        # Sterically hindered: 2,2',6,6'-tetramethylbiphenyl
        hindered = Chem.MolFromSmiles("Cc1cccc(C)c1c2c(C)cccc2C")
        assert hindered is not None
        h_L_raw = _longest_conjugation_path(hindered)
        h_L_penalized = _apply_torsional_strain_penalty(hindered, h_L_raw)
        # The hindered molecule should have its L reduced by the penalty
        assert h_L_penalized < h_L_raw, (
            f"Torsional penalty should reduce L for twisted molecule: "
            f"{h_L_penalized} >= {h_L_raw}"
        )

        # Verify that reducing L widens the base gap (ΔE ∝ 1/L²)
        from aurelius.scoring.oracle.quantum import _compute_tom_base_energies
        base_raw_h, base_raw_l = _compute_tom_base_energies(h_L_raw)
        base_pen_h, base_pen_l = _compute_tom_base_energies(h_L_penalized)
        gap_raw = base_raw_l - base_raw_h
        gap_pen = base_pen_l - base_pen_h
        assert gap_pen > gap_raw, (
            f"Base gap after penalty ({gap_pen:.3f}) should be wider than "
            f"base gap without penalty ({gap_raw:.3f}) because reducing L "
            f"widens the particle-in-a-box gap"
        )

    def test_tom_cross_conjugation_benzophenone(self):
        """Benzophenone (cross-conjugated via carbonyl C) should have wider gap
        than linearly conjugated diphenyl butadiene of similar size."""
        benzophenone_h, benzophenone_l = predict_tom_orbitals(
            Chem.MolFromSmiles("O=C(c1ccccc1)c1ccccc1")
        )
        # 1,4-diphenyl-1,3-butadiene: linear conjugation through 4 sp2 carbons
        diphenyl_butadiene_h, diphenyl_butadiene_l = predict_tom_orbitals(
            Chem.MolFromSmiles("c1ccccc1C=CC=Cc2ccccc2")
        )

        gap_bp = benzophenone_l - benzophenone_h
        gap_dpb = diphenyl_butadiene_l - diphenyl_butadiene_h

        assert gap_bp > gap_dpb, (
            f"Benzophenone gap ({gap_bp:.3f}) should be > "
            f"diphenyl butadiene gap ({gap_dpb:.3f})"
        )

    def test_cross_conjugation_lumo_penalty_applied(self):
        """Cross-conjugated molecules must receive +0.30 eV LUMO penalty.

        Benzophenone (O=C(c1ccccc1)c1ccccc1) has a carbonyl carbon with three
        conjugated neighbors (O, two aromatic C atoms), triggering the
        cross-conjugation penalty. The _apply_cross_conjugation_penalty function
        should add exactly +0.30 eV to the LUMO for such molecules.
        """
        benzophenone = Chem.MolFromSmiles("O=C(c1ccccc1)c1ccccc1")
        assert benzophenone is not None
        assert _is_cross_conjugated(benzophenone), (
            "Benzophenone should be detected as cross-conjugated"
        )

        _, lumo_before = predict_tom_orbitals(benzophenone)

        # Recompute without the penalty to verify the +0.30 eV shift.
        # Simulate the TOM pipeline up to cross-conjugation.
        from aurelius.scoring.oracle.quantum import (
            _apply_aromatic_stabilization,
            _apply_fluorine_correction,
            _apply_heteroatom_perturbations,
            _apply_nitrile_correction,
            _apply_peierls_damping,
            _apply_phosphate_correction,
            _apply_sigma_star_correction,
            _apply_torsional_strain_penalty,
            _apply_wiener_compactness,
            _compute_tom_base_energies,
            _count_heteroatom_perturbations,
            _longest_conjugation_path,
            _topological_sanity_l,
        )
        L = _longest_conjugation_path(benzophenone)
        L = max(L, 2)
        L = _topological_sanity_l(benzophenone, L)
        L = _apply_wiener_compactness(benzophenone, L)
        L = _apply_torsional_strain_penalty(benzophenone, L)
        L = _apply_peierls_damping(L)
        n_ew, n_ed, _ = _count_heteroatom_perturbations(benzophenone)
        homo, lumo = _compute_tom_base_energies(L)
        homo, lumo = _apply_heteroatom_perturbations(n_ew, n_ed, homo, lumo)
        homo, lumo = _apply_fluorine_correction(benzophenone, homo, lumo)
        homo, lumo = _apply_aromatic_stabilization(benzophenone, homo, lumo)
        lumo = _apply_nitrile_correction(benzophenone, lumo)
        homo = _apply_phosphate_correction(benzophenone, homo)
        lumo = _apply_sigma_star_correction(benzophenone, L, lumo)

        # Now apply the cross-conjugation penalty
        lumo_with_penalty = _apply_cross_conjugation_penalty(benzophenone, lumo)

        assert math.isclose(lumo_with_penalty, lumo + 0.30, rel_tol=1e-6), (
            f"Cross-conjugation penalty should add +0.30 eV to LUMO: "
            f"lumo={lumo:.4f}, lumo_with_penalty={lumo_with_penalty:.4f}, "
            f"expected={lumo + 0.30:.4f}"
        )

        # Also verify a non-cross-conjugated molecule (DMC) does NOT get the penalty
        dmc = Chem.MolFromSmiles("COC(=O)OC")
        assert dmc is not None
        assert not _is_cross_conjugated(dmc), (
            "DMC should NOT be detected as cross-conjugated"
        )
        assert _apply_cross_conjugation_penalty(dmc, -0.5) == -0.5, (
            "Non-cross-conjugated molecule should not get LUMO penalty"
        )

    def test_tom_holdout_mae_below_threshold(self, calibration_data):
        """TOM holdout MAE must remain < 1.5 eV.

        Uses an 80/20 holdout split of orbital_calibration.json (same
        methodology as _compute_holdout_generalization in test_net_progress.py)
        to verify that TOM generalises to unseen scaffolds.
        """
        import random
        random.seed(42)
        indices = list(range(len(calibration_data)))
        random.shuffle(indices)
        n_holdout = max(1, int(len(calibration_data) * 0.20))
        holdout_idx = set(indices[:n_holdout])
        holdout = [calibration_data[i] for i in holdout_idx]

        errors = []
        for entry in holdout:
            mol = Chem.MolFromSmiles(entry["smiles"])
            assert mol is not None, f"Invalid SMILES: {entry['smiles']}"
            homo_pred, lumo_pred = predict_tom_orbitals(mol)
            homo_err = abs(homo_pred - entry["homo_eV"])
            lumo_err = abs(lumo_pred - entry["lumo_eV"])
            errors.append((homo_err + lumo_err) / 2.0)

        mae = sum(errors) / len(errors)
        assert mae < 1.5, (
            f"TOM holdout MAE is {mae:.4f} eV. "
            f"Expected < 1.5 eV. Cross-conjugated molecules must be correctly "
            f"penalized to maintain generalization."
        )
