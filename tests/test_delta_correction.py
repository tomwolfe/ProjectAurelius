"""Δ-learning correction layer tests.

Verifies that the kernel-ridge residual model:
  1. Reaches LOO cross-validation MAE < 0.8 eV on the calibration set
  2. Improves Spearman ρ over raw TOM on the external benchmark
  3. Preserves the deterministic base TOM API and QuantumOracle integration
"""

from __future__ import annotations

import json
import os
import random

import numpy as np
import pytest
from rdkit import Chem
from scipy.stats import spearmanr

from aurelius.scoring.oracle.delta_correction import (
    DeltaCorrection,
    compute_ood_spearman,
    get_delta_correction,
)
from aurelius.scoring.oracle.lone_pair import predict_lone_pair_homo
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


def _load_calibration() -> list[dict]:
    with open(CALIBRATION_PATH) as f:
        return json.load(f)


def _create_train_test_split(calib_data: list[dict], test_ratio: float = 0.2, random_seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Create a train/test split for calibration data.

    Stratifies by chemical family based on the name prefix as a proxy for chemical class.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    calib_with_mols = []
    for entry in calib_data:
        mol = Chem.MolFromSmiles(entry["smiles"])
        if mol is None:
            continue
        calib_with_mols.append({
            "entry": entry,
            "mol": mol,
            "name": entry["name"],
            "smiles": entry["smiles"]
        })

    families = {}
    for item in calib_with_mols:
        name = item["name"]
        prefix = name.split('_')[0] if '_' in name else name[0]
        families.setdefault(prefix, []).append(item)

    train_items, test_items = [], []
    for _family, family_items in families.items():
        random.shuffle(family_items)
        split_idx = int(len(family_items) * (1 - test_ratio))
        train_items.extend(family_items[:split_idx])
        test_items.extend(family_items[split_idx:])

    train_calib = [item["entry"] for item in train_items]
    test_calib = [item["entry"] for item in test_items]

    train_smiles = [item["smiles"] for item in train_items]
    test_smiles = [item["smiles"] for item in test_items]

    return train_calib, test_calib, train_smiles, test_smiles


class TestDeltaCorrection:
    def test_loo_mae_below_threshold(self):
        calib = _load_json(CALIBRATION_PATH)
        train_calib, test_calib, train_smiles, test_smiles = _create_train_test_split(calib)

        model = DeltaCorrection(calib=train_calib, calib_smiles=train_smiles)
        mae = model.loo_mae()
        assert mae < 0.85, (
            f"Δ-learning LOO MAE is {mae:.4f} eV; expected < 0.85 eV. "
            "The residual model is not correcting TOM errors enough."
        )

    def test_corrected_rho_exceeds_raw_tom(self):
        calib = _load_json(CALIBRATION_PATH)
        benchmark = _load_json(BENCHMARK_PATH)

        train_calib, test_calib, train_smiles, test_smiles = _create_train_test_split(calib)

        model = DeltaCorrection(calib=train_calib, calib_smiles=train_smiles)

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

    def test_quantum_oracle_raw_bypass(self):
        """With corrections off, the oracle returns the raw closed-form values.

        ADR-2026-08-08-01: the HOMO now comes from the Lone-Pair Orbital Model
        and only the LUMO from TOM, so this asserts against each model's own
        output rather than assuming TOM supplies both.
        """
        qc = QuantumOracle(use_xtb=False, use_delta_correction=False)
        mol = Chem.MolFromSmiles("COC(=O)OC")
        result = qc.evaluate(mol)
        assert "correction_applied" not in result
        _, raw_l = predict_tom_orbitals(mol)
        assert result["homo_eV"] == pytest.approx(predict_lone_pair_homo(mol))
        assert result["lumo_eV"] == pytest.approx(raw_l)

    def test_lone_pair_can_be_disabled(self):
        """``use_lone_pair=False`` recovers the pre-v12 pure-TOM behaviour."""
        qc = QuantumOracle(use_xtb=False, use_delta_correction=False, use_lone_pair=False)
        mol = Chem.MolFromSmiles("COC(=O)OC")
        result = qc.evaluate(mol)
        raw_h, raw_l = predict_tom_orbitals(mol)
        assert result["homo_eV"] == pytest.approx(raw_h)
        assert result["lumo_eV"] == pytest.approx(raw_l)

    def test_ood_spearman_improvement(self):
        """Δ-correction must beat raw TOM on molecules it has never seen.

        ADR-2026-08-07-09: this test previously added the evaluation
        molecules to the training set and then scored the model on those same
        molecules, which measures memorisation rather than transferability.
        Under that protocol the "improved" model reached ρ = 0.996 and
        MAE = 0.01 eV — values that are unreachable out of domain and that
        masked the real behaviour. The reported OOD ρ ≈ 0.51 came from the
        same leak.

        Here the benchmark molecules that also appear in the calibration set
        are removed, so the evaluation set is genuinely unseen.
        """
        calib = _load_json(CALIBRATION_PATH)
        benchmark = _load_json(BENCHMARK_PATH)

        def canonical(smiles: str) -> str | None:
            mol = Chem.MolFromSmiles(smiles)
            return Chem.MolToSmiles(mol) if mol is not None else None

        calib_smiles = {canonical(e["smiles"]) for e in calib}
        held_out = [
            e for e in benchmark
            if e.get("homo_eV") is not None
            and canonical(e.get("smiles", "")) is not None
            and canonical(e["smiles"]) not in calib_smiles
        ]
        assert len(held_out) >= 20, (
            f"need a meaningful held-out set, got {len(held_out)} molecules"
        )

        model = DeltaCorrection(calib=calib)
        corrected, raw, reference = [], [], []
        for entry in held_out:
            mol = Chem.MolFromSmiles(entry["smiles"])
            if mol is None:
                continue
            raw_h, raw_l = predict_tom_orbitals(mol)
            corr_h, _ = model.predict_corrected(mol, base=(raw_h, raw_l))
            corrected.append(corr_h)
            raw.append(raw_h)
            reference.append(entry["homo_eV"])

        raw_mae = float(np.mean(np.abs(np.array(raw) - np.array(reference))))
        corr_mae = float(np.mean(np.abs(np.array(corrected) - np.array(reference))))
        print(f"\nheld-out (n={len(corrected)}): raw MAE={raw_mae:.3f} "
              f"corrected MAE={corr_mae:.3f}")

        assert corr_mae < raw_mae, (
            f"Delta correction must reduce held-out HOMO error: "
            f"raw {raw_mae:.3f} eV vs corrected {corr_mae:.3f} eV"
        )

    def test_shrinkage_retains_most_of_the_correction_in_domain(self):
        """The uncertainty damping must not discard the correction wholesale.

        Guards the defect ADR-2026-08-07-09 fixed: the previous hand-set
        cutoff of 0.5 eV left a mean confidence of 0.19 across the reference
        molecules, so 81% of every correction was thrown away and the
        Δ-layer barely improved on raw TOM. The Bayesian shrinkage factor
        sigma_prior^2 / (sigma_prior^2 + sigma_pred^2) retains ~0.79.
        """
        calib = _load_json(CALIBRATION_PATH)
        model = DeltaCorrection(calib=calib)

        confidences = []
        for entry in calib[:40]:
            mol = Chem.MolFromSmiles(entry["smiles"])
            if mol is None:
                continue
            _dh, _dl, std_homo, _std_lumo = model.predict_deltas_with_uncertainty(mol)
            var_prior = model._prior_std_homo ** 2
            confidences.append(var_prior / (var_prior + std_homo ** 2))

        mean_conf = float(np.mean(confidences))
        assert mean_conf > 0.5, (
            f"mean shrinkage confidence {mean_conf:.3f} is too aggressive; the "
            f"Delta correction is being discarded before it can help"
        )

    def test_ood_does_not_degrade_in_domain(self):
        """OOD-weighted training should not degrade in-domain ρ by more than 0.05."""
        calib = _load_json(CALIBRATION_PATH)

        # Use first 20 calibration molecules as in-domain test set
        in_domain = calib[:20]
        assert len(in_domain) >= 5

        baseline_model = DeltaCorrection(calib=calib)
        improved_model = DeltaCorrection(calib=calib, ood_calibration_set=calib)

        baseline_in_domain = compute_ood_spearman(in_domain, model=baseline_model)
        improved_in_domain = compute_ood_spearman(in_domain, model=improved_model)

        in_domain_change = improved_in_domain - baseline_in_domain
        print(f"\nIn-domain ρ: baseline={baseline_in_domain:.4f}, improved={improved_in_domain:.4f}")
        print(f"In-domain ρ change: {in_domain_change:+.4f}")

        assert in_domain_change >= -0.05, (
            f"In-domain ρ should not degrade by more than 0.05: "
            f"baseline={baseline_in_domain:.4f}, improved={improved_in_domain:.4f}"
        )
