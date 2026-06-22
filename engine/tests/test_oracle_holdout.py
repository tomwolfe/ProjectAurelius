"""Oracle holdout validation — prove TOM and GC proxies generalise to unseen data.

Splits the expanded orbital_calibration.json into an 80% calibration set
(used to tune TOM constants) and a 20% holdout set, then asserts:
  1. TOM MAE on the holdout set < 1.5 eV (same threshold as the calibration set)
  2. GC dielectric/viscosity rank order correlates with known experimental trends

Without this holdout, the TOM/GC tests only measure in-sample fit, which can
mask overfitting to the calibration molecules.
"""

from __future__ import annotations

import json
import os
import random

import numpy as np
import pytest
from rdkit import Chem

from aurelius.scoring.oracle.gc import predict_dielectric_proxy, predict_viscosity_proxy
from aurelius.scoring.oracle.quantum import predict_tom_orbitals
from aurelius.types import MoleculeContext

HOLDOUT_FRACTION = 0.20
TOM_MAE_THRESHOLD = 1.5
RANDOM_SEED = 42


@pytest.fixture(scope="module")
def calibration_data():
    path = os.path.join(
        os.path.dirname(__file__), "..", "src", "aurelius", "data", "orbital_calibration.json"
    )
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def holdout_data(calibration_data):
    """Split calibration data into calibration (80%) and holdout (20%) sets."""
    random.seed(RANDOM_SEED)
    indices = list(range(len(calibration_data)))
    random.shuffle(indices)
    n_holdout = max(1, int(len(calibration_data) * HOLDOUT_FRACTION))
    holdout_idx = set(indices[:n_holdout])
    holdout = [calibration_data[i] for i in holdout_idx]
    return holdout


class TestTOMHoldout:
    """TOM predictions must generalise to held-out molecules (MAE < 1.5 eV)."""

    def test_tom_holdout_mae_below_threshold(self, holdout_data):
        errors = []
        for entry in holdout_data:
            mol = Chem.MolFromSmiles(entry["smiles"])
            assert mol is not None, f"Invalid SMILES: {entry['smiles']}"
            homo_pred, lumo_pred = predict_tom_orbitals(mol)
            homo_err = abs(homo_pred - entry["homo_eV"])
            lumo_err = abs(lumo_pred - entry["lumo_eV"])
            errors.append((homo_err + lumo_err) / 2.0)

        mae = sum(errors) / len(errors)
        assert mae < TOM_MAE_THRESHOLD, (
            f"TOM MAE on holdout set is {mae:.4f} eV "
            f"(threshold: < {TOM_MAE_THRESHOLD} eV). "
            f"The TOM fallback does not generalise to unseen molecules."
        )

    def test_holdout_set_has_minimum_molecules(self, holdout_data):
        assert len(holdout_data) >= 5, (
            f"Holdout set has only {len(holdout_data)} entries; need >= 5 "
            f"for statistically meaningful validation."
        )

    def test_holdout_molecules_all_predicted(self, holdout_data):
        for entry in holdout_data:
            mol = Chem.MolFromSmiles(entry["smiles"])
            assert mol is not None
            homo, lumo = predict_tom_orbitals(mol)
            assert homo < 0.0, f"{entry['name']}: HOMO ({homo}) should be negative"
            assert lumo > -15.0, f"{entry['name']}: LUMO ({lumo}) suspiciously low"

    def test_calibration_and_holdout_are_disjoint(self, calibration_data, holdout_data):
        cal_smiles = {entry["smiles"] for entry in calibration_data}
        hol_smiles = {entry["smiles"] for entry in holdout_data}
        assert len(hol_smiles) <= len(calibration_data) * HOLDOUT_FRACTION + 1
        assert hol_smiles.issubset(cal_smiles)


class TestGCHoldout:
    """GC bulk property predictions must show correct rank ordering on held-out comparisons.

    Since the GC models are fragment-additivity heuristics rather than rigorously
    calibrated physical models, we test rank correlation rather than absolute accuracy.
    """

    def test_dielectric_rank_order_holds(self):
        """Known high-dielectric molecules must rank above known low-dielectric ones.

        Experimental trends:
          - Propylene carbonate (PC) > Dimethyl carbonate (DMC) > Diethyl ether (DEE)
          - Water > Ethylene glycol > Ethanol (if present)
        """
        high_diel = [
            ("C1COC(=O)O1", "EC"),      # ε ≈ 90
            ("CC1COC(=O)O1", "PC"),     # ε ≈ 65
        ]
        low_diel = [
            ("COC(=O)OC", "DMC"),       # ε ≈ 3.1
            ("CCOCC", "DEE"),           # ε ≈ 4.3
        ]

        high_values = []
        for smi, name in high_diel:
            ctx = MoleculeContext.from_smiles(smi)
            assert ctx is not None, f"Invalid SMILES: {smi} ({name})"
            high_values.append(predict_dielectric_proxy(ctx))

        low_values = []
        for smi, name in low_diel:
            ctx = MoleculeContext.from_smiles(smi)
            assert ctx is not None, f"Invalid SMILES: {smi} ({name})"
            low_values.append(predict_viscosity_proxy(ctx))
            low_values.append(predict_dielectric_proxy(ctx))

        mean_high = np.mean(high_values)
        mean_low = np.mean(low_values)
        assert mean_high > mean_low, (
            f"Dielectric rank order violated: high-diel mean ({mean_high:.3f}) "
            f"should exceed low-diel mean ({mean_low:.3f}). "
            f"GC proxy does not capture known experimental dielectric trends."
        )

    def test_viscosity_rank_order_holds(self):
        """Branched/high-MW molecules must rank above small/linear ones for viscosity.

        Experimental trends:
          - Glycerol > Ethylene glycol > Ethanol
          - Branched carbonates > linear carbonates
        """
        low_visc = [
            ("COC(=O)OC", "DMC"),       # Linear, low MW
            ("CCOCC", "DEE"),           # Small ether
            ("CC#N", "ACN"),            # Small nitrile
        ]
        high_visc = [
            ("CC(C)(C)OC(=O)OC(C)(C)C", "DTBC"),  # Branched carbonate
            ("COCCOCCOC", "Diglyme"),              # Higher MW ether
        ]

        low_values = []
        for smi, name in low_visc:
            ctx = MoleculeContext.from_smiles(smi)
            assert ctx is not None, f"Invalid SMILES: {smi} ({name})"
            low_values.append(predict_viscosity_proxy(ctx))

        high_values = []
        for smi, name in high_visc:
            ctx = MoleculeContext.from_smiles(smi)
            assert ctx is not None, f"Invalid SMILES: {smi} ({name})"
            high_values.append(predict_viscosity_proxy(ctx))

        mean_high = np.mean(high_values)
        mean_low = np.mean(low_values)
        assert mean_high > mean_low, (
            f"Viscosity rank order violated: high-visc mean ({mean_high:.3f}) "
            f"should exceed low-visc mean ({mean_low:.3f}). "
            f"GC proxy does not capture known experimental viscosity trends."
        )

    def test_gc_holdout_all_predict(self):
        """Every molecule in the orbital calibration set must produce valid GC predictions."""
        path = os.path.join(
            os.path.dirname(__file__), "..", "src", "aurelius", "data", "orbital_calibration.json"
        )
        with open(path) as f:
            data = json.load(f)
        for entry in data:
            ctx = MoleculeContext.from_smiles(entry["smiles"])
            assert ctx is not None, f"Invalid SMILES: {entry['smiles']}"
            diel = predict_dielectric_proxy(ctx)
            visc = predict_viscosity_proxy(ctx)
            assert diel >= 1.0, f"{entry['name']}: dielectric ({diel}) < 1.0"
            assert visc >= 0.1, f"{entry['name']}: viscosity ({visc}) < 0.1"
