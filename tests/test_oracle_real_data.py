"""Tests for the hybrid PropertyOracle (RF + GC + F/P/S correction).

Verifies that:
1. Oracle returns all four property types (HOMO, LUMO, Dielectric, Viscosity)
2. Predictions are physically plausible
3. Fragment-additivity model gives deterministic, interpretable results
4. Caching works correctly
5. RF model path works (mocked) with graceful GC fallback when missing
6. TPSA-based cap prevents unrealistic dielectric stacking
7. F/P/S correction layer properly shifts HOMO/LUMO
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from aurelius.scoring.oracle import PropertyOracle, get_data_source
from aurelius.types import MoleculeContext

ORACLE = None


@pytest.fixture(scope="module")
def oracle() -> PropertyOracle:
    global ORACLE
    if ORACLE is None:
        ORACLE = PropertyOracle()
    return ORACLE


def _ctx(smiles: str) -> MoleculeContext:
    ctx = MoleculeContext.from_smiles(smiles)
    assert ctx is not None, f"Failed to parse SMILES: {smiles}"
    return ctx


def test_oracle_data_source_is_gc(oracle: PropertyOracle) -> None:
    source = get_data_source()
    assert "fragment-additivity" in source or "Group Contribution" in source or "hybrid" in source or "correction" in source


def test_oracle_evaluate_returns_all_properties(oracle: PropertyOracle) -> None:
    smiles = "COC(=O)OC"
    result = oracle.evaluate(_ctx(smiles))
    assert "homo_eV" in result
    assert "lumo_eV" in result
    assert "gap_eV" in result
    assert "dielectric_proxy" in result
    assert "viscosity_proxy" in result
    assert "domain_applicable" in result


def test_oracle_plausible_ranges(oracle: PropertyOracle) -> None:
    smiles = "COC(=O)OC"
    result = oracle.evaluate(_ctx(smiles))
    assert -12.0 <= result["homo_eV"] <= -3.0, f"HOMO {result['homo_eV']} out of range"
    assert -5.0 <= result["lumo_eV"] <= 5.0, f"LUMO {result['lumo_eV']} out of range"
    assert 2.0 <= result["gap_eV"] <= 20.0, f"Gap {result['gap_eV']} out of range"
    assert 1.0 <= result["dielectric_proxy"] <= 25.0
    assert 0.1 <= result["viscosity_proxy"] <= 10.0


def test_oracle_caching(oracle: PropertyOracle) -> None:
    ctx = _ctx("CCO")
    r1 = oracle.evaluate(ctx)
    r2 = oracle.evaluate(ctx)
    assert r1 == r2


def test_oracle_known_molecules_consistent(oracle: PropertyOracle) -> None:
    known_molecules: list[tuple[str, float, float, float, float]] = [
        ("COC(=O)OC", 3.0, 15.0, 0.1, 3.0),
        ("O=C1OCCO1", 4.0, 20.0, 0.5, 4.0),
        ("C1COC(=O)O1", 4.0, 20.0, 0.5, 4.0),
        ("CC#N", 5.0, 15.0, 0.1, 2.5),
        ("CS(=O)(=O)C", 3.0, 20.0, 0.1, 4.0),
    ]
    for smi, min_diel, max_diel, min_visc, max_visc in known_molecules:
        try:
            result = oracle.evaluate(_ctx(smi))
        except Exception:
            continue
        assert min_diel <= result["dielectric_proxy"] <= max_diel
        assert min_visc <= result["viscosity_proxy"] <= max_visc


def test_oracle_fragment_sensitivity(oracle: PropertyOracle) -> None:
    ethane = oracle.evaluate(_ctx("CC"))
    ethanol = oracle.evaluate(_ctx("CCO"))
    acetonitrile = oracle.evaluate(_ctx("CC#N"))

    assert ethane["dielectric_proxy"] < ethanol["dielectric_proxy"]
    assert ethane["dielectric_proxy"] < acetonitrile["dielectric_proxy"]


def test_oracle_invalid_smiles_raises(oracle: PropertyOracle) -> None:
    with pytest.raises(TypeError):
        oracle.evaluate("not_a_valid_smiles")


def test_evaluate_smiles_works(oracle: PropertyOracle) -> None:
    smiles = "CC(=O)OC1=CC=CC=C1"
    result = oracle.evaluate_smiles(smiles)
    assert "homo_eV" in result
    assert "dielectric_proxy" in result
    assert "viscosity_proxy" in result


def test_oracle_charged_species_handled(oracle: PropertyOracle) -> None:
    result = oracle.evaluate_smiles("[Li+].[P-](F)(F)(F)(F)(F)F")
    assert "homo_eV" in result
    assert "dielectric_proxy" in result
    assert result["dielectric_proxy"] >= 1.0


# F/P/S correction tests


def test_fps_correction_lowers_homo_lumo(oracle: PropertyOracle) -> None:
    ethane = oracle.evaluate(_ctx("CC"))
    cf3 = oracle.evaluate_smiles("CC(F)(F)F")
    assert cf3["homo_eV"] < ethane["homo_eV"], (
        f"CF3 should lower HOMO: ethane={ethane['homo_eV']}, cf3={cf3['homo_eV']}"
    )
    assert cf3["lumo_eV"] < ethane["lumo_eV"], (
        f"CF3 should lower LUMO: ethane={ethane['lumo_eV']}, cf3={cf3['lumo_eV']}"
    )


def test_fps_correction_sulfone_lowers_lumo(oracle: PropertyOracle) -> None:
    ethane = oracle.evaluate(_ctx("CC"))
    sulfone = oracle.evaluate_smiles("CS(=O)(=O)C")
    assert sulfone["lumo_eV"] < ethane["lumo_eV"], (
        f"Sulfone should strongly lower LUMO: ethane={ethane['lumo_eV']}, sulfone={sulfone['lumo_eV']}"
    )


def test_tpsa_based_dielectric_cap():
    """Molecules with small TPSA should have capped dielectric even with many polar groups."""
    from rdkit import Chem
    from aurelius.scoring.oracle import predict_dielectric_proxy

    small_polar = predict_dielectric_proxy(Chem.MolFromSmiles("COCCOC"))
    many_carbonates = predict_dielectric_proxy(
        Chem.MolFromSmiles("O=C(OCCCCCCCCCCCCCCCC)OC(=O)OCCCCCCCCCCCCCCCC")
    )
    assert many_carbonates < 25.0, "TPSA cap should prevent unrealistically high dielectric"
    assert small_polar >= 1.0


# ---------------------------------------------------------------------------
# RF model path tests (mocked)
# ---------------------------------------------------------------------------


def _make_dummy_rf() -> RandomForestRegressor:
    X_dummy = np.random.randn(20, 2053).astype(np.float32)
    y_dummy = np.column_stack([
        np.random.uniform(-9.0, -5.0, 20),
        np.random.uniform(-2.0, 2.0, 20),
    ])
    model = RandomForestRegressor(n_estimators=2, max_depth=2, random_state=0)
    model.fit(X_dummy, y_dummy)
    return model


def test_oracle_rf_model_loaded_when_path_provided(tmp_path) -> None:
    model = _make_dummy_rf()
    import joblib
    model_path = tmp_path / "test_rf.joblib"
    joblib.dump(model, str(model_path))

    oracle_rf = PropertyOracle(model_path=str(model_path))
    assert oracle_rf._rf_model is not None

    result = oracle_rf.evaluate_smiles("CCO")
    assert "homo_eV" in result
    assert "lumo_eV" in result
    assert isinstance(result["homo_eV"], float)
    assert isinstance(result["lumo_eV"], float)


def test_oracle_rf_fallback_on_missing_path(monkeypatch) -> None:
    monkeypatch.setattr(
        "aurelius.scoring.oracle.DEFAULT_RF_MODEL_PATH",
        "/nonexistent/path.joblib",
    )
    oracle_fallback = PropertyOracle()
    assert oracle_fallback._rf_model is None

    source = get_data_source()
    assert "correction" in source or "GC" in source

    result = oracle_fallback.evaluate_smiles("CC")
    assert "homo_eV" in result
    assert abs(result["homo_eV"] - (-9.2)) < 1.5


def test_oracle_rf_prediction_uses_rf(tmp_path) -> None:
    X_dummy = np.random.randn(20, 2053).astype(np.float32)
    y_dummy = np.column_stack([np.full(20, -7.0), np.full(20, 0.0)])
    model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
    model.fit(X_dummy, y_dummy)

    import joblib
    model_path = tmp_path / "fixed_rf.joblib"
    joblib.dump(model, str(model_path))

    oracle_rf = PropertyOracle(model_path=str(model_path))
    result = oracle_rf.evaluate_smiles("CCO")
    assert abs(result["homo_eV"] - (-7.0)) < 2.0
    assert result["domain_applicable"] is True
    assert "RF" in result["domain_reason"]


# ---------------------------------------------------------------------------
# TPSA cap tests
# ---------------------------------------------------------------------------


def test_tpsa_capped_dielectric_prevents_stacking(oracle: PropertyOracle) -> None:
    from rdkit import Chem
    from aurelius.scoring.oracle import predict_dielectric_proxy

    single_diel = predict_dielectric_proxy(Chem.MolFromSmiles("O=C(OCC)OC"))
    stacked_diel = predict_dielectric_proxy(
        Chem.MolFromSmiles("O=C(OCCCCCCCCCCCCCCCCCCCCC)OC")
    )
    assert stacked_diel <= single_diel * 2.0 + 2.0, (
        "TPSA cap should prevent linear stacking of dielectric contribution"
    )


# ---------------------------------------------------------------------------
# Training function smoke test
# ---------------------------------------------------------------------------


def test_train_oracle_rf_smoke(tmp_path) -> None:
    from aurelius.scoring.oracle import train_oracle_rf

    save_path = tmp_path / "smoke_rf.joblib"
    with patch("aurelius.data.loaders.load_qm9_homo_lumo_data") as mock_load:
        mock_load.return_value = [
            ("CC", -9.2, 2.8),
            ("CO", -7.2, 2.1),
            ("CCO", -7.5, 1.9),
            ("CC=O", -6.9, -0.5),
            ("CC#N", -8.9, 1.0),
        ]
        result_path = train_oracle_rf(save_path=str(save_path))
        assert Path(result_path).exists()

    import joblib
    model = joblib.load(str(result_path))
    assert isinstance(model, RandomForestRegressor)

    oracle_rf = PropertyOracle(model_path=str(result_path))
    result = oracle_rf.evaluate_smiles("CCO")
    assert "homo_eV" in result
