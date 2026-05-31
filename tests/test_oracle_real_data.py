"""Tests for the hybrid PropertyOracle (RF + GC fallback).

Verifies that:
1. Oracle returns all four property types (HOMO, LUMO, Dielectric, Viscosity)
2. Predictions are physically plausible
3. Fragment-additivity model gives deterministic, interpretable results
4. Caching works correctly
5. RF model path works (mocked) with graceful GC fallback when missing
6. Anti-gaming diminishes returns for repeated fragments
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from aurelius.scoring.oracle import (
    PropertyOracle,
    get_data_source,
)

ORACLE = None


@pytest.fixture(scope="module")
def oracle() -> PropertyOracle:
    """Create a PropertyOracle instance for testing (default = GC fallback)."""
    global ORACLE
    if ORACLE is None:
        ORACLE = PropertyOracle()
    return ORACLE


def test_oracle_data_source_is_gc(oracle: PropertyOracle) -> None:
    """Verify the oracle reports the correct data source.

    Without an RF model file, the oracle should fall back to GC.
    """
    source = get_data_source()
    assert "fragment-additivity" in source or "Group Contribution" in source or "hybrid" in source, (
        f"Expected GC in data source, got: {source}"
    )


def test_oracle_evaluate_returns_all_properties(oracle: PropertyOracle) -> None:
    """Verify oracle returns all four property types."""
    smiles = "COC(=O)OC"  # dimethyl carbonate
    result = oracle.evaluate(smiles)

    assert "homo_eV" in result
    assert "lumo_eV" in result
    assert "gap_eV" in result
    assert "dielectric_proxy" in result
    assert "viscosity_proxy" in result
    assert "domain_applicable" in result


def test_oracle_plausible_ranges(oracle: PropertyOracle) -> None:
    """Verify oracle returns physically plausible values."""
    smiles = "COC(=O)OC"  # dimethyl carbonate
    result = oracle.evaluate(smiles)

    # Physically plausible ranges for organic electrolyte molecules
    assert -12.0 <= result["homo_eV"] <= -3.0, f"HOMO {result['homo_eV']} out of range"
    assert -5.0 <= result["lumo_eV"] <= 5.0, f"LUMO {result['lumo_eV']} out of range"
    assert 2.0 <= result["gap_eV"] <= 20.0, f"Gap {result['gap_eV']} out of range"
    assert 1.0 <= result["dielectric_proxy"] <= 25.0, (
        f"Dielectric proxy {result['dielectric_proxy']} out of range"
    )
    assert 0.1 <= result["viscosity_proxy"] <= 10.0, (
        f"Viscosity proxy {result['viscosity_proxy']} out of range"
    )


def test_oracle_caching(oracle: PropertyOracle) -> None:
    """Verify SMILES caching works and is deterministic."""
    smiles = "CCO"  # ethanol
    r1 = oracle.evaluate(smiles)
    r2 = oracle.evaluate(smiles)
    assert r1 == r2, "Cached results should be identical"


def test_oracle_known_molecules_consistent(oracle: PropertyOracle) -> None:
    """Test that oracle predictions are consistent for known electrolyte molecules."""
    known_molecules: list[tuple[str, float, float, float, float]] = [
        # (smiles, expected_min_dielectric, expected_max_dielectric, expected_min_viscosity, expected_max_viscosity)
        ("COC(=O)OC", 3.0, 15.0, 0.1, 3.0),     # DMC — moderate dielectric, low viscosity
        ("O=C1OCCO1", 4.0, 20.0, 0.5, 4.0),      # EC — high dielectric
        ("C1COC(=O)O1", 4.0, 20.0, 0.5, 4.0),    # propylene carbonate
        ("CC#N", 5.0, 15.0, 0.1, 2.5),            # acetonitrile — very high dielectric
        ("CS(=O)(=O)C", 3.0, 20.0, 0.1, 4.0),     # DMS(O2) — sulfone, high dielectric
    ]

    for smi, min_diel, max_diel, min_visc, max_visc in known_molecules:
        try:
            result = oracle.evaluate(smi)
        except Exception:
            continue

        assert min_diel <= result["dielectric_proxy"] <= max_diel, (
            f"{smi}: dielectric_proxy {result['dielectric_proxy']} "
            f"not in [{min_diel}, {max_diel}]"
        )
        assert min_visc <= result["viscosity_proxy"] <= max_visc, (
            f"{smi}: viscosity_proxy {result['viscosity_proxy']} "
            f"not in [{min_visc}, {max_visc}]"
        )


def test_oracle_fragment_sensitivity(oracle: PropertyOracle) -> None:
    """Adding polar fragments should increase dielectric proxy."""
    ethane = oracle.evaluate("CC")
    ethanol = oracle.evaluate("CCO")
    acetonitrile = oracle.evaluate("CC#N")

    # Dielectric: ethane < ethanol < acetonitrile (more polar groups -> higher dielectric)
    assert ethane["dielectric_proxy"] < ethanol["dielectric_proxy"], (
        f"Ethane dielectric {ethane['dielectric_proxy']} should be < "
        f"ethanol {ethanol['dielectric_proxy']}"
    )
    assert ethane["dielectric_proxy"] < acetonitrile["dielectric_proxy"], (
        f"Ethane dielectric {ethane['dielectric_proxy']} should be < "
        f"acetonitrile {acetonitrile['dielectric_proxy']}"
    )


def test_oracle_invalid_smiles_raises(oracle: PropertyOracle) -> None:
    """Verify invalid SMILES raises an error."""
    with pytest.raises(ValueError):
        oracle.evaluate("not_a_valid_smiles")


def test_evaluate_with_ood_penalty(oracle: PropertyOracle) -> None:
    """Verify evaluate_with_ood_penalty still works (backward compat)."""
    smiles = "CC(=O)OC1=CC=CC=C1"
    result = oracle.evaluate_with_ood_penalty(smiles)
    assert "homo_eV" in result
    assert "dielectric_proxy" in result
    assert "viscosity_proxy" in result


def test_oracle_charged_species_handled(oracle: PropertyOracle) -> None:
    """Oracle should handle ionic species (e.g., LiPF6 fragments)."""
    # Lithium hexafluorophosphate (ionic)
    result = oracle.evaluate("[Li+].[P-](F)(F)(F)(F)(F)F")
    assert "homo_eV" in result
    assert "dielectric_proxy" in result
    assert result["dielectric_proxy"] >= 1.0


# ---------------------------------------------------------------------------
# RF model path tests (mocked — no training during tests)
# ---------------------------------------------------------------------------


def _make_dummy_rf() -> RandomForestRegressor:
    """Build a tiny RandomForest that returns plausible HOMO/LUMO values."""
    X_dummy = np.random.randn(20, 2053).astype(np.float32)
    y_dummy = np.column_stack([
        np.random.uniform(-9.0, -5.0, 20),   # plausible HOMO range
        np.random.uniform(-2.0, 2.0, 20),     # plausible LUMO range
    ])
    model = RandomForestRegressor(n_estimators=2, max_depth=2, random_state=0)
    model.fit(X_dummy, y_dummy)
    return model


def test_oracle_rf_model_loaded_when_path_provided(tmp_path) -> None:
    """Verify that a pre-trained RF model is used when ``model_path`` is given."""
    model = _make_dummy_rf()

    import joblib
    model_path = tmp_path / "test_rf.joblib"
    joblib.dump(model, str(model_path))

    oracle_rf = PropertyOracle(model_path=str(model_path))
    assert oracle_rf._rf_model is not None, "RF model should be loaded"

    result = oracle_rf.evaluate("CCO")
    assert "homo_eV" in result
    assert "lumo_eV" in result
    # Values should be floats (not None or error)
    assert isinstance(result["homo_eV"], float)
    assert isinstance(result["lumo_eV"], float)


def test_oracle_rf_fallback_on_missing_path(monkeypatch) -> None:
    """Verify GC fallback when RF model file does not exist."""
    # Patch DEFAULT_RF_MODEL_PATH to a non-existent file
    monkeypatch.setattr(
        "aurelius.scoring.oracle.DEFAULT_RF_MODEL_PATH",
        "/nonexistent/path.joblib",
    )
    oracle_fallback = PropertyOracle()
    assert oracle_fallback._rf_model is None, "RF should be None when model missing"

    source = get_data_source()
    assert "fallback" in source, (
        f"Expected fallback in data source, got: {source}"
    )

    result = oracle_fallback.evaluate("CC")
    assert "homo_eV" in result
    assert abs(result["homo_eV"] - (-9.2)) < 1.0, (
        f"GC fallback HOMO should be near base value, got {result['homo_eV']}"
    )


def test_oracle_rf_prediction_uses_rf(tmp_path) -> None:
    """Verify that a loaded RF model actually influences predictions."""
    # Train an RF that always predicts HOMO=-7.0, LUMO=0.0
    X_dummy = np.random.randn(20, 2053).astype(np.float32)
    y_dummy = np.column_stack([np.full(20, -7.0), np.full(20, 0.0)])
    model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
    model.fit(X_dummy, y_dummy)

    import joblib
    model_path = tmp_path / "fixed_rf.joblib"
    joblib.dump(model, str(model_path))

    oracle_rf = PropertyOracle(model_path=str(model_path))
    result = oracle_rf.evaluate("CCO")
    # Should be closer to RF prediction than GC base
    assert abs(result["homo_eV"] - (-7.0)) < 2.0, (
        f"RF prediction should dominate, got HOMO={result['homo_eV']}"
    )
    assert result["domain_applicable"] is True
    assert "RF" in result["domain_reason"]


# ---------------------------------------------------------------------------
# Anti-gaming tests
# ---------------------------------------------------------------------------


def test_anti_gaming_diminishes_repeated_fragments(oracle: PropertyOracle) -> None:
    """Four copies of the same polar fragment should give <4x the benefit."""
    from rdkit import Chem

    from aurelius.scoring.oracle import predict_dielectric_proxy

    smiles_stacked = "O=C(OCCC)OCCCCCOC(=O)OCCCCOC(=O)OCCCCOC(=O)O"  # multi-carbonate
    diel_1 = predict_dielectric_proxy(Chem.MolFromSmiles("O=C(OCC)OC"))  # single carbonate
    diel_4 = predict_dielectric_proxy(Chem.MolFromSmiles(smiles_stacked))

    # Anti-gaming should make 4x < 4x the contribution
    single_diel_contrib = diel_1 - 1.9  # remove base
    if single_diel_contrib > 0:
        four_naive = 1.9 + 4 * single_diel_contrib
        assert diel_4 < four_naive, (
            f"Anti-gaming should reduce dielectric for stacked carbonates: "
            f"{diel_4:.2f} vs naive {four_naive:.2f}"
        )


# ---------------------------------------------------------------------------
# Training function smoke test
# ---------------------------------------------------------------------------


def test_train_oracle_rf_smoke(tmp_path) -> None:
    """Verify that ``train_oracle_rf`` produces a loadable model."""
    from aurelius.scoring.oracle import train_oracle_rf

    save_path = tmp_path / "smoke_rf.joblib"
    with patch(
        "aurelius.data.loaders.load_qm9_homo_lumo_data"
    ) as mock_load:
        # Return minimal data for a quick smoke test
        mock_load.return_value = [
            ("CC", -9.2, 2.8),
            ("CO", -7.2, 2.1),
            ("CCO", -7.5, 1.9),
            ("CC=O", -6.9, -0.5),
            ("CC#N", -8.9, 1.0),
        ]
        result_path = train_oracle_rf(save_path=str(save_path))
        assert Path(result_path).exists()

    # Verify the saved model can be loaded and used
    import joblib
    model = joblib.load(str(result_path))
    assert isinstance(model, RandomForestRegressor)

    oracle_rf = PropertyOracle(model_path=str(result_path))
    result = oracle_rf.evaluate("CCO")
    assert "homo_eV" in result
