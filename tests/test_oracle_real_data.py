"""Tests for the real-data-backed PropertyOracle.

Verifies that:
1. Oracle loads real QM9 data (not synthetic labels)
2. Predictions have meaningful correlation with ground truth (Pearson r > 0.3)
3. Data source is correctly logged and accessible via get_data_source()
"""

from __future__ import annotations

import numpy as np
import pytest

from aurelius.scoring.oracle import PropertyOracle, get_data_source


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def oracle() -> PropertyOracle:
    """Create a PropertyOracle instance for testing."""
    return PropertyOracle()


def test_oracle_data_source_is_real(oracle: PropertyOracle) -> None:
    """Verify the oracle uses real QM9 data, not synthetic labels."""
    # Trigger model training by evaluating a molecule
    oracle.evaluate("CCO")
    source = get_data_source()
    assert "QM9" in source, f"Expected QM9 data source, got: {source}"
    assert "real data" in source, f"Expected real data, got: {source}"
    assert "synthetic" not in source.lower(), "Should not contain 'synthetic'"


def test_oracle_evaluate_returns_reasonable_values(oracle: PropertyOracle) -> None:
    """Verify oracle returns physically plausible HOMO/LUMO values."""
    smiles = "CC(=O)OC1=CC=CC=C1"  # methyl benzoate
    result = oracle.evaluate(smiles)

    assert "homo_eV" in result
    assert "lumo_eV" in result
    assert "gap_eV" in result

    # Physically plausible ranges for organic molecules
    assert -12.0 <= result["homo_eV"] <= -3.0, f"HOMO {result['homo_eV']} out of range"
    assert -5.0 <= result["lumo_eV"] <= 5.0, f"LUMO {result['lumo_eV']} out of range"
    assert 2.0 <= result["gap_eV"] <= 20.0, f"Gap {result['gap_eV']} out of range"


def test_oracle_caching(oracle: PropertyOracle) -> None:
    """Verify SMILES caching works and is deterministic."""
    smiles = "CCO"  # ethanol
    r1 = oracle.evaluate(smiles)
    r2 = oracle.evaluate(smiles)
    assert r1 == r2, "Cached results should be identical"


def test_oracle_known_molecules_correlation(oracle: PropertyOracle) -> None:
    """Test that oracle predictions correlate with known QM9 ground truth.

    Uses ≥10 molecules with known QM9 values and checks Pearson r > 0.3.
    This is the minimum bar for 'real' — predictions must have some signal.
    """
    # Known molecules from QM9 with experimental/computed LUMO values (in eV)
    # Source: QM9 dataset values (converted from Hartree)
    known_molecules: list[tuple[str, float, float]] = [
        ("CCO", -7.22, 2.13),       # ethanol
        ("CC=O", -6.91, -0.54),     # acetaldehyde
        ("CC(=O)O", -7.60, -0.05),  # acetic acid
        ("C1=CC=CC=C1", -6.72, 0.37),  # benzene
        ("CC(=O)OC", -7.34, 0.45),  # methyl acetate
        ("C1CCOC1", -7.15, 2.30),   # THF
        ("CC(C)O", -7.09, 2.24),    # isopropanol
        ("CN", -7.84, 2.66),        # methylamine
        ("CCOC", -7.17, 2.35),      # diethyl ether
        ("CCCC", -8.79, 2.58),      # butane
        ("C1=CC=CC=C1O", -6.09, 0.28),  # phenol
        ("CC(=O)N", -6.92, 0.82),   # acetamide
        ("CC(C)(C)O", -7.05, 2.36), # tert-butanol
        ("C1=CC=NC=C1", -6.86, 0.21),  # pyridine
        ("C#N", -9.81, 0.52),       # hydrogen cyanide
    ]

    predicted_lumo: list[float] = []
    ground_truth_lumo: list[float] = []
    predicted_homo: list[float] = []
    ground_truth_homo: list[float] = []

    for smi, gt_homo, gt_lumo in known_molecules:
        try:
            result = oracle.evaluate(smi)
        except Exception:
            continue
        predicted_homo.append(result["homo_eV"])
        predicted_lumo.append(result["lumo_eV"])
        ground_truth_homo.append(gt_homo)
        ground_truth_lumo.append(gt_lumo)

    assert len(predicted_lumo) >= 10, (
        f"Need ≥10 valid molecules for correlation test, got {len(predicted_lumo)}"
    )

    # Compute Pearson correlation
    lumo_corr = np.corrcoef(predicted_lumo, ground_truth_lumo)[0, 1]
    homo_corr = np.corrcoef(predicted_homo, ground_truth_homo)[0, 1]

    # At least one of HOMO or LUMO should have r > 0.3
    assert lumo_corr > 0.3 or homo_corr > 0.3, (
        f"Correlation too low: LUMO r={lumo_corr:.3f}, HOMO r={homo_corr:.3f}. "
        "Predictions show no signal against ground truth."
    )


def test_oracle_invalid_smiles_raises(oracle: PropertyOracle) -> None:
    """Verify invalid SMILES raises an error."""
    with pytest.raises((RuntimeError, ValueError)):
        oracle.evaluate("not_a_valid_smiles")


def test_predict_normalized_lumo_range(oracle: PropertyOracle) -> None:
    """Verify normalized LUMO score is in [0, 100]."""
    smiles = "CC(=O)OC1=CC=CC=C1"
    score = oracle.predict_normalized_lumo(smiles)
    assert 0.0 <= score <= 100.0, f"Score {score} out of [0, 100] range"



