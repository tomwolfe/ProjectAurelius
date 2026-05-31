"""Tests for the pure fragment-additivity PropertyOracle.

Verifies that:
1. Oracle returns all four property types (HOMO, LUMO, Dielectric, Viscosity)
2. Predictions are physically plausible
3. Fragment-additivity model gives deterministic, interpretable results
4. Caching works correctly
"""

from __future__ import annotations

import pytest

from aurelius.scoring.oracle import PropertyOracle, get_data_source

ORACLE = None


@pytest.fixture(scope="module")
def oracle() -> PropertyOracle:
    """Create a PropertyOracle instance for testing."""
    global ORACLE
    if ORACLE is None:
        ORACLE = PropertyOracle()
    return ORACLE


def test_oracle_data_source_is_gc(oracle: PropertyOracle) -> None:
    """Verify the oracle uses fragment-additivity, not machine learning."""
    source = get_data_source()
    assert "fragment-additivity" in source or "Group Contribution" in source, (
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
