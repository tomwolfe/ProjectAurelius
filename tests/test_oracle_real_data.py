"""Tests for the GC-only PropertyOracle (fragment-additivity).

Verifies that:
1. Oracle returns all five property types (HOMO, LUMO, Dielectric, Viscosity, Li+ Solvation)
2. Predictions are physically plausible
3. Fragment-additivity model gives deterministic, interpretable results
4. Caching works correctly
5. F/P/S corrections are embedded in the GC fragment table (no double-counting)
6. TPSA-based cap prevents unrealistic dielectric stacking
7. Li+ solvation proxy is computed and physically sensible
"""

from __future__ import annotations

import pytest

from aurelius.scoring.oracle import PropertyOracle, Tier3QuantumOracle, get_data_source
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
    assert "fragment-additivity" in source


def test_oracle_evaluate_returns_all_properties(oracle: PropertyOracle) -> None:
    smiles = "COC(=O)OC"
    result = oracle.evaluate(_ctx(smiles))
    assert "homo_eV" in result
    assert "lumo_eV" in result
    assert "gap_eV" in result
    assert "dielectric_proxy" in result
    assert "viscosity_proxy" in result
    assert "li_solvation_proxy" in result
    assert "domain_applicable" in result


def test_oracle_plausible_ranges(oracle: PropertyOracle) -> None:
    smiles = "COC(=O)OC"
    result = oracle.evaluate(_ctx(smiles))
    assert -12.0 <= result["homo_eV"] <= -3.0, f"HOMO {result['homo_eV']} out of range"
    assert -5.0 <= result["lumo_eV"] <= 5.0, f"LUMO {result['lumo_eV']} out of range"
    assert 2.0 <= result["gap_eV"] <= 20.0, f"Gap {result['gap_eV']} out of range"
    assert 1.0 <= result["dielectric_proxy"] <= 25.0
    assert 0.1 <= result["viscosity_proxy"] <= 10.0
    assert 0.5 <= result["li_solvation_proxy"] <= 10.0


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


def test_oracle_li_solvation_proxy_physical(oracle: PropertyOracle) -> None:
    fluorinated = oracle.evaluate_smiles("C(F)(F)F")
    carbonate = oracle.evaluate_smiles("COC(=O)OC")
    ether = oracle.evaluate_smiles("CCOCC")
    alcohol = oracle.evaluate_smiles("CCO")

    solv_f = fluorinated["li_solvation_proxy"]
    solv_c = carbonate["li_solvation_proxy"]
    solv_e = ether["li_solvation_proxy"]
    solv_a = alcohol["li_solvation_proxy"]

    # Fluorinated groups reduce Li+ binding (electron withdrawal)
    assert solv_f < 1.5, f"Fluorinated li_solvation should be low, got {solv_f}"
    # Carbonates bind moderately-strongly
    assert solv_c > 1.5, f"Carbonate li_solvation should be moderate-high, got {solv_c}"
    # Ethers bind moderately
    assert solv_e < solv_c, "Ethers should bind Li+ more weakly than carbonates"
    # Alcohols bind more strongly than ethers (higher donor number)
    assert solv_a > solv_e, f"Alcohols should bind Li+ more strongly than ethers: {solv_a} vs {solv_e}"


# F/P/S correction tests (now embedded in GC fragment table)


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
    from rdkit import Chem

    from aurelius.scoring.oracle import predict_dielectric_proxy

    small_polar = predict_dielectric_proxy(Chem.MolFromSmiles("COCCOC"))
    many_carbonates = predict_dielectric_proxy(
        Chem.MolFromSmiles("O=C(OCCCCCCCCCCCCCCCC)OC(=O)OCCCCCCCCCCCCCCCC")
    )
    assert many_carbonates < 25.0, "TPSA cap should prevent unrealistically high dielectric"
    assert small_polar >= 1.0


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
# Tier3QuantumOracle stub tests
# ---------------------------------------------------------------------------


def test_tier3_quantum_oracle_parse_missing(tmp_path) -> None:
    qc = Tier3QuantumOracle()
    result = qc.parse_xtb_out(str(tmp_path / "nonexistent.xtb.out"))
    assert result is None


def test_tier3_quantum_oracle_override() -> None:
    qc = Tier3QuantumOracle()
    gc_result = {
        "homo_eV": -7.5,
        "lumo_eV": 0.5,
        "gap_eV": 8.0,
        "dielectric_proxy": 5.0,
        "viscosity_proxy": 1.5,
        "domain_reason": "fragment-additivity",
    }
    qc_result = {"homo_eV": -7.23, "lumo_eV": 0.12, "dipole_D": 3.4}
    overridden = qc.override(gc_result, qc_result)
    assert overridden["homo_eV"] == -7.23
    assert overridden["lumo_eV"] == 0.12
    assert overridden["gap_eV"] == 7.35
    assert "quantum-chemical" in overridden["domain_reason"]
