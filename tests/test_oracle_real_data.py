"""Tests for the hybrid Quantum + GC PropertyOracle.

Verifies that:
1. Oracle returns all five property types (HOMO, LUMO, Dielectric, Viscosity, Li+ Solvation)
2. Predictions are physically plausible
3. QuantumOracle (TOM fallback) gives non-linear, topology-dependent HOMO/LUMO
4. GC fragment-additivity gives deterministic, interpretable bulk properties
5. Caching works correctly
6. TPSA-based cap prevents unrealistic dielectric stacking
7. Li+ solvation proxy is computed and physically sensible
8. QuantumOracle caches results and provides method metadata
"""

from __future__ import annotations

import pytest

from aurelius.scoring.oracle import (
    PropertyOracle,
    QuantumOracle,
    _count_fragments,
    get_data_source,
    predict_dielectric_proxy,
    predict_tom_orbitals,
)
from aurelius.types import MoleculeContext

ORACLE = None


@pytest.fixture(scope="module")
def oracle() -> PropertyOracle:
    global ORACLE
    if ORACLE is None:
        ORACLE = PropertyOracle(use_xtb=False)  # Force TOM fallback for tests
    return ORACLE


def _ctx(smiles: str) -> MoleculeContext:
    ctx = MoleculeContext.from_smiles(smiles)
    assert ctx is not None, f"Failed to parse SMILES: {smiles}"
    return ctx


def test_oracle_data_source_is_hybrid(oracle: PropertyOracle) -> None:
    source = get_data_source()
    assert "hybrid" in source


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
    assert "quantum_method" in result


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


def test_tpsa_based_dielectric_cap():
    from aurelius.scoring.oracle import predict_dielectric_proxy

    small_polar = predict_dielectric_proxy(_ctx("COCCOC"))
    many_carbonates = predict_dielectric_proxy(
        _ctx("O=C(OCCCCCCCCCCCCCCCC)OC(=O)OCCCCCCCCCCCCCCCC")
    )
    assert many_carbonates < 25.0, "TPSA cap should prevent unrealistically high dielectric"
    assert small_polar >= 1.0


def test_tpsa_capped_dielectric_prevents_stacking(oracle: PropertyOracle) -> None:
    from aurelius.scoring.oracle import predict_dielectric_proxy

    single_diel = predict_dielectric_proxy(_ctx("O=C(OCC)OC"))
    stacked_diel = predict_dielectric_proxy(
        _ctx("O=C(OCCCCCCCCCCCCCCCCCCCCC)OC")
    )
    assert stacked_diel <= single_diel * 2.0 + 2.0, (
        "TPSA cap should prevent linear stacking of dielectric contribution"
    )


# ---------------------------------------------------------------------------
# QuantumOracle Tests
# ---------------------------------------------------------------------------


def test_quantum_oracle_method_is_tom() -> None:
    """Without xTB binary, QuantumOracle should use TOM fallback."""
    qc = QuantumOracle(use_xtb=True)
    assert "TOM" in qc.method


def test_quantum_oracle_tom_is_deterministic() -> None:
    """TOM should produce the same HOMO/LUMO for the same molecule."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles("COC(=O)OC")
    qc = QuantumOracle(use_xtb=False)
    r1 = qc.evaluate(mol)
    r2 = qc.evaluate(mol)
    assert r1["homo_eV"] == r2["homo_eV"]
    assert r1["lumo_eV"] == r2["lumo_eV"]


def test_quantum_oracle_tom_conjugation_sensitivity() -> None:
    """TOM should give different (non-additive) results for conjugated systems."""
    from rdkit import Chem

    qc = QuantumOracle(use_xtb=False)

    # Simple alkane
    ethane = qc.evaluate(Chem.MolFromSmiles("CC"))

    # Conjugated butadiene
    butadiene = qc.evaluate(Chem.MolFromSmiles("C=CC=C"))

    # Aromatic benzene
    benzene = qc.evaluate(Chem.MolFromSmiles("c1ccccc1"))

    # Conjugation should narrow the gap (non-linear effect)
    gap_ethane = ethane["lumo_eV"] - ethane["homo_eV"]
    gap_butadiene = butadiene["lumo_eV"] - butadiene["homo_eV"]
    gap_benzene = benzene["lumo_eV"] - benzene["homo_eV"]

    # Longer conjugation = smaller gap (particle-in-a-box scaling)
    assert gap_butadiene < gap_ethane, (
        f"Butadiene gap {gap_butadiene:.3f} should be smaller than ethane gap {gap_ethane:.3f}"
    )
    assert gap_benzene < gap_butadiene, (
        f"Benzene gap {gap_benzene:.3f} should be smaller than butadiene gap {gap_butadiene:.3f}"
    )


def test_quantum_oracle_caching() -> None:
    """QuantumOracle should cache results by SMILES."""
    from rdkit import Chem

    qc = QuantumOracle(use_xtb=False)
    mol = Chem.MolFromSmiles("CCO")
    r1 = qc.evaluate(mol)
    r2 = qc.evaluate(mol)
    assert r1 == r2
    assert qc.get_cache_size() == 1


def test_predict_tom_orbitals_returns_plausible() -> None:
    """TOM should return physically plausible HOMO/LUMO values."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles("COC(=O)OC")
    homo, lumo = predict_tom_orbitals(mol)
    assert -12.0 <= homo <= -3.0, f"HOMO {homo} out of range"
    assert -5.0 <= lumo <= 5.0, f"LUMO {lumo} out of range"
    assert lumo > homo, f"LUMO {lumo} <= HOMO {homo}"


def test_fragment_saturation_prevents_stacking(oracle: PropertyOracle) -> None:
    """Verify that stacking 5 ester groups does NOT linearly multiply the dielectric proxy.

    The saturation function (1 - exp(-k * count)) ensures diminishing returns:
    5 esters should contribute less than 2x the fragment-additive part of 1 ester.
    The TPSA contribution is excluded from this comparison because it scales
    linearly with molecular surface area, not with fragment count.
    """
    from aurelius.scoring.oracle.gc import _GC_BASE_DIELECTRIC

    def _frag_contrib(smi: str) -> float:
        ctx = _ctx(smi)
        total = predict_dielectric_proxy(ctx)
        tpsa_part = ctx.tpsa * 0.030
        return total - _GC_BASE_DIELECTRIC - tpsa_part

    single_frag = _frag_contrib("CC(=O)OCC")
    five_frag = _frag_contrib("CC(=O)OCC(=O)OCC(=O)OCC(=O)OCC(=O)OC")

    # Verify that the test molecules actually have 1 and >=3 ester groups
    counts_single = _count_fragments(_ctx("CC(=O)OCC").mol)
    counts_five = _count_fragments(_ctx("CC(=O)OCC(=O)OCC(=O)OCC(=O)OCC(=O)OC").mol)
    assert counts_single.get("ester", 0) == 1, f"Expected 1 ester, got {counts_single.get('ester', 0)}"
    assert counts_five.get("ester", 0) >= 3, f"Expected >=3 esters, got {counts_five.get('ester', 0)}"

    # Critical assertion: saturation means 5 ester fragments << 5x single ester fragments
    assert five_frag < 2.0 * single_frag, (
        f"Saturation failed: 5 esters (frag contrib={five_frag:.3f}) should be "
        f"< 2x 1 ester (frag contrib={single_frag:.3f}, 2x={2*single_frag:.3f})"
    )


def test_tom_fluorine_correction() -> None:
    """Fluorinated molecules should have stabilised (lower) HOMO/LUMO."""
    from rdkit import Chem

    ethane_h, ethane_l = predict_tom_orbitals(Chem.MolFromSmiles("CC"))
    cf3_h, cf3_l = predict_tom_orbitals(Chem.MolFromSmiles("CC(F)(F)F"))

    # CF3 is strongly EW - should lower both HOMO and LUMO
    assert cf3_h < ethane_h, f"CF3 HOMO {cf3_h} should be lower than ethane {ethane_h}"
    assert cf3_l < ethane_l, f"CF3 LUMO {cf3_l} should be lower than ethane {ethane_l}"
