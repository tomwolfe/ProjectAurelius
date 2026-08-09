"""LUMO proxy tests — reduction stability proxy validation.

Validates the Δ-learning LUMO correction layer:
  - Scaffold-disjoint CV MAE improves over raw TOM
  - OOD shrinkage reverts to TOM baseline
  - Confidence is bounded [0, 1]
  - Integration with PropertyOracle
"""

import warnings

import numpy as np
from rdkit import Chem

from aurelius.scoring.oracle.lumo_proxy import (
    LumoProxy,
    predict_reduction_stability,
)
from aurelius.scoring.oracle.quantum import predict_tom_orbitals


def test_lumo_proxy_improves_mae_over_tom():
    """Scaffold-disjoint CV: corrected MAE must beat raw TOM MAE."""
    proxy = LumoProxy()
    result = proxy.scaffold_disjoint_mae(n_splits=5)

    assert result["n_test_total"] > 0, "No test molecules in CV"
    assert result["mae_corrected"] < result["mae_raw_tom"], (
        f"Corrected MAE ({result['mae_corrected']:.4f}) must beat "
        f"raw TOM MAE ({result['mae_raw_tom']:.4f})"
    )


def test_lumo_proxy_mae_below_threshold():
    """Corrected LUMO MAE must be below 0.75 eV on scaffold-disjoint CV."""
    proxy = LumoProxy()
    result = proxy.scaffold_disjoint_mae(n_splits=5)

    assert result["mae_corrected"] < 0.75, (
        f"Corrected MAE {result['mae_corrected']:.4f} exceeds 0.75 eV threshold. "
        "The proxy is not accurate enough for soft penalty use."
    )


def test_confidence_bounded():
    """Confidence must be in [0, 1] for all molecules."""
    proxy = LumoProxy()
    test_smiles = [
        "COC(=O)OC", "C1COC(=O)O1", "CC#N", "c1ccccc1",
        "C1CCOC1", "CCOCC", "CS(=O)(=O)C", "C1CS(=O)(=O)CC1",
    ]
    for smi in test_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        _, conf = proxy.predict_corrected(mol)
        assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of [0, 1] for {smi}"


def test_ood_shrinkage_to_tom():
    """Out-of-domain molecules must have correction shrunk toward TOM.

    Exotic molecules (Si, Ge, As organics — absent from the electrolyte
    calibration set) should have lower confidence than DMC (a calibration
    molecule), and their corrected LUMO should be closer to the raw TOM
    prediction than DMC's.
    """
    proxy = LumoProxy()

    dmc = Chem.MolFromSmiles("COC(=O)OC")
    exotic = Chem.MolFromSmiles("[Si](C)(C)(C)C")
    if dmc is None or exotic is None:
        return

    _, conf_dmc = proxy.predict_corrected(dmc)
    _, tom_exotic = predict_tom_orbitals(exotic)
    corrected_exotic, conf_exotic = proxy.predict_corrected(exotic)

    assert conf_exotic < conf_dmc, (
        f"Exotic molecule confidence ({conf_exotic}) should be lower "
        f"than DMC ({conf_dmc})"
    )
    assert abs(corrected_exotic - tom_exotic) < 0.4, (
        f"Exotic molecule LUMO should be near TOM, "
        f"got {corrected_exotic:.3f} vs TOM {tom_exotic:.3f}"
    )


def test_predict_reduction_stability_api():
    """Public API returns expected keys."""
    mol = Chem.MolFromSmiles("COC(=O)OC")
    result = predict_reduction_stability(mol)

    assert "lumo_eV" in result, "Missing lumo_eV key"
    assert "confidence" in result, "Missing confidence key"
    assert -5.0 <= result["lumo_eV"] <= 5.0, (
        f"LUMO {result['lumo_eV']} outside physical bounds"
    )
    assert 0.0 <= result["confidence"] <= 1.0, (
        f"Confidence {result['confidence']} outside [0, 1]"
    )


def test_lumo_proxy_deterministic():
    """Same molecule must produce same output on repeated calls."""
    proxy = LumoProxy()
    mol = Chem.MolFromSmiles("COC(=O)OC")

    r1 = proxy.predict_corrected(mol)
    r2 = proxy.predict_corrected(mol)
    assert r1 == r2, f"Non-deterministic: {r1} != {r2}"


def test_multiple_scaffold_splits():
    """Scaffold-disjoint CV must work with different split counts."""
    proxy = LumoProxy()

    for n in (3, 5):
        result = proxy.scaffold_disjoint_mae(n_splits=n)
        assert result["n_test_total"] > 0, f"No test molecules with n_splits={n}"
        assert np.isfinite(result["mae_corrected"]), (
            f"Non-finite MAE with n_splits={n}"
        )
