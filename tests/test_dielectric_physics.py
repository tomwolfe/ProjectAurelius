"""Accuracy and invariance tests for the Kirkwood-Fröhlich dielectric model.

ADR-2026-08-07-04. These lock in the behaviour that the v11.0 review asked
for — cyclic carbonates correct, linear carbonates untouched — and guard the
mechanisms that make the model generalise rather than merely fit EC and PC.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

from aurelius.scoring.oracle.gc import (
    _count_dipole_groups,
    _kirkwood_g_factor,
    _mcgowan_molar_volume,
    _molecular_dipole,
    _optical_dielectric,
    predict_dielectric_constant,
)
from aurelius.types import MoleculeContext

_VERIFIED_PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "data" / "dielectric_verified.json"


def _eps(smiles: str) -> float:
    ctx = MoleculeContext.from_smiles(smiles)
    assert ctx is not None, f"failed to parse {smiles}"
    return predict_dielectric_constant(ctx)


@pytest.fixture(scope="module")
def verified() -> list[dict]:
    return json.loads(_VERIFIED_PATH.read_text())["entries"]


# --- The headline requirement from the v11.0 review ------------------------

# The brief specified "Dielectric MAE < 3.0 for top-10 commercial solvents
# while maintaining Spearman rho > 0.8".
_COMMERCIAL_SOLVENTS: list[tuple[str, str, float]] = [
    ("EC", "C1COC(=O)O1", 89.78),
    ("PC", "CC1COC(=O)O1", 64.92),
    ("DMC", "COC(=O)OC", 3.11),
    ("DEC", "CCOC(=O)OCC", 2.82),
    ("EMC", "CCOC(=O)OC", 2.96),
    ("GBL", "O=C1CCCO1", 39.0),
    ("ACN", "CC#N", 35.94),
    ("DMSO", "CS(=O)C", 46.7),
    ("THF", "C1CCOC1", 7.58),
    ("DME", "COCCOC", 7.20),
]


def test_commercial_solvent_mae_below_target() -> None:
    """MAE across the ten canonical battery-electrolyte solvents.

    The v11.0 brief set a target of MAE < 3.0. The model now reaches 1.80,
    down from 3.01, via the g-factor recalibration in ADR-2026-08-07-08.

    That change set each Kirkwood mechanism constant to the median g
    back-solved from the verified-set molecules exhibiting only that
    mechanism. It is a recalibration of existing physics under one uniform
    rule, not a new term: the previous _G_RING_LOCKED_DIPOLE of 1.30 lay
    below every member of its own class (1.349-1.534), so EC, PC, FEC and
    sulfolane were all underpredicted together.

    Two alternatives were implemented and rejected on evidence:
      - The brief's cyclic-carbonate boost g*(1 + 0.12*(eps_inf - 1)).
        eps_inf spans only 1.64-2.33 here (CV 6%), so this is a constant in
        disguise; and applied to cyclic carbonates alone it fixes EC only by
        overshooting PC (error 0.26 -> 3.99) and FEC (1.72 -> 2.47).
      - The uniform Onsager enhancement mu/(1 - f*alpha), which worsened the
        verified set to MAE 4.77 by overshooting the strongly polar solvents.

    EC remains the dominant residual at 81.5 against 89.78. That gap is the
    gas-phase dipole (4.90 D versus ~5.35 D condensed-phase) entering
    squared, and cannot be closed through g without breaking the rest of the
    ring-locked class. Closing it properly needs condensed-phase dipole
    moments.
    """
    errors = [abs(_eps(smiles) - experimental)
              for _, smiles, experimental in _COMMERCIAL_SOLVENTS]
    mae = float(np.mean(errors))
    assert mae < 2.8, f"commercial solvent dielectric MAE {mae:.2f} regressed"

    without_ec = float(np.mean(errors[1:]))
    assert without_ec < 1.5, (
        f"MAE excluding EC is {without_ec:.2f}; the residual error should "
        f"remain concentrated in EC's dipole moment"
    )


def test_g_factors_match_backsolved_class_medians() -> None:
    """Each g mechanism constant must sit within its own class's spread.

    Guards the defect ADR-2026-08-07-08 fixed: a constant that lies outside
    the range of the molecules it describes underpredicts every member of
    that class at once. Ranges are the back-solved g values over verified-set
    molecules exhibiting only the given mechanism.
    """
    from aurelius.scoring.oracle.gc import (
        _G_HYDROGEN_BONDED,
        _G_NITRILE_ANTIPARALLEL,
        _G_RING_LOCKED_DIPOLE,
        _G_SOFT_DIPOLE_ASSOCIATION,
    )

    for name, value, low, high in (
        ("ring_locked", _G_RING_LOCKED_DIPOLE, 1.349, 1.534),
        ("soft_dipole", _G_SOFT_DIPOLE_ASSOCIATION, 1.195, 1.487),
        ("hydrogen_bonded", _G_HYDROGEN_BONDED, 2.822, 4.723),
        ("nitrile", _G_NITRILE_ANTIPARALLEL, 0.663, 0.942),
    ):
        assert low <= value <= high, (
            f"_G_{name.upper()} = {value} lies outside the back-solved range "
            f"[{low}, {high}] of the molecules it describes; every member of "
            f"the class will be biased in the same direction"
        )


def test_commercial_solvent_rank_correlation() -> None:
    """Spearman rho > 0.8 across the same set."""
    predicted = [_eps(smiles) for _, smiles, _ in _COMMERCIAL_SOLVENTS]
    experimental = [value for _, _, value in _COMMERCIAL_SOLVENTS]
    rho = spearmanr(predicted, experimental).statistic
    assert rho > 0.8, f"commercial solvent Spearman rho {rho:.3f} below 0.8"


def test_cyclic_carbonates_are_high_dielectric() -> None:
    """EC and PC must land near their experimental values, not the old ~15 cap."""
    assert 70.0 < _eps("C1COC(=O)O1") < 105.0    # exp 89.78
    assert 50.0 < _eps("CC1COC(=O)O1") < 80.0    # exp 64.92


def test_linear_carbonates_unaffected() -> None:
    """The regression guard the brief explicitly asked for: linear eps < 4."""
    for name, smiles in (("DMC", "COC(=O)OC"),
                         ("DEC", "CCOC(=O)OCC"),
                         ("EMC", "CCOC(=O)OC")):
        value = _eps(smiles)
        assert value < 4.0, f"{name} dielectric {value:.2f} should stay below 4"


def test_cyclic_linear_carbonate_ratio() -> None:
    """EC/DMC ratio must reproduce the ~20-30x experimental gap."""
    ratio = _eps("C1COC(=O)O1") / _eps("COC(=O)OC")
    assert 18.0 < ratio < 35.0, f"EC/DMC dielectric ratio {ratio:.1f} outside 18-35"


# --- Accuracy on the full verified reference set ---------------------------

def test_verified_set_accuracy(verified: list[dict]) -> None:
    """Aggregate accuracy against 55 formula-checked, literature-cited values."""
    predicted = [_eps(e["smiles"]) for e in verified]
    experimental = [e["dielectric_constant"] for e in verified]

    mae = float(np.mean([abs(p - t) for p, t in zip(predicted, experimental, strict=True)]))
    rho = spearmanr(predicted, experimental).statistic

    assert mae < 5.0, f"verified-set MAE {mae:.2f} regressed above 5.0"
    assert rho > 0.85, f"verified-set Spearman rho {rho:.3f} regressed below 0.85"


def test_beats_fingerprint_baseline_on_clean_labels(verified: list[dict]) -> None:
    """Physics must beat ECFP4+RF when labels are trustworthy.

    The v11.0 audit reported the oracle losing to a fingerprint regressor.
    That comparison ran against benchmark entries whose dielectric values
    were themselves wrong, which a fingerprint model can memorise and a
    physical model cannot. On verified labels the ordering reverses.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_predict

    features = np.array([
        np.array(AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(e["smiles"]), 2, 2048))
        for e in verified
    ])
    target = np.array([e["dielectric_constant"] for e in verified])
    physics = np.array([_eps(e["smiles"]) for e in verified])

    baseline = cross_val_predict(
        RandomForestRegressor(n_estimators=300, random_state=0),
        features, target, cv=5,
    )

    physics_mae = float(np.mean(np.abs(physics - target)))
    baseline_mae = float(np.mean(np.abs(baseline - target)))
    assert physics_mae < baseline_mae, (
        f"physics MAE {physics_mae:.2f} should beat ECFP4+RF {baseline_mae:.2f}"
    )


# --- The physical mechanisms, tested individually --------------------------

def test_dipole_dominates_cyclic_linear_difference() -> None:
    """EC vs DMC must be driven by dipole moment, not the g-factor.

    This is the specific physics the v11.0 brief got wrong: it attributed the
    20-30x gap to Kirkwood g. Back-solving from experiment gives g(EC)=1.53
    versus g(DMC)=0.98 — a factor of 1.6 — while the dipoles are 4.90 D and
    0.90 D, a factor of 30 once squared. If a future change reintroduces a
    large cyclic-specific g, this test fails.
    """
    ec = MoleculeContext.from_smiles("C1COC(=O)O1")
    dmc = MoleculeContext.from_smiles("COC(=O)OC")
    ec_groups = _count_dipole_groups(ec.mol)
    dmc_groups = _count_dipole_groups(dmc.mol)

    dipole_ratio = _molecular_dipole(ec_groups) / _molecular_dipole(dmc_groups)
    g_ratio = _kirkwood_g_factor(ec.mol, ec_groups) / _kirkwood_g_factor(dmc.mol, dmc_groups)

    assert dipole_ratio > 3.0, "EC/DMC dipole ratio should be large"
    assert g_ratio < 2.0, "EC/DMC g-factor ratio should be modest"
    assert dipole_ratio > g_ratio, (
        "dipole moment, not the Kirkwood factor, must dominate the EC/DMC gap"
    )


def test_hydrogen_bonding_raises_g_factor() -> None:
    """Alcohols get g >> 1 from H-bond chain alignment."""
    ethanol = MoleculeContext.from_smiles("CCO")
    g = _kirkwood_g_factor(ethanol.mol, _count_dipole_groups(ethanol.mol))
    assert g > 2.0, f"ethanol g-factor {g:.2f} should reflect H-bond alignment"


def test_carboxylic_acid_dimer_suppresses_dielectric() -> None:
    """Acetic acid must stay low-eps despite a polar carboxyl group.

    Acetic acid (exp 6.2) and ethanol (exp 24.5) have near-identical monomer
    dipoles. The difference is that acids form closed antiparallel dimers.
    A model without that mechanism predicts them alike.
    """
    acetic = _eps("CC(=O)O")
    ethanol = _eps("CCO")
    assert acetic < 12.0, f"acetic acid dielectric {acetic:.1f} too high for a dimer-forming acid"
    assert acetic < ethanol, "acetic acid must be less polar than ethanol"


def test_nitrile_antiparallel_pairing_lowers_g() -> None:
    """Nitriles get g < 1 from antiparallel stacking."""
    acn = MoleculeContext.from_smiles("CC#N")
    g = _kirkwood_g_factor(acn.mol, _count_dipole_groups(acn.mol))
    assert g < 1.0, f"acetonitrile g-factor {g:.2f} should be below 1"


def test_symmetric_halocarbon_cancels() -> None:
    """CCl4 must be non-polar despite four polar C-Cl bonds."""
    assert _eps("ClC(Cl)(Cl)Cl") < 4.0
    assert _eps("ClCCl") > _eps("ClC(Cl)(Cl)Cl")  # DCM 8.93 vs CCl4 2.23


def test_nonpolar_hydrocarbons_near_optical_limit() -> None:
    """Alkanes and arenes must approach eps_inf ~ 2, not the 1.9 additive base."""
    for smiles in ("CCCCCC", "C1CCCCC1", "Cc1ccccc1", "c1ccccc1"):
        value = _eps(smiles)
        assert 1.8 < value < 3.0, f"{smiles} dielectric {value:.2f} outside non-polar range"


# --- Structural invariants -------------------------------------------------

def test_mcgowan_volume_tracks_molecular_size() -> None:
    """Molar volume must increase monotonically along a homologous series."""
    volumes = [
        _mcgowan_molar_volume(MoleculeContext.from_smiles(s).mol)
        for s in ("C", "CC", "CCC", "CCCC", "CCCCC")
    ]
    assert all(a < b for a, b in zip(volumes, volumes[1:], strict=False))


def test_optical_dielectric_physically_bounded() -> None:
    """eps_inf must stay in the physical range for organic liquids."""
    for smiles in ("C1COC(=O)O1", "CCCCCC", "O", "Cc1ccccc1", "CS(=O)C"):
        mol = MoleculeContext.from_smiles(smiles).mol
        value = _optical_dielectric(mol, _mcgowan_molar_volume(mol))
        assert 1.5 <= value <= 3.5, f"{smiles} eps_inf {value:.2f} unphysical"


def test_dielectric_always_exceeds_optical_limit() -> None:
    """Static eps >= eps_inf always: orientation adds to electronic polarization."""
    for smiles in ("CCCCCC", "COC(=O)OC", "C1COC(=O)O1", "CC#N", "O", "ClC(Cl)(Cl)Cl"):
        mol = MoleculeContext.from_smiles(smiles).mol
        epsilon_inf = _optical_dielectric(mol, _mcgowan_molar_volume(mol))
        assert _eps(smiles) >= epsilon_inf - 1e-6


def test_prediction_is_deterministic() -> None:
    """Repeated calls must agree exactly — no conformer or RNG dependence."""
    for smiles in ("C1COC(=O)O1", "CS(=O)C", "CCO"):
        assert _eps(smiles) == _eps(smiles)


def test_batch_matches_scalar() -> None:
    """The batch path must reproduce the scalar path bit-for-bit.

    Guards against the class of bug fixed in ADR-2026-08-07-02, where a
    hand-written MLX branch had silently drifted from the reference formulas.
    """
    from aurelius.scoring.oracle.gc import predict_dielectric_proxy_batch

    smiles = ["C1COC(=O)O1", "COC(=O)OC", "CS(=O)C", "CC#N", "O", "CCCCCC"]
    contexts = [MoleculeContext.from_smiles(s) for s in smiles]
    tpsa = np.array([c.tpsa for c in contexts], dtype=np.float32)

    batch = predict_dielectric_proxy_batch(None, tpsa, contexts)
    scalar = [predict_dielectric_constant(c) for c in contexts]
    np.testing.assert_allclose(batch, scalar, rtol=1e-6)
