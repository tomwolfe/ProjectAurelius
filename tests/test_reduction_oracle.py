"""Reduction-stability oracle tests (ADR-2026-08-10).

Validates the ΔSCF electron-affinity reduction axis:
  - the clean experimental EA set is confound-free by construction
  - the structural fallback beats the TOM LUMO it replaces, class-disjoint
  - ΔSCF ranking clears the permutation bar when xTB is present
  - known SEI chemistry is ordered correctly
  - graceful degradation without xTB
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from scipy.stats import spearmanr

from aurelius.scoring.oracle.quantum import predict_tom_orbitals
from aurelius.scoring.oracle.reduction import (
    ReductionOracle,
    _StructuralEAModel,
    calibrate_ea,
    compute_dscf_ea,
    has_xtb,
    load_experimental_ea,
    structural_features,
)

# Permutation-control bar measured on this label set (n=40): |rho| 95th
# percentile under shuffled labels. Nothing below this counts as signal.
PERMUTATION_RHO_P95 = 0.31


@pytest.fixture(scope="module")
def entries():
    data = load_experimental_ea()
    assert data, "experimental EA set failed to load"
    return data


def test_ea_set_is_provenance_free(entries):
    """Every label shares one reference string, so citation carries no signal."""
    refs = {e["reference"] for e in entries}
    assert len(refs) == 1, f"EA set must be single-source, found {len(refs)}"

    classes = {e["measurement_class"] for e in entries}
    assert classes <= {"PES", "ETE", "ETS", "ECD"}, f"unexpected method: {classes}"


def test_ea_set_has_usable_dynamic_range(entries):
    """A ranking target needs spread; a 0.5 eV window would not be rankable."""
    labels = np.array([e["ea_eV"] for e in entries])
    assert labels.max() - labels.min() > 3.0
    assert len(set(np.round(labels, 3))) >= 0.9 * len(labels), "too many tied labels"


def test_no_duplicate_molecules(entries):
    smiles = [Chem.CanonSmiles(e["smiles"]) for e in entries]
    assert len(set(smiles)) == len(smiles)


def test_structural_fallback_beats_tom_lumo(entries):
    """The xTB-free path must carry real ranking signal, class-disjoint.

    This is the guard against silently regressing to the uninformative
    frontier-orbital descriptor.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    mols = [Chem.MolFromSmiles(e["smiles"]) for e in entries]
    labels = np.array([e["ea_eV"] for e in entries])
    feats = np.vstack([structural_features(m) for m in mols])
    classes = np.array([e["chemical_class"] for e in entries])

    preds = np.full(len(labels), np.nan)
    for cls in np.unique(classes):
        test = classes == cls
        train = ~test
        if train.sum() < 8:
            continue
        sc = StandardScaler().fit(feats[train])
        model = Ridge(alpha=1.0).fit(sc.transform(feats[train]), labels[train])
        preds[test] = model.predict(sc.transform(feats[test]))

    mask = np.isfinite(preds)
    rho_ridge = spearmanr(preds[mask], labels[mask]).statistic

    tom = np.array([-predict_tom_orbitals(m)[1] for m in mols])
    rho_tom = spearmanr(tom, labels).statistic

    assert rho_ridge > PERMUTATION_RHO_P95, (
        f"fallback rho {rho_ridge:.3f} does not clear the permutation bar"
    )
    assert rho_ridge > rho_tom, (
        f"fallback rho {rho_ridge:.3f} must beat negated TOM LUMO {rho_tom:.3f}"
    )


def test_fallback_loo_is_honest(entries):
    model = _StructuralEAModel(entries)
    assert model.available
    m = model.loo_metrics()
    assert m["n"] == len(entries)
    assert m["spearman_rho"] > PERMUTATION_RHO_P95
    assert m["mae_eV"] < 1.0


def test_calibration_is_affine_and_rank_preserving():
    raw = np.linspace(-4.0, 9.0, 25)
    cal = np.array([calibrate_ea(v) for v in raw])
    assert spearmanr(raw, cal).statistic == pytest.approx(1.0)


def test_oracle_degrades_gracefully_without_xtb(tmp_path, monkeypatch):
    """No xTB must yield a well-formed record, never an exception."""
    monkeypatch.setattr("aurelius.scoring.oracle.reduction.has_xtb", lambda: False)
    oracle = ReductionOracle(cache_path=str(tmp_path / "cache.json"))
    result = oracle.evaluate(Chem.MolFromSmiles("O=C1OCCO1"))

    assert result["method"] == "structural_ridge"
    assert result["ea_eV"] is not None
    assert 0.0 <= result["confidence"] <= 1.0


def test_oracle_caches_by_canonical_smiles(tmp_path):
    oracle = ReductionOracle(cache_path=str(tmp_path / "cache.json"))
    a = oracle.evaluate(Chem.MolFromSmiles("COC(=O)OC"))
    b = oracle.evaluate(Chem.MolFromSmiles("O=C(OC)OC"))  # same molecule
    assert a["ea_eV"] == b["ea_eV"]
    assert oracle.report()["hits"] >= 1


def test_out_of_span_predictions_are_flagged(tmp_path):
    """Extrapolation must be visible, not silently reported as a measurement."""
    if not has_xtb():
        pytest.skip("xTB not available")
    oracle = ReductionOracle(cache_path=str(tmp_path / "cache.json"))
    # DME has a strongly negative EA, far below the calibrated span.
    result = oracle.evaluate(Chem.MolFromSmiles("COCCOC"))
    assert result["in_calibrated_span"] is False
    assert result["confidence"] < 0.6


@pytest.mark.skipif(not has_xtb(), reason="xTB not available")
def test_dscf_ranks_experimental_ea(entries):
    """The headline claim: ΔSCF EA ranks measured electron affinities."""
    mols = [Chem.MolFromSmiles(e["smiles"]) for e in entries]
    labels = np.array([e["ea_eV"] for e in entries])
    preds = np.array([compute_dscf_ea(m) for m in mols], dtype=float)

    mask = np.isfinite(preds)
    assert mask.sum() >= 0.9 * len(labels), "too many ΔSCF failures"

    rho = spearmanr(preds[mask], labels[mask]).statistic
    assert rho > 0.85, f"ΔSCF rho regressed to {rho:.3f}"

    slope, icpt = np.polyfit(preds[mask], labels[mask], 1)
    mae = np.mean(np.abs(labels[mask] - (slope * preds[mask] + icpt)))
    assert mae < 0.40, f"ΔSCF MAE regressed to {mae:.3f} eV"


@pytest.mark.skipif(not has_xtb(), reason="xTB not available")
def test_batch_matches_serial():
    """Parallel ΔSCF must be numerically identical to the serial path."""
    from aurelius.scoring.oracle.reduction import compute_dscf_ea_batch

    smiles = ["O=C1OCCO1", "COC(=O)OC", "COCCOC", "CC#N", "O=S1(=O)CCCC1"]
    mols = [Chem.MolFromSmiles(s) for s in smiles]

    serial = [compute_dscf_ea(m) for m in mols]
    parallel = compute_dscf_ea_batch(mols)

    assert len(parallel) == len(serial), "batch must preserve length and order"
    for s, p in zip(serial, parallel, strict=True):
        assert (s is None) == (p is None)
        if s is not None:
            assert s == pytest.approx(p, abs=1e-9)


def test_batch_handles_empty_input():
    from aurelius.scoring.oracle.reduction import compute_dscf_ea_batch

    assert compute_dscf_ea_batch([]) == []


@pytest.mark.skipif(not has_xtb(), reason="xTB not available")
def test_sei_additive_ordering():
    """Known battery chemistry: SEI-formers must be easier to reduce.

    FEC and VC are added to electrolytes precisely because they reduce before
    EC does; DME is far harder to reduce than any carbonate. A reduction axis
    that gets this backwards is unusable regardless of its benchmark score.
    """
    oracle = ReductionOracle()
    ea = {
        name: oracle.evaluate(Chem.MolFromSmiles(smi))["ea_eV"]
        for name, smi in [
            ("FEC", "O=C1OCC(F)O1"), ("VC", "O=C1OC=CO1"),
            ("EC", "O=C1OCCO1"), ("DMC", "COC(=O)OC"), ("DME", "COCCOC"),
        ]
    }
    assert ea["FEC"] > ea["EC"], "FEC must be easier to reduce than EC"
    assert ea["VC"] > ea["EC"], "VC must be easier to reduce than EC"
    assert ea["EC"] > ea["DME"], "ethers must be far harder to reduce"
    assert ea["DMC"] > ea["DME"]


@pytest.mark.skipif(not has_xtb(), reason="xTB not available")
def test_pipeline_ranks_by_electron_affinity():
    """The composite score must consume the validated reduction axis.

    Regression guard for the wiring, not just the oracle: before ADR-2026-08-10
    the largest single scoring weight (0.23) rode on the frontier LUMO, which
    sits at the permutation noise floor. Every molecule the agent ever selected
    on reduction grounds was selected against a descriptor carrying almost no
    information.
    """
    from aurelius.pipeline import AureliusPipeline

    pipeline = AureliusPipeline()
    pipeline.initialize()
    scores = {}
    for name, smiles in [
        ("EC", "O=C1OCCO1"),
        ("DME", "COCCOC"),
        ("benzoquinone", "O=C1C=CC(=O)C=C1"),
    ]:
        result = pipeline.screen_smiles(smiles)
        scores[name] = result["score"]["total_score"]

    assert scores["EC"] > scores["DME"], (
        "a carbonate in the SEI-forming band must outrank a glyme"
    )
    assert scores["EC"] > scores["benzoquinone"], (
        "an easily reduced quinone must not outrank a real electrolyte solvent"
    )
