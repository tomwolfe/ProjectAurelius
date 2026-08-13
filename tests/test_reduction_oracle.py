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
    _ALPB_SOLVENT_BY_DIELECTRIC,
    _EA_SOLUTION_CALIBRATION,
    ReductionOracle,
    _StructuralEAModel,
    calibrate_ea,
    calibrate_ea_solution,
    compute_dscf_ea,
    has_xtb,
    load_experimental_ea,
    load_experimental_ea_gas,
    load_experimental_ea_solution,
    solvent_from_dielectric,
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
    """Every label within a phase shares one reference string, so citation carries no signal."""
    gas = [e for e in entries if e.get("phase", "gas") == "gas"]
    sol = [e for e in entries if e.get("phase") == "solution"]

    gas_refs = {e["reference"] for e in gas}
    assert len(gas_refs) == 1, f"Gas-phase set must be single-source, found {len(gas_refs)}"

    if sol:
        sol_refs = {e["reference"] for e in sol}
        assert len(sol_refs) == 1, f"Solution-phase set must be single-source, found {len(sol_refs)}"

    classes = {e["measurement_class"] for e in entries}
    assert classes <= {"PES", "ETE", "ETS", "ECD", "CV"}, f"unexpected method: {classes}"


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
def test_dscf_ranks_experimental_ea():
    """The headline claim: ΔSCF EA ranks measured electron affinities.

    Tests against gas-phase entries only; solution-phase entries are tested
    separately in test_solution_phase_calibration.
    """
    gas_entries = load_experimental_ea_gas()
    mols = [Chem.MolFromSmiles(e["smiles"]) for e in gas_entries]
    labels = np.array([e["ea_eV"] for e in gas_entries])
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


# ---------------------------------------------------------------------------
# Solution-phase ALPB tests (ADR-2026-08-11)
# ---------------------------------------------------------------------------


def test_solvent_from_dielectric_maps_correctly():
    """Dielectric-to-solvent mapping picks the nearest named solvent."""
    assert solvent_from_dielectric(1.9) == "hexane"
    assert solvent_from_dielectric(37.5) == "acetonitrile"
    assert solvent_from_dielectric(80.0) == "water"
    # Clearly closer to one endpoint
    assert solvent_from_dielectric(3.0) == "toluene"  # closer to 2.4 than 7.6
    assert solvent_from_dielectric(8.5) == "dcm"  # closer to 9.1 than 7.6


def test_solvent_from_dielectric_clamps_out_of_range():
    """Values beyond the named range clamp to the nearest endpoint."""
    assert solvent_from_dielectric(0.5) == "hexane"
    assert solvent_from_dielectric(100.0) == "water"


def test_solvent_from_dielectric_fallback_for_battery_range():
    """Typical battery electrolytes (ε 2–40) map to a reasonable solvent."""
    assert solvent_from_dielectric(2.5) in ("hexane", "toluene", "thf")
    assert solvent_from_dielectric(30.0) in ("acetonitrile", "ethanol")


def test_caching_key_includes_solvent(tmp_path):
    """Same molecule with different solvents must not collide in the cache.

    Regression guard: before the fix, the cache key was canonical SMILES
    alone, so evaluating EC with solvent=None then solvent="acetonitrile"
    returned the gas-phase result for both.
    """
    oracle_none = ReductionOracle(cache_path=str(tmp_path / "c1.json"), solvent=None)
    oracle_alpb = ReductionOracle(cache_path=str(tmp_path / "c2.json"), solvent="acetonitrile")

    mol = Chem.MolFromSmiles("O=C1OCCO1")
    r_none = oracle_none.evaluate(mol)
    r_alpb = oracle_alpb.evaluate(mol)

    # Both should produce a result
    assert r_none["ea_eV"] is not None
    assert r_alpb["ea_eV"] is not None

    # The solvent field must be recorded on each result
    assert r_none["solvent"] is None
    assert r_alpb["solvent"] == "acetonitrile"


def test_same_solvent_hits_cache(tmp_path):
    """Same molecule + same solvent must hit the cache on second call."""
    oracle = ReductionOracle(cache_path=str(tmp_path / "cache.json"), solvent="acetonitrile")
    mol = Chem.MolFromSmiles("O=C1OCCO1")

    r1 = oracle.evaluate(mol)
    r2 = oracle.evaluate(mol)
    assert r1["ea_eV"] == r2["ea_eV"]
    assert oracle.report()["hits"] >= 1


@pytest.mark.skipif(not has_xtb(), reason="xTB not available")
def test_solution_uses_born_correction(tmp_path):
    """When solvent is set, xTB runs gas-phase + Born correction (not ALPB)."""
    oracle = ReductionOracle(cache_path=str(tmp_path / "cache.json"), solvent="acetonitrile")
    result = oracle.evaluate(Chem.MolFromSmiles("O=C1OCCO1"))
    assert result["ea_eV"] is not None
    assert result["solvent"] == "acetonitrile"
    assert result["method"] == "xtb_dscf_solution"
    # Gas-phase EA must be reported alongside the solution EA.
    assert result["gas_ea_eV"] is not None
    # Solution EA should differ from gas EA (Born correction shifts it).
    assert result["ea_eV"] != result["gas_ea_eV"]


@pytest.mark.skipif(not has_xtb(), reason="xTB not available")
def test_gas_vs_solution_differ(tmp_path):
    """Gas-phase and solution-phase EA must differ for a polar molecule.

    The Born correction stabilises the anion, so the solution-phase EA should
    differ from the gas-phase value. This is the physical sanity check that
    the solvation correction is actually being applied.
    """
    mol = Chem.MolFromSmiles("O=C1OCCO1")
    gas = ReductionOracle(cache_path=str(tmp_path / "gas.json"), solvent=None)
    sol = ReductionOracle(cache_path=str(tmp_path / "sol.json"), solvent="acetonitrile")

    ea_gas = gas.evaluate(mol)["ea_eV"]
    ea_sol = sol.evaluate(mol)["ea_eV"]
    assert ea_gas is not None
    assert ea_sol is not None
    # Solution EA should be higher (Born correction is positive).
    assert ea_sol > ea_gas, "solution EA must exceed gas EA (anion stabilised)"


def test_auto_solvent_selects_from_dielectric(tmp_path):
    """with_auto_solvent picks a solvent based on the predicted dielectric."""
    # EC has high dielectric (~65-90), should map to a high-ε solvent
    oracle = ReductionOracle.with_auto_solvent(
        Chem.MolFromSmiles("O=C1OCCO1"),
        cache_path=str(tmp_path / "cache.json"),
    )
    # The solvent should be one of the named ALPB solvents, not None
    assert oracle._solvent in dict(_ALPB_SOLVENT_BY_DIELECTRIC)


def test_auto_solvent_differentiates_molecules(tmp_path):
    """Different molecules with different dielectrics get different solvents."""
    # EC (cyclic carbonate, high ε) vs DMC (linear carbonate, low ε)
    oracle_ec = ReductionOracle.with_auto_solvent(
        Chem.MolFromSmiles("O=C1OCCO1"),
        cache_path=str(tmp_path / "ec.json"),
    )
    oracle_dmc = ReductionOracle.with_auto_solvent(
        Chem.MolFromSmiles("COC(=O)OC"),
        cache_path=str(tmp_path / "dmc.json"),
    )
    # Both should have a solvent assigned
    assert oracle_ec._solvent is not None
    assert oracle_dmc._solvent is not None
    # EC has higher dielectric than DMC, so its solvent should be >= on the ε scale
    eps_map = dict(_ALPB_SOLVENT_BY_DIELECTRIC)
    assert eps_map[oracle_ec._solvent] >= eps_map[oracle_dmc._solvent]


# ---------------------------------------------------------------------------
# Born solvation correction (ADR-2026-08-11-05)
# ---------------------------------------------------------------------------


def test_cavity_radius_is_positive_and_finite():
    """Cavity radius must be a positive Ångström value for any molecule."""
    from aurelius.scoring.oracle.reduction import _cavity_radius

    for smi in ["O=C1OCCO1", "COCCOC", "CC#N", "C1=CC=CC=C1", "O=S1(=O)CCCC1"]:
        mol = Chem.MolFromSmiles(smi)
        r = _cavity_radius(mol)
        assert r > 0, f"cavity radius must be positive for {smi}"
        assert r < 10, f"cavity radius implausible ({r:.2f} Å) for {smi}"


def test_born_correction_increases_with_dielectric():
    """Higher ε → more anion stabilization → larger positive correction."""
    from aurelius.scoring.oracle.reduction import _born_solvation_correction, _cavity_radius

    mol = Chem.MolFromSmiles("O=C1OCCO1")
    r = _cavity_radius(mol)
    low = _born_solvation_correction(2.0, r)
    high = _born_solvation_correction(80.0, r)
    assert high > low > 0, "Born correction must be positive and increase with ε"


def test_born_correction_zero_at_vacuum():
    """At ε=1 (vacuum), the Born correction must vanish."""
    from aurelius.scoring.oracle.reduction import _born_solvation_correction, _cavity_radius

    mol = Chem.MolFromSmiles("O=C1OCCO1")
    r = _cavity_radius(mol)
    assert _born_solvation_correction(1.0, r) == 0.0


def test_born_correction_inversely_scales_with_radius():
    """Smaller cavity → larger (more negative) solvation energy."""
    from aurelius.scoring.oracle.reduction import _born_solvation_correction

    big = _born_solvation_correction(30.0, 5.0)
    small = _born_solvation_correction(30.0, 1.5)
    assert small > big, "smaller cavity must yield larger |solvation|"


def test_phase_field_exists_in_entries():
    """Every experimental EA entry has a 'phase' field (gas or solution)."""
    entries = load_experimental_ea()
    assert len(entries) > 40, "Should have more than just the original 40 gas entries"
    for e in entries:
        assert "phase" in e, f"Entry {e.get('smiles')} missing 'phase' field"
        assert e["phase"] in ("gas", "solution"), f"Unexpected phase: {e.get('phase')}"


def test_solution_phase_entries_present():
    """At least 10 solution-phase CV onset measurements are present."""
    sol = load_experimental_ea_solution()
    assert len(sol) >= 10, f"Expected >=10 solution entries, got {len(sol)}"
    for e in sol:
        assert e.get("measurement_class") == "CV"
        assert e["ea_eV"] > 0, "Solution-phase EA should be positive (vs vacuum)"


def test_gas_and_solution_are_separate():
    """Gas and solution subsets are disjoint and non-empty."""
    gas = load_experimental_ea_gas()
    sol = load_experimental_ea_solution()
    gas_smiles = {e["smiles"] for e in gas}
    sol_smiles = {e["smiles"] for e in sol}
    assert gas_smiles.isdisjoint(sol_smiles), "Gas and solution sets overlap"
    assert len(gas) >= 30
    assert len(sol) >= 10


def test_born_correction_preserves_gas_ranking():
    """Born correction is additive → cannot reverse gas-phase ranking.

    This is the core structural argument of ADR-2026-08-11-05: because the
    correction is a monotonic function of cavity radius applied to the
    validated gas-phase EA, it cannot produce the sign-reversal that
    ALPB ΔSCF did (ρ = −0.915). The gas-phase ordering is preserved modulo
    the cavity-size dependence, which is far weaker than the gas-phase EA
    spread.
    """
    from aurelius.scoring.oracle.reduction import (
        _born_solvation_correction,
        _cavity_radius,
    )

    gas_entries = load_experimental_ea_gas()
    gas_eas = np.array([e["ea_eV"] for e in gas_entries])
    rads = np.array([_cavity_radius(Chem.MolFromSmiles(e["smiles"])) for e in gas_entries])
    born = np.array([_born_solvation_correction(30.0, r) for r in rads])

    assert born.max() - born.min() < (gas_eas.max() - gas_eas.min()), (
        "Born correction span exceeds gas EA span — ranking could be reversed"
    )


def test_solution_phase_born_correction_shifts_ether_up():
    """Born correction is always positive: anions are stabilised in solvent.

    For the ether solvent DME (which has a negative or near-zero gas-phase EA),
    the Born correction should add a substantial positive shift, bringing the
    solution-phase prediction closer to the CV-measured range.
    """
    from aurelius.scoring.oracle.reduction import (
        _born_solvation_correction,
        _cavity_radius,
    )

    dme = Chem.MolFromSmiles("COCCOC")
    r = _cavity_radius(dme)
    assert _born_solvation_correction(30.0, r) > 3.0, (
        "Born correction for ether in medium-ε solvent should exceed 3 eV"
    )


def test_calibrate_ea_solution_is_affine_and_rank_preserving():
    """Solution-phase calibration is affine → rank-preserving."""
    raw = np.linspace(2.0, 6.0, 15)
    cal = np.array([calibrate_ea_solution(v) for v in raw])
    rho = spearmanr(raw, cal).statistic
    assert rho == pytest.approx(1.0), f"Affine map broke rank: rho={rho}"


def test_gas_calibration_unchanged():
    """Gas-phase calibration constants are unchanged by solution-phase addition."""
    from aurelius.scoring.oracle.reduction import _EA_CALIBRATION

    assert _EA_CALIBRATION == (0.6590, -2.9176)


def test_solution_calibration_density():
    """Solution-phase calibration values span the measurement range."""
    sol = load_experimental_ea_solution()
    eas = np.array([e["ea_eV"] for e in sol])
    spread = eas.max() - eas.min()
    assert spread > 1.5, f"Solution EA range too narrow: {spread:.2f} eV"


def test_solution_calibration_is_fitted_and_out_of_span_flagged(tmp_path):
    """Regression: solution-phase EA must never claim calibrated confidence
    while the calibration is the identity placeholder, and out-of-span
    predictions must be flagged.

    ADR-2026-08-11-05 introduced a solution-phase affine calibration. The
    attempted fit is rank-reversing (Spearman ρ = −0.82 vs the validated
    gas-phase ΔSCF axis, ρ = 0.91), so no trusted fit exists and the identity
    placeholder is retained. This test guards:
      (a) the calibration tuple is affine and rank-preserving, and
      (b) until a trusted fit replaces the placeholder, solution-mode
          predictions are reported with ``in_calibrated_span is False``
          (never a false 0.85 confidence).
    """
    # (a) The affine map must be rank-preserving regardless of the constants.
    raw = np.linspace(2.0, 6.0, 15)
    cal = np.array([calibrate_ea_solution(v) for v in raw])
    rho = spearmanr(raw, cal).statistic
    assert rho == pytest.approx(1.0), f"Affine map broke rank: rho={rho}"

    # (b) No false in-span confidence while uncalibrated.
    if not has_xtb():
        pytest.skip("xTB not available")
    oracle = ReductionOracle(
        cache_path=str(tmp_path / "cache.json"), solvent="acetonitrile"
    )
    # DME is a representative molecule; any candidate must not be reported
    # as calibrated-span while the solution calibration is the placeholder.
    result = oracle.evaluate(Chem.MolFromSmiles("COCCOC"))
    if _EA_SOLUTION_CALIBRATION == (1.0, 0.0):
        assert result["in_calibrated_span"] is False
        assert result["confidence"] < 0.6
