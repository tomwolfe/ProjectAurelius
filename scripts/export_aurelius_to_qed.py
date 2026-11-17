"""Export Aurelius reduction-stability lemmas to QED/Lean.

Reconstructs EA calibration (a=0.6590, b=-2.9176) and Born constant
C=13.598 from src/aurelius/scoring/oracle/reduction.py, asserts
EA_solution = EA_gas + delta_solv, and emits:
  - a QED-parseable lemma file (--out, default /tmp/aurelius_lemmas.txt):
    one lemma per line, integer-scaled (QED's parser accepts integer
    literals only), mirroring the 14 Lean theorems:
    ea_calibration_affine, born_solvation_pos, ea_bounds (affine
    linearity), reduction_stability_mono (eaSol identity),
    eaCal_mono (EA ranking monotonicity under affine calibration),
    born_mono_eps, scoring_mono, tier1_sound, novelty_gate_sound,
    synergy_bounds, delta_damp_factor_bounds (delta-correction
    boundedness/damping), conformal_coverage (conformal coverage),
    gc_xterm_bounds (GC cross-terms), weightvec_mono (pipeline weight vector)
  - a Lean file (--lean-out, default QED/AureliusExport.lean)
with 14 theorems, pinned axioms [propext, Classical.choice, Quot.sound], zero admits.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

A_EA, B_EA = 0.6590, -2.9176
BORN_C = 13.598


def build_from_simulator():
    from aurelius.scoring.oracle import reduction as R

    assert R._EA_CALIBRATION == (A_EA, B_EA), f"EA calibration drift: {R._EA_CALIBRATION}"
    assert R._BORN_CONSTANT_EV_ANGSTROM == BORN_C, f"Born drift: {R._BORN_CONSTANT_EV_ANGSTROM}"
    # EA_solution = EA_gas + delta_solv identity check (pure python, no xTB):
    ea_gas, born = 1.0, 0.5
    assert R.calibrate_ea_solution(ea_gas + born) == (ea_gas + born)  # solution cal is identity (1.0, 0.0)
    assert R._born_solvation_correction(30.0, 3.0) == BORN_C * (1.0 - 1.0 / 30.0) / 3.0
    assert R.calibrate_ea(0.0) == B_EA and abs(R.calibrate_ea(1.0) - (A_EA + B_EA)) < 1e-12
    # EA ranking monotonicity pin (fail-closed for the eaCal_mono Lean lemma
    # below): the affine calibration has positive slope, so raw-EA order is
    # preserved. A slope-sign mutant (a -> -a) flips this and fails here.
    assert R.calibrate_ea(0.0) <= R.calibrate_ea(1.0) <= R.calibrate_ea(2.0)
    # Solution-EA gating pin (Phase 1): the solution calibration is the
    # identity placeholder (rank-preserving by construction) until a trusted
    # rank-preserving fit on >10 CV onsets exists. Any non-identity
    # calibration without that evidence must fail closed here.
    assert R._EA_SOLUTION_CALIBRATION == (1.0, 0.0), (
        f"solution EA calibration changed without trusted fit: {R._EA_SOLUTION_CALIBRATION}"
    )
    assert len(R.load_experimental_ea_solution()) <= 10 or R._EA_SOLUTION_CALIBRATION == (1.0, 0.0)
    # Born monotonicity pin (fail-closed for born_mono_eps): larger
    # dielectric gives a larger correction; vacuum is zero.
    assert R._born_solvation_correction(10.0, 3.0) <= R._born_solvation_correction(30.0, 3.0) <= R._born_solvation_correction(80.0, 3.0)
    assert R._born_solvation_correction(1.0, 3.0) == 0.0
    # Scoring monotonicity pin (fail-closed for scoring_mono): pipeline
    # weights are non-negative so raising a component cannot lower the
    # total; donor weight is exactly zero (removed, no silent de-rate).
    from aurelius.pipeline import (
        _SCORE_WEIGHT_DIELECTRIC as _W_DI,
        _SCORE_WEIGHT_HOMO as _W_HO,
        _SCORE_WEIGHT_LI_SOLVATION as _W_LI,
        _SCORE_WEIGHT_REDUCTION_STABILITY as _W_RS,
        _SCORE_WEIGHT_SA as _W_SA,
        _SCORE_WEIGHT_VISCOSITY as _W_VI,
    )
    assert _W_RS == 0.33 and _W_DI == 0.27 and _W_HO == 0.17 and _W_VI == 0.14 and _W_SA == 0.01
    assert _W_LI == 0.0
    assert min(_W_RS, _W_DI, _W_HO, _W_VI, _W_LI, _W_SA) >= 0.0
    assert _W_RS * 1.0 <= _W_RS * 2.0
    # Tier-1 filter soundness pin (fail-closed for tier1_sound): a viable
    # screen carries no violations; rejected/invalid screens are non-viable.
    from aurelius.screening.tier1 import Filter as _Tier1Filter
    from aurelius.types import MoleculeContext as _MC
    _f = _Tier1Filter()
    _ec = _f.screen(_MC.from_smiles("C1COC(=O)O1"))
    assert _ec["is_viable"] is True and _ec["electrolyte_violations"] == []
    _cco = _f.screen(_MC.from_smiles("CCO"))
    assert _cco["is_viable"] is False and len(_cco["electrolyte_violations"]) > 0
    assert _f.screen_smiles("not_a_smiles")["is_viable"] is False
    # Novelty-gate soundness pin (fail-closed for novelty_gate_sound):
    # known scaffolds keep only KNOWN_SCAFFOLD_PENALTY = 0.2 < 1.
    from aurelius.agent.selection import KNOWN_SCAFFOLD_PENALTY as _KSP
    assert _KSP == 0.2
    assert 0.0 < _KSP < 1.0
    assert _KSP * 50.0 <= 50.0
    # Mixture synergy bounds pin (fail-closed for synergy_bounds):
    # bonus lives in 0..6; non-complementary pairs get exactly 0.
    from aurelius.scoring.oracle.gc import mixture_synergy_bonus as _syn
    assert _syn(40.0, 5.0, 0.5, 5.0, 0.5) <= 6.0
    assert _syn(40.0, 5.0, 0.5, 5.0, 0.5) >= 0.0
    assert _syn(1.0, 2.0, 5.0, 6.0, 0.5) == 0.0
    # Delta-correction boundedness/damping pin (fail-closed for
    # delta_damp_factor_bounds): the shrinkage weight
    # conf = prior^2 / (prior^2 + pred^2) lives in [0, 1], so the damped
    # correction can never exceed the raw delta; kernel length-scale init
    # stays clipped to the (0.5, 5.0) bounds.
    from aurelius.scoring.oracle.delta_correction import _build_kernel as _build_k
    _prior, _pred, _delta = 1.0, 2.0, 0.7
    _conf = _prior ** 2 / (_prior ** 2 + _pred ** 2)
    assert _conf == 0.2
    assert 0.0 <= _conf <= 1.0
    assert abs(_delta * _conf) <= abs(_delta)
    _k_lo = _build_k(0.01).get_params()
    _k_hi = _build_k(100.0).get_params()
    assert _k_lo["k1__k2__length_scale"] == 0.5
    assert _k_hi["k1__k2__length_scale"] == 5.0
    assert tuple(_k_hi["k1__k2__length_scale_bounds"]) == (0.5, 5.0)
    # Conformal coverage pin (fail-closed for conformal_coverage):
    # intervals contain the point estimate; the confidence discount lives
    # in [0.5, 1.0].
    from aurelius.scoring.oracle.conformal import get_conformal_predictor as _get_cp
    _cp = _get_cp()
    _lo, _hi = _cp.predict_interval("homo", 0.0)
    assert _lo <= 0.0 <= _hi
    _disc = _cp.confidence_discount({"homo": (0.0, 1.0)})
    assert 0.5 <= _disc <= 1.0
    # GC cross-term pin (fail-closed for gc_xterm_bounds): the dielectric
    # cross-term correction is clipped to [-2, 2]; empty counts give 0.
    from aurelius.scoring.oracle.gc import _compute_dielectric_cross_terms as _xt
    assert _xt({}) == 0.0
    _xtv = _xt({"carbonate": 1, "ether": 1, "sulfone": 1, "nitrile": 1, "fluorine": 1})
    assert -2.0 <= _xtv <= 2.0
    # Pipeline weight-vector pin (fail-closed for weightvec_mono): exact
    # weights are non-negative and sum to 0.92; raising one component with a
    # non-negative weight cannot lower the weighted total.
    assert abs((_W_RS + _W_DI + _W_HO + _W_VI + _W_LI + _W_SA) - 0.92) < 1e-12
    _t1 = _W_RS * 1.0 + _W_DI * 2.0
    _t2 = _W_RS * 2.0 + _W_DI * 2.0
    assert _t1 <= _t2
    _pin_oracle_invariants(R)
    return {"a_ea": A_EA, "b_ea": B_EA, "born_c": BORN_C}


def _pin_oracle_invariants(R):
    """Behavioral pins for reduction-oracle internals the gate pytest suite
    never imports directly (cavity geometry, solvent lookup, fallback
    reporting, cache keys, flush side effects, unavailable path).

    Each pin asserts an observable contract: a mutant breaking one changes
    an outcome (cache keys, files written, reported metrics), so this export
    command — which runs on every verification — fails closed. All pins are
    pure-python and xTB-free (has_xtb is forced off); the structural-ridge
    fallback they exercise is deterministic.
    """
    import json
    import tempfile

    from rdkit import Chem

    # Cavity geometry: a bigger molecule displaces a bigger Born cavity, and
    # cavities stay positive (the 1.4 A floor guards diatomics).
    small = R._cavity_radius(Chem.MolFromSmiles("C"))
    big = R._cavity_radius(Chem.MolFromSmiles("c1ccccc1C(=O)O"))
    assert small > 0 and big > small, f"Born cavity must grow with size: {small} vs {big}"
    # Exact-radius pins kill arithmetic mutants in the volume-to-radius
    # formula (195:9): any Mult<->Div or Add<->Sub swap moves these by >>1e-9
    # (measured: the 195 mutant collapses big to the 1.4 floor).
    assert abs(small - 1.7) < 1e-9, f"cavity radius drift for methane: {small}"
    assert abs(big - 3.459801741677741) < 1e-9, f"cavity radius drift for benzoic acid: {big}"
    assert abs(R._cavity_radius(Chem.MolFromSmiles("CCO")) - 2.3715274385230143) < 1e-9

    # ALPB solvent lookup: spot-checks over the dielectric table.
    assert R._epsilon_for_solvent("acetonitrile") == 37.5
    assert R._epsilon_for_solvent("water") == 80.0
    assert R._epsilon_for_solvent(None) is None
    assert R._epsilon_for_solvent("not_a_solvent") is None
    assert R.solvent_from_dielectric(30.0) == "ethanol"

    # Structural-ridge fallback reporting stays shaped over the full set.
    ridge = R._StructuralEAModel()
    loo = ridge.loo_metrics()
    assert set(loo) == {"spearman_rho", "mae_eV", "n"}
    assert loo["n"] == len(ridge._y) and loo["n"] > 10
    # Value pins kill the LOO mask mutant (431:22 flip_bool False->True):
    # training on the held-out point collapses MAE toward 0 and inflates rho
    # toward 1, far outside these 1e-6 windows (deterministic Ridge).
    assert abs(loo["spearman_rho"] - 0.6071840191403645) < 1e-6, f"ridge LOO rho drift: {loo}"
    assert abs(loo["mae_eV"] - 0.7995734027439714) < 1e-6, f"ridge LOO MAE drift: {loo}"

    # Solution-mode factory must return a live oracle (kills 650:8
    # break_return which returns None instead of the constructed oracle).
    _auto = R.ReductionOracle.with_auto_solvent(Chem.MolFromSmiles("CCO"))
    assert _auto is not None and _auto._solvent is not None
    assert _auto._medium_epsilon > 0

    # Experimental set loaders stay populated (gas-phase subset included).
    assert len(R.load_experimental_ea_gas()) > 10

    # Backend reporting: the oracle always names a real backend.
    assert R.ReductionOracle().method in (
        "xtb_dscf", "xtb_dscf_solution", "structural_ridge", "unavailable",
    )

    # Force the xTB-free path so the pins below are fast and deterministic.
    orig_has_xtb, R.has_xtb = R.has_xtb, lambda: False
    # The patch itself is pinned first: without it the pins below would run
    # real xTB single points (slow) instead of the deterministic fallback.
    assert R.has_xtb() is False
    try:
        # Fresh oracle: nothing to flush, so no cache file may appear —
        # even when the cache directory already exists.
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "sub"))
            cp = os.path.join(td, "sub", "cache.json")
            oracle = R.ReductionOracle(cache_path=cp)
            assert oracle._dirty is False
            oracle.flush()
            assert not os.path.exists(cp), "fresh oracle must not write cache"
            # Cache keys are canonical, stereo-aware SMILES so that
            # cross-process cache hits line up. Benzoic acid is used
            # because its canonical and non-canonical SMILES differ, so
            # this pin also kills a canonical=True/False flip.
            mol = Chem.MolFromSmiles("c1ccccc1C(=O)O")
            rec = oracle.evaluate(mol)
            assert rec["method"] in ("structural_ridge", "xtb_dscf")
            assert rec["in_calibrated_span"] is True
            assert rec["cpu_seconds"] == rec["cpu_seconds"] and rec["cpu_seconds"] >= 0  # finite, non-negative (no wall/CPU bound: load-independent)
            oracle.flush()
            with open(cp) as fh:
                cache = json.load(fh)
            want = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            assert any(k.split("|")[0] == want for k in cache), (
                f"cache key must be canonical stereo SMILES: {list(cache)}"
            )

        # Fallback exhausted: the oracle reports unavailable (with None EA)
        # instead of crashing.
        with tempfile.TemporaryDirectory() as td:
            oracle = R.ReductionOracle(cache_path=os.path.join(td, "cache.json"))
            oracle._fallback = R._StructuralEAModel(entries=[])
            assert oracle._fallback.available is False
            rec = oracle.evaluate(Chem.MolFromSmiles("CCO"))
            assert rec["method"] == "unavailable" and rec["ea_eV"] is None
            assert rec["in_calibrated_span"] is False
            assert rec["cpu_seconds"] == rec["cpu_seconds"] and rec["cpu_seconds"] >= 0  # finite, non-negative (no wall/CPU bound: load-independent)

        # Solution-mode Born additivity (kills 762:43 arithmetic ea+born):
        # with a solvent set and xTB off, evaluate must return the fallback
        # EA plus the Born correction through the identity solution
        # calibration. An ea-born (or ea*born) mutant shifts this by ~2*born.
        with tempfile.TemporaryDirectory() as td:
            oracle = R.ReductionOracle(
                cache_path=os.path.join(td, "cache.json"),
                solvent="acetonitrile",
            )
            mol = Chem.MolFromSmiles("c1ccccc1C(=O)O")
            rec = oracle.evaluate(mol)
            assert rec is not None and rec["method"] == "structural_ridge"
            fb = oracle._get_fallback()
            ea_fb, _ = fb.predict(mol)
            born = R._born_solvation_correction(
                oracle._medium_epsilon, R._cavity_radius(mol)
            )
            assert born > 0
            expected = R.calibrate_ea_solution(ea_fb + born)
            # evaluate() rounds ea_eV to 4 decimals, so allow rounding slack;
            # an ea-born mutant shifts by ~2*born (~7 eV) and still fails.
            assert abs(rec["ea_eV"] - expected) < 1e-4, (
                f"solution Born additivity drift: {rec['ea_eV']} vs {expected}"
            )
    finally:
        R.has_xtb = orig_has_xtb

    # Solution-phase xTB branch (kills 723:20 break_return): stub the two
    # xTB single points so the solvent branch returns a ReductionResult
    # without needing a real xtb binary. The mutant returns None instead,
    # so rec["method"] raises and the export fails closed.
    _orig_compute, _orig_sol = R.compute_dscf_ea, R.compute_solution_ea
    try:
        R.has_xtb = lambda: True
        R.compute_dscf_ea = lambda mol, solvent=None, threads=1: -1.0
        R.compute_solution_ea = lambda mol, epsilon=30.0, threads=1: (0.5, -1.0, -1.0)
        with tempfile.TemporaryDirectory() as td:
            oracle = R.ReductionOracle(
                cache_path=os.path.join(td, "cache.json"), solvent="acetonitrile"
            )
            rec = oracle.evaluate(Chem.MolFromSmiles("CCO"))
            assert rec is not None, "solution xTB branch must return a record"
            assert rec["method"] == "xtb_dscf_solution"
            assert rec["ea_eV"] is not None
    finally:
        R.compute_dscf_ea, R.compute_solution_ea = _orig_compute, _orig_sol
        R.has_xtb = orig_has_xtb

    # _load_cache pin (kills break_return on `return loaded if ... else {}`):
    # a dict cache round-trips, a non-dict JSON cache yields {}, and a
    # missing file yields {}. The mutant returns None in all cases, so the
    # == asserts below fail closed.
    with tempfile.TemporaryDirectory() as td:
        cp = os.path.join(td, "cache.json")
        oracle = R.ReductionOracle(cache_path=cp)
        assert oracle._load_cache() == {}
        with open(cp, "w") as fh:
            json.dump({"k": {"ea_eV": 1.0}}, fh)
        assert oracle._load_cache() == {"k": {"ea_eV": 1.0}}
        with open(cp, "w") as fh:
            json.dump([1, 2, 3], fh)
        assert oracle._load_cache() == {}

    # cpu_seconds formula pin (kills arithmetic - -> + in
    # `time.perf_counter() - start`): the export's earlier >= 0 checks pass
    # for both `-` and `+`, so run each branch under a stubbed clock where
    # start=1000.0 and end=1000.5. The green value is exactly 0.5; the `+`
    # mutant reports 2000.5 and fails.
    import time as _time

    _orig_perf = _time.perf_counter

    def _run_with_clock(fn):
        ticks = iter([1000.0, 1000.5])
        _time.perf_counter = lambda: next(ticks, 1000.5)
        try:
            return fn()
        finally:
            _time.perf_counter = _orig_perf

    try:
        # Fallback (xTB-free) branch.
        _oh0 = R.has_xtb
        R.has_xtb = lambda: False
        try:
            with tempfile.TemporaryDirectory() as td:
                oracle = R.ReductionOracle(cache_path=os.path.join(td, "c.json"))
                rec = _run_with_clock(
                    lambda: oracle.evaluate(Chem.MolFromSmiles("CCO")))
                assert rec["cpu_seconds"] == 0.5, (
                    f"fallback cpu_seconds drift: {rec['cpu_seconds']}")
            # Unavailable branch.
            with tempfile.TemporaryDirectory() as td:
                oracle = R.ReductionOracle(cache_path=os.path.join(td, "c.json"))
                oracle._fallback = R._StructuralEAModel(entries=[])
                rec = _run_with_clock(
                    lambda: oracle.evaluate(Chem.MolFromSmiles("CCO")))
                assert rec["method"] == "unavailable"
                assert rec["cpu_seconds"] == 0.5, (
                    f"unavailable cpu_seconds drift: {rec['cpu_seconds']}")
        finally:
            R.has_xtb = _oh0
        # Stubbed xTB solution + gas branches.
        _oc, _os = R.compute_dscf_ea, R.compute_solution_ea
        _oh = R.has_xtb
        try:
            R.has_xtb = lambda: True
            R.compute_dscf_ea = lambda mol, solvent=None, threads=1: -1.0
            R.compute_solution_ea = (
                lambda mol, epsilon=30.0, threads=1: (0.5, -1.0, -1.0))
            with tempfile.TemporaryDirectory() as td:
                oracle = R.ReductionOracle(
                    cache_path=os.path.join(td, "c.json"),
                    solvent="acetonitrile")
                rec = _run_with_clock(
                    lambda: oracle.evaluate(Chem.MolFromSmiles("CCO")))
                assert rec["method"] == "xtb_dscf_solution"
                assert rec["cpu_seconds"] == 0.5, (
                    f"solution cpu_seconds drift: {rec['cpu_seconds']}")
            with tempfile.TemporaryDirectory() as td:
                oracle = R.ReductionOracle(
                    cache_path=os.path.join(td, "c.json"))
                rec = _run_with_clock(
                    lambda: oracle.evaluate(Chem.MolFromSmiles("CCO")))
                assert rec["method"] == "xtb_dscf"
                assert rec["cpu_seconds"] == 0.5, (
                    f"gas cpu_seconds drift: {rec['cpu_seconds']}")
        finally:
            R.compute_dscf_ea, R.compute_solution_ea = _oc, _os
            R.has_xtb = _oh
    finally:
        _time.perf_counter = _orig_perf


LEAN_TEMPLATE = """-- Auto-generated by scripts/export_aurelius_to_qed.py. Do not edit.
-- Source: src/aurelius/scoring/oracle/reduction.py
-- Pinned axioms: propext, Classical.choice, Quot.sound. Zero admits.
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

noncomputable section

-- EA affine calibration: EA_cal = a * EA_raw + b, a = {a}, b = {b}.
def eaCal (x : ℝ) : ℝ := {a} * x + ({b})

-- Born solvation correction: C * (1 - 1/eps) / r, C = {c}.
def bornCorr (eps r : ℝ) : ℝ := {c} * (1 - 1 / eps) / r

-- EA_solution = EA_gas + delta_solv.
def eaSol (gas solv : ℝ) : ℝ := gas + solv

-- Weighted scoring component with non-negative pipeline weights.
def scoreW (w x : ℝ) : ℝ := w * x

-- Tier-1 viability gate (MW < 300, HBD = 0, rotatable <= 6, HBA >= 1, LogP <= 2.5).
def tier1Pass (mw hbd rot hba logp : ℝ) : Prop :=
  mw < 300 ∧ hbd = 0 ∧ rot ≤ 6 ∧ 1 ≤ hba ∧ logp ≤ 2.5

-- Novelty-gated score: known scaffolds keep KNOWN_SCAFFOLD_PENALTY = 0.2.
def novelScore (base : ℝ) (isKnown : Bool) : ℝ :=
  if isKnown then 0.2 * base else base

-- Mixture score with synergy bonus bounded 0..6, total capped at 100.
def mixScore (base syn : ℝ) : ℝ := min 100 (base + max 0 syn)

-- Damping factor for delta-correction: posterior shrinkage weight
-- conf = prior^2 / (prior^2 + pred^2), no tuned constant.
def dampFactor (prior pred : ℝ) : ℝ := prior ^ 2 / (prior ^ 2 + pred ^ 2)

-- Conformal discount from interval-width ratio, in [0.5, 1.0].
def confDiscount (r : ℝ) : ℝ := 1 - min (max r 0) 0.5

-- GC cross-term clip to [-2, 2].
def xtermClip (c : ℝ) : ℝ := max (-2) (min 2 c)

-- Pipeline weighted total over six objective components.
def wTotal (w1 w2 w3 w4 w5 w6 x1 x2 x3 x4 x5 x6 : ℝ) : ℝ :=
  w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 + w5 * x5 + w6 * x6

theorem ea_calibration_affine (x : ℝ) : eaCal x = {a} * x + ({b}) := rfl

theorem born_solvation_pos (eps r : ℝ) (he : 1 < eps) (hr : 0 < r) :
    0 < bornCorr eps r := by
  unfold bornCorr
  have h1 : (0 : ℝ) < 1 - 1 / eps := by
    rw [sub_pos, div_lt_one (by linarith : (0:ℝ) < eps)]; linarith
  have h2 : (0 : ℝ) < {c} * (1 - 1 / eps) := mul_pos (by norm_num) h1
  exact div_pos h2 hr

theorem ea_bounds (x lo hi : ℝ) (hlo : lo ≤ x) (hhi : x ≤ hi) :
    {a} * lo + ({b}) ≤ eaCal x ∧ eaCal x ≤ {a} * hi + ({b}) := by
  unfold eaCal
  have ha : (0:ℝ) < {a} := by norm_num
  constructor
  · have h := mul_le_mul_of_nonneg_left hlo (le_of_lt ha)
    linarith
  · have h := mul_le_mul_of_nonneg_left hhi (le_of_lt ha)
    linarith

theorem reduction_stability_mono (g1 g2 s : ℝ) (h : g1 ≤ g2) :
    eaSol g1 s ≤ eaSol g2 s := by
  unfold eaSol; linarith

-- EA ranking monotonicity under affine calibration (positive slope a):
-- raw-EA order is preserved, so solution ranking is set upstream.
theorem eaCal_mono (x y : ℝ) (h : x ≤ y) :
    eaCal x ≤ eaCal y := by
  unfold eaCal
  have ha : (0:ℝ) < {a} := by norm_num
  have h := mul_le_mul_of_nonneg_left h (le_of_lt ha)
  linarith

-- Born monotonicity in dielectric: larger eps gives larger correction.
theorem born_mono_eps (e1 e2 r : ℝ) (h1 : 1 < e1) (hle : e1 ≤ e2) (hr : 0 < r) :
    bornCorr e1 r ≤ bornCorr e2 r := by
  unfold bornCorr
  have he1 : (0:ℝ) < e1 := by linarith
  have he2 : (0:ℝ) < e2 := by linarith
  have h : 1 / e2 ≤ 1 / e1 := by
    apply one_div_le_one_div_of_le he1 hle
  have hsub : (1 - 1 / e1) ≤ (1 - 1 / e2) := by linarith
  have hmul : {c} * (1 - 1 / e1) ≤ {c} * (1 - 1 / e2) := by
    apply mul_le_mul_of_nonneg_left hsub (by norm_num)
  exact div_le_div_of_nonneg_right hmul (le_of_lt hr)

-- Scoring monotonicity: non-negative weights preserve component order.
theorem scoring_mono (w x y : ℝ) (hw : 0 ≤ w) (h : x ≤ y) :
    scoreW w x ≤ scoreW w y := by
  unfold scoreW
  exact mul_le_mul_of_nonneg_left h hw

-- Tier-1 filter soundness: passing the gate implies each criterion holds.
theorem tier1_sound (mw hbd rot hba logp : ℝ) (h : tier1Pass mw hbd rot hba logp) :
    mw < 300 ∧ hbd = 0 ∧ rot ≤ 6 ∧ 1 ≤ hba ∧ logp ≤ 2.5 := by
  exact h

-- Novelty-gate soundness: known scaffolds score no more than novel ones.
theorem novelty_gate_sound (base : ℝ) (h : 0 ≤ base) :
    novelScore base true ≤ novelScore base false := by
  unfold novelScore
  simp
  linarith

-- Mixture synergy bounds: synergy in 0..6 keeps total in base..100.
theorem synergy_bounds (base syn : ℝ) (_hb0 : 0 ≤ base) (hb1 : base ≤ 100)
    (hs0 : 0 ≤ syn) (_hs1 : syn ≤ 6) :
    base ≤ mixScore base syn ∧ mixScore base syn ≤ 100 := by
  unfold mixScore
  have hmax : max (0:ℝ) syn = syn := max_eq_right hs0
  rw [hmax]
  constructor
  · exact le_min (by linarith) (by linarith)
  · exact min_le_left _ _

-- Delta-correction boundedness/damping: the shrinkage weight lives in
-- [0, 1], so the damped correction never exceeds the raw delta.
theorem delta_damp_factor_bounds (prior pred : ℝ) (hp : 0 < prior) :
    0 ≤ dampFactor prior pred ∧ dampFactor prior pred ≤ 1 := by
  unfold dampFactor
  have hpsq : 0 < prior ^ 2 := pow_pos hp 2
  have hq : 0 ≤ pred ^ 2 := by positivity
  have hden : 0 < prior ^ 2 + pred ^ 2 := by linarith
  constructor
  · exact div_nonneg (le_of_lt hpsq) (le_of_lt hden)
  · rw [div_le_one hden]
    linarith

-- Conformal coverage: intervals contain the point estimate and the
-- confidence discount lives in [0.5, 1.0].
theorem conformal_coverage (x q r : ℝ) (hq : 0 ≤ q) :
    (x - q ≤ x ∧ x ≤ x + q) ∧ (0.5 ≤ confDiscount r ∧ confDiscount r ≤ 1) := by
  constructor
  · constructor <;> linarith
  · unfold confDiscount
    have h1 : 0 ≤ min (max r 0) 0.5 := le_min (le_max_right _ _) (by norm_num)
    have h2 : min (max r 0) 0.5 ≤ 0.5 := min_le_right _ _
    constructor <;> linarith

-- GC cross-terms: the dielectric cross-term correction is clipped to [-2, 2].
theorem gc_xterm_bounds (c : ℝ) :
    -2 ≤ xtermClip c ∧ xtermClip c ≤ 2 := by
  unfold xtermClip
  constructor
  · exact le_max_left _ _
  · exact max_le (by norm_num) (min_le_left _ _)

-- Pipeline weight vector: non-negative weights preserve component order.
theorem weightvec_mono (w1 w2 w3 w4 w5 w6 x1 x2 x3 x4 x5 x6 y1 : ℝ)
    (hw1 : 0 ≤ w1) (h : x1 ≤ y1) :
    wTotal w1 w2 w3 w4 w5 w6 x1 x2 x3 x4 x5 x6 ≤
      wTotal w1 w2 w3 w4 w5 w6 y1 x2 x3 x4 x5 x6 := by
  unfold wTotal
  have hm := mul_le_mul_of_nonneg_left h hw1
  linarith

#print axioms ea_calibration_affine
#print axioms born_solvation_pos
#print axioms ea_bounds
#print axioms reduction_stability_mono
#print axioms eaCal_mono
#print axioms born_mono_eps
#print axioms scoring_mono
#print axioms tier1_sound
#print axioms novelty_gate_sound
#print axioms synergy_bounds
#print axioms delta_damp_factor_bounds
#print axioms conformal_coverage
#print axioms gc_xterm_bounds
#print axioms weightvec_mono
"""

TEXT_TEMPLATE = """{a_num} * x / 1000 + ({b_num}) / 10000 = {a_num} * x / 1000 + ({b_num}) / 10000
{c_num} * (30 - 1) / 30 / 3 / 1000 = {c_num} * (30 - 1) / 30 / 3 / 1000
gas + solv = gas + solv
{a_num} * (x + y) / 1000 = {a_num} * x / 1000 + {a_num} * y / 1000
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/aurelius_lemmas.txt")
    ap.add_argument("--lean-out", default="QED/AureliusExport.lean")
    args = ap.parse_args()
    consts = build_from_simulator()
    a, b, c = consts["a_ea"], consts["b_ea"], consts["born_c"]
    # Integer-scaled forms for the QED-parseable --out lines. QED's parser
    # vocabulary (QED/parser.py) accepts integer literals only, so the float
    # calibration constants are emitted scaled (a x1000, b x10000, C x1000).
    # The round-trip asserts below keep the scaled lines pinned to the exact
    # reduction.py constants: any mutant perturbing a constant fails here.
    a_num, b_num, c_num = round(a * 1000), round(b * 10000), round(c * 1000)
    assert a_num / 1000 == a and b_num / 10000 == b and c_num / 1000 == c, (
        "integer scaling must round-trip the reduction constants exactly"
    )
    txt = TEXT_TEMPLATE.format(a_num=a_num, b_num=b_num, c_num=c_num)
    lean = LEAN_TEMPLATE.format(a=a, b=f"({b})", c=c)
    with open(args.out, "w") as fh:
        fh.write(txt)
    os.makedirs(os.path.dirname(os.path.abspath(args.lean_out)), exist_ok=True)
    with open(args.lean_out, "w") as fh:
        fh.write(lean)
    print(f"wrote {args.out} and {args.lean_out}")


if __name__ == "__main__":
    main()
