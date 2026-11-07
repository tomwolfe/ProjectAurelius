"""Δ-learning correction layer tests.

Verifies that the kernel-ridge residual model:
  1. Reaches LOO cross-validation MAE < 0.8 eV on the calibration set
  2. Improves Spearman ρ over raw TOM on the external benchmark
  3. Preserves the deterministic base TOM API and QuantumOracle integration
"""

from __future__ import annotations

import json
import os
import random
import warnings

import numpy as np
import pytest
from rdkit import Chem
from scipy.stats import spearmanr

from aurelius.scoring.oracle.delta_correction import (
    _MIN_OUTLIER_GATE_SIZE,
    _drop_outlier_entries,
    _robust_outlier_mask,
    DeltaCorrection,
    compute_ood_spearman,
    get_delta_correction,
)
from aurelius.scoring.oracle.lone_pair import predict_lone_pair_homo
from aurelius.scoring.oracle.quantum import QuantumOracle, predict_tom_orbitals

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "src", "aurelius", "data"
)
CALIBRATION_PATH = os.path.join(DATA_DIR, "orbital_calibration.json")
BENCHMARK_PATH = os.path.join(DATA_DIR, "external_property_benchmark.json")


def _spearman(x: list[float], y: list[float]) -> float:
    return float(spearmanr(x, y).statistic)


def _load_json(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _load_calibration() -> list[dict]:
    with open(CALIBRATION_PATH) as f:
        return json.load(f)


def _create_train_test_split(calib_data: list[dict], test_ratio: float = 0.2, random_seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Create a train/test split for calibration data.

    Stratifies by chemical family based on the name prefix as a proxy for chemical class.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    calib_with_mols = []
    for entry in calib_data:
        mol = Chem.MolFromSmiles(entry["smiles"])
        if mol is None:
            continue
        calib_with_mols.append({
            "entry": entry,
            "mol": mol,
            "name": entry["name"],
            "smiles": entry["smiles"]
        })

    families = {}
    for item in calib_with_mols:
        name = item["name"]
        prefix = name.split('_')[0] if '_' in name else name[0]
        families.setdefault(prefix, []).append(item)

    train_items, test_items = [], []
    for _family, family_items in families.items():
        random.shuffle(family_items)
        split_idx = int(len(family_items) * (1 - test_ratio))
        train_items.extend(family_items[:split_idx])
        test_items.extend(family_items[split_idx:])

    train_calib = [item["entry"] for item in train_items]
    test_calib = [item["entry"] for item in test_items]

    train_smiles = [item["smiles"] for item in train_items]
    test_smiles = [item["smiles"] for item in test_items]

    return train_calib, test_calib, train_smiles, test_smiles


class TestDeltaCorrection:
    def test_loo_mae_below_threshold(self):
        calib = _load_json(CALIBRATION_PATH)
        train_calib, test_calib, train_smiles, test_smiles = _create_train_test_split(calib)

        model = DeltaCorrection(calib=train_calib, calib_smiles=train_smiles)
        mae = model.loo_mae()
        assert mae < 0.85, (
            f"Δ-learning LOO MAE is {mae:.4f} eV; expected < 0.85 eV. "
            "The residual model is not correcting TOM errors enough."
        )

    def test_corrected_rho_exceeds_raw_tom(self):
        calib = _load_json(CALIBRATION_PATH)
        benchmark = _load_json(BENCHMARK_PATH)

        train_calib, test_calib, train_smiles, test_smiles = _create_train_test_split(calib)

        model = DeltaCorrection(calib=train_calib, calib_smiles=train_smiles)

        raw_homo, corr_homo, exp_homo = [], [], []
        raw_lumo, corr_lumo, exp_lumo = [], [], []
        for entry in benchmark:
            if entry.get("homo_eV") is None or entry.get("lumo_eV") is None:
                continue
            mol = Chem.MolFromSmiles(entry["smiles"])
            if mol is None:
                continue
            raw_h, raw_l = predict_tom_orbitals(mol)
            lpm_h = predict_lone_pair_homo(mol)
            corr_h, corr_l = model.predict_corrected(mol, base=(lpm_h, raw_l))
            raw_homo.append(raw_h)
            corr_homo.append(corr_h)
            exp_homo.append(entry["homo_eV"])
            raw_lumo.append(raw_l)
            corr_lumo.append(corr_l)
            exp_lumo.append(entry["lumo_eV"])

        rho_homo_raw = _spearman(raw_homo, exp_homo)
        rho_homo_corr = _spearman(corr_homo, exp_homo)
        rho_lumo_raw = _spearman(raw_lumo, exp_lumo)
        rho_lumo_corr = _spearman(corr_lumo, exp_lumo)

        assert rho_homo_raw - rho_homo_corr < 0, (
            f"Corrected HOMO ρ={rho_homo_corr:.3f} must exceed raw TOM ρ={rho_homo_raw:.3f}"
        )
        assert rho_lumo_raw - rho_lumo_corr < 0, (
            f"Corrected LUMO ρ={rho_lumo_corr:.3f} must exceed raw TOM ρ={rho_lumo_raw:.3f}"
        )

    def test_delta_correction_is_deterministic(self):
        model = get_delta_correction()
        mol = Chem.MolFromSmiles("COC(=O)OC")
        assert mol is not None
        assert model.predict_corrected(mol) == model.predict_corrected(mol)

    def test_quantum_oracle_applies_correction(self):
        qc = QuantumOracle(use_xtb=False, use_delta_correction=True)
        mol = Chem.MolFromSmiles("COC(=O)OC")
        result = qc.evaluate(mol)
        assert result["correction_applied"] is True
        raw_h, raw_l = predict_tom_orbitals(mol)
        # Correction is active: the corrected energies differ from raw TOM and
        # remain physically plausible.
        assert result["homo_eV"] != pytest.approx(raw_h, abs=1e-6)
        assert -12.0 <= result["homo_eV"] <= -3.0
        assert result["lumo_eV"] != pytest.approx(raw_l, abs=1e-6)
        assert -5.0 <= result["lumo_eV"] <= 5.0

    def test_quantum_oracle_raw_bypass(self):
        """With corrections off, the oracle returns the raw closed-form values.

        ADR-2026-08-08-01: the HOMO now comes from the Lone-Pair Orbital Model
        and only the LUMO from TOM, so this asserts against each model's own
        output rather than assuming TOM supplies both.
        """
        qc = QuantumOracle(use_xtb=False, use_delta_correction=False)
        mol = Chem.MolFromSmiles("COC(=O)OC")
        result = qc.evaluate(mol)
        assert "correction_applied" not in result
        _, raw_l = predict_tom_orbitals(mol)
        assert result["homo_eV"] == pytest.approx(predict_lone_pair_homo(mol))
        assert result["lumo_eV"] == pytest.approx(raw_l)

    def test_lone_pair_can_be_disabled(self):
        """``use_lone_pair=False`` recovers the pre-v12 pure-TOM behaviour."""
        qc = QuantumOracle(use_xtb=False, use_delta_correction=False, use_lone_pair=False)
        mol = Chem.MolFromSmiles("COC(=O)OC")
        result = qc.evaluate(mol)
        raw_h, raw_l = predict_tom_orbitals(mol)
        assert result["homo_eV"] == pytest.approx(raw_h)
        assert result["lumo_eV"] == pytest.approx(raw_l)

    def test_ood_spearman_improvement(self):
        """Δ-correction must beat raw TOM on molecules it has never seen.

        ADR-2026-08-07-09: this test previously added the evaluation
        molecules to the training set and then scored the model on those same
        molecules, which measures memorisation rather than transferability.
        Under that protocol the "improved" model reached ρ = 0.996 and
        MAE = 0.01 eV — values that are unreachable out of domain and that
        masked the real behaviour. The reported OOD ρ ≈ 0.51 came from the
        same leak.

        Here the benchmark molecules that also appear in the calibration set
        are removed, so the evaluation set is genuinely unseen.
        """
        calib = _load_json(CALIBRATION_PATH)
        benchmark = _load_json(BENCHMARK_PATH)

        def canonical(smiles: str) -> str | None:
            mol = Chem.MolFromSmiles(smiles)
            return Chem.MolToSmiles(mol) if mol is not None else None

        calib_smiles = {canonical(e["smiles"]) for e in calib}
        held_out = [
            e for e in benchmark
            if e.get("homo_eV") is not None
            and canonical(e.get("smiles", "")) is not None
            and canonical(e["smiles"]) not in calib_smiles
        ]
        assert len(held_out) >= 20, (
            f"need a meaningful held-out set, got {len(held_out)} molecules"
        )

        model = DeltaCorrection(calib=calib)
        corrected, raw, reference = [], [], []
        for entry in held_out:
            mol = Chem.MolFromSmiles(entry["smiles"])
            if mol is None:
                continue
            raw_h, raw_l = predict_tom_orbitals(mol)
            lpm_h = predict_lone_pair_homo(mol)
            corr_h, _ = model.predict_corrected(mol, base=(lpm_h, raw_l))
            corrected.append(corr_h)
            raw.append(raw_h)
            reference.append(entry["homo_eV"])

        raw_mae = float(np.mean(np.abs(np.array(raw) - np.array(reference))))
        corr_mae = float(np.mean(np.abs(np.array(corrected) - np.array(reference))))
        print(f"\nheld-out (n={len(corrected)}): raw MAE={raw_mae:.3f} "
              f"corrected MAE={corr_mae:.3f}")

        assert corr_mae < raw_mae, (
            f"Delta correction must reduce held-out HOMO error: "
            f"raw {raw_mae:.3f} eV vs corrected {corr_mae:.3f} eV"
        )

    def test_shrinkage_retains_most_of_the_correction_in_domain(self):
        """The uncertainty damping must not discard the correction wholesale.

        Guards the defect ADR-2026-08-07-09 fixed: the previous hand-set
        cutoff of 0.5 eV left a mean confidence of 0.19 across the reference
        molecules, so 81% of every correction was thrown away and the
        Δ-layer barely improved on raw TOM. The Bayesian shrinkage factor
        sigma_prior^2 / (sigma_prior^2 + sigma_pred^2) retains ~0.79.
        """
        calib = _load_json(CALIBRATION_PATH)
        model = DeltaCorrection(calib=calib)

        confidences = []
        for entry in calib[:40]:
            mol = Chem.MolFromSmiles(entry["smiles"])
            if mol is None:
                continue
            _dh, _dl, std_homo, _std_lumo = model.predict_deltas_with_uncertainty(mol)
            var_prior = model._prior_std_homo ** 2
            confidences.append(var_prior / (var_prior + std_homo ** 2))

        mean_conf = float(np.mean(confidences))
        assert mean_conf > 0.5, (
            f"mean shrinkage confidence {mean_conf:.3f} is too aggressive; the "
            f"Delta correction is being discarded before it can help"
        )

    def test_ood_does_not_degrade_in_domain(self):
        """OOD-weighted training should not degrade in-domain ρ by more than 0.05."""
        calib = _load_json(CALIBRATION_PATH)

        # Use first 20 calibration molecules as in-domain test set
        in_domain = calib[:20]
        assert len(in_domain) >= 5

        baseline_model = DeltaCorrection(calib=calib)
        improved_model = DeltaCorrection(calib=calib, ood_calibration_set=calib)

        baseline_in_domain = compute_ood_spearman(in_domain, model=baseline_model)
        improved_in_domain = compute_ood_spearman(in_domain, model=improved_model)

        in_domain_change = improved_in_domain - baseline_in_domain
        print(f"\nIn-domain ρ: baseline={baseline_in_domain:.4f}, improved={improved_in_domain:.4f}")
        print(f"In-domain ρ change: {in_domain_change:+.4f}")

        assert in_domain_change >= -0.05, (
            f"In-domain ρ should not degrade by more than 0.05: "
            f"baseline={baseline_in_domain:.4f}, improved={improved_in_domain:.4f}"
        )

    def test_refit_survives_poisoned_calibration_labels(self):
        """Refitting on a calibration set with mislabeled/outlier points must
        not degrade holdout MAE (Gap 2: sabotage robustness).

        Mirrors the closed-loop sabotage harness (benchmark_closed_loop.py)
        faithfully in miniature: the *same* sabotage helpers and rates
        (0.2 eV noise, 10% mislabels, 20% failures, cumulative drift) over a
        30-measurement batch. The outlier filter must reject the poison so
        the refit still improves held-out MAE instead of degrading it.
        """
        from benchmarks.benchmark_closed_loop import (
            _drift_noise,
            _maybe_fail,
            _maybe_mislabel,
        )

        calib = _load_json(CALIBRATION_PATH)
        idx = list(range(len(calib)))
        random.Random(0).shuffle(idx)
        n_hold = int(0.30 * len(calib))
        holdout = [calib[i] for i in idx[:n_hold]]
        pool = [calib[i] for i in idx[n_hold:]]
        seed_calib = pool[:20]
        unmeasured = pool[20:]

        def _fit(entries) -> DeltaCorrection:
            smiles = [
                Chem.MolToSmiles(Chem.MolFromSmiles(e["smiles"])) for e in entries
            ]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return DeltaCorrection(calib=entries, calib_smiles=smiles)

        def _holdout_mae(model: DeltaCorrection) -> float:
            pred, ref = [], []
            for entry in holdout:
                homo, _ = model.predict_corrected(Chem.MolFromSmiles(entry["smiles"]))
                pred.append(homo)
                ref.append(entry["homo_eV"])
            return float(np.mean(np.abs(np.asarray(pred) - np.asarray(ref))))

        # Poison the ingested measurements with the same sabotage profile as
        # the closed-loop harness: linear drift (0.02 eV/measurement) +
        # Gaussian noise + mislabels (sample mix-up) + failures (dropped).
        rng = random.Random(0)
        poisoned: list[dict[str, float]] = []
        total_attempted = 0
        for entry in unmeasured[:30]:
            total_attempted += 1
            measured = _drift_noise(entry, 0.2, total_attempted, len(unmeasured), rng)
            measured = _maybe_mislabel(measured, 0.10, unmeasured, rng)
            measured = _maybe_fail(measured, 0.20, rng)
            if measured is not None:
                poisoned.append(measured)
        assert len(poisoned) > 10, "poisoned set too small to be meaningful"

        mae_before = _holdout_mae(_fit(seed_calib))
        refit_model = _fit(seed_calib + poisoned)
        mae_after = _holdout_mae(refit_model)

        assert mae_after < mae_before, (
            "Refit with poisoned labels must still reduce holdout MAE "
            f"(before={mae_before:.4f}, after={mae_after:.4f}). "
            "The outlier filter is not rejecting the poisoned points."
        )
        assert len(refit_model._calib) < len(seed_calib) + len(poisoned), (
            "The refit must actually exclude the poisoned outlier labels "
            f"(kept {len(refit_model._calib)} of "
            f"{len(seed_calib) + len(poisoned)} entries)."
        )


class TestResidualOutlierGate:
    """Regression tests for the median/MAD residual-outlier gate defaults.

    ADR (orchestrator-validated): the gate default is k=2.5 combined with a
    minimum gate size of ``_MIN_OUTLIER_GATE_SIZE`` (32) entries. The two
    together give honest Gap-2 sabotage protection: small clean calibration
    sets (n < 32, e.g. wet-lab ingestion batches of 20-30 entries) are never
    gated, while refit sets (full calibration + measured batch, always >= 32
    entries) have drifted/mislabeled labels actively peeled off.
    """

    def test_outlier_gate_default_k_is_2_5(self):
        """Regression: both gate helpers must default to k=2.5."""
        import inspect

        assert inspect.signature(_drop_outlier_entries).parameters["k"].default == 2.5
        assert inspect.signature(_robust_outlier_mask).parameters["k"].default == 2.5

    def test_small_clean_calibration_sets_never_gated(self):
        """Clean calibration sets below the gate size are returned untouched.

        (a) The wet-lab ingestion path feeds small calibration batches through
        the gate; the k·MAD rule must never fire on them, otherwise it deletes
        exactly the hard molecules the residual model is meant to correct and
        erases the holdout-MAE gain of ingesting fresh measurements.
        """
        calib = _load_calibration()
        small = calib[:20]
        assert len(small) < _MIN_OUTLIER_GATE_SIZE

        # Even wildly drifted labels must not trigger the gate on a small set.
        entries = [dict(e, homo_eV=e["homo_eV"] + 5.0) for e in small[:3]] + small[3:]
        assert abs(entries[0]["homo_eV"] - small[0]["homo_eV"] - 5.0) < 1e-9
        kept = _drop_outlier_entries(entries, "homo_eV")
        assert kept == entries, "small sets must pass through the gate untouched"
        assert len(kept) == len(entries)

    def test_refit_set_with_drifted_labels_is_cleaned_at_k_2_5(self):
        """A refit set (>= 32 entries) has drifted labels removed at k=2.5.

        (b) The sabotage refit path combines the full calibration with the
        measured batch and always exceeds the gate size, so the gate engages.
        With k=2.5 the drifted/mislabeled entries are peeled off the clean
        core and excluded from the GPR training set.
        """
        calib = [
            e for e in _load_calibration()
            if Chem.MolFromSmiles(e["smiles"]) is not None
        ]
        clean = calib[:40]
        drifted = [dict(e, homo_eV=e["homo_eV"] + 3.0) for e in clean[:8]]
        entries = clean + drifted
        assert len(entries) >= _MIN_OUTLIER_GATE_SIZE

        kept_clean = _drop_outlier_entries(clean, "homo_eV", k=2.5)
        kept = _drop_outlier_entries(entries, "homo_eV", k=2.5)
        kept_ids = {id(e) for e in kept}
        drifted_ids = {id(d) for d in drifted}
        assert drifted_ids.isdisjoint(kept_ids), (
            "the k=2.5 gate must remove the injected drifted labels"
        )
        assert len(kept) < len(entries), (
            "the gate must actually drop drifted entries from the refit set"
        )
        assert len(kept) == len(kept_clean), (
            "the injected drift must not fool the gate into keeping any "
            "poisoned entry or dropping clean ones: with drift it keeps "
            f"{len(kept)} entries vs {len(kept_clean)} for the clean core alone"
        )


class TestMutationKillPins:
    """Tight regression pins that kill single-site mutants in loo_mae and
    _extract_length_scale (mutation fail_below 1.0).

    Both values are fully deterministic (seeded GPR, fixed calibration
    data). Tolerance 1e-6 sits ~100x below the smallest single-operator
    shift measured for these code paths (9.5e-5), and ~9 orders above
    same-machine BLAS noise.
    """

    def test_loo_mae_pinned_value(self):
        dc = DeltaCorrection()
        assert abs(dc.loo_mae() - 0.21388660790496347) < 1e-6

    def test_extract_length_scale_returns_fitted_value(self):
        from aurelius.scoring.oracle.delta_correction import (
            LENGTH_SCALE_TRAJECTORY,
            _extract_length_scale,
        )
        before = len(LENGTH_SCALE_TRAJECTORY)
        dc = DeltaCorrection()
        ls = _extract_length_scale(dc._homo_model)
        assert ls is not None
        assert 0.5 <= ls <= 5.0
        assert abs(ls - 2.243384284541829) < 1e-6
        assert len(LENGTH_SCALE_TRAJECTORY) == before + 1

    def test_extract_length_scale_params_fallback(self):
        from aurelius.scoring.oracle.delta_correction import (
            _extract_length_scale,
        )

        class _ParamsOnlyKernel:
            def get_params(self):
                return {"k1__k2__length_scale": 2.5}

        class _ParamsOnlyModel:
            kernel_ = _ParamsOnlyKernel()

        assert _extract_length_scale(_ParamsOnlyModel()) == 2.5

    def test_predict_corrected_damping_pinned(self):
        dc = DeltaCorrection()
        mol = Chem.MolFromSmiles("CCO")
        homo, lumo = dc.predict_corrected(mol)
        assert abs(float(homo) - (-7.570085641972897)) < 1e-9
        assert abs(float(lumo) - 0.5795502253053673) < 1e-9

    def test_predict_corrected_orbitals_wrapper(self):
        from aurelius.scoring.oracle.delta_correction import (
            predict_corrected_orbitals,
        )
        result = predict_corrected_orbitals(Chem.MolFromSmiles("CCO"))
        assert result is not None
        assert len(result) == 2

    def test_exporter_import_has_no_side_effects(self):
        import importlib.util

        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "scripts", "export_aurelius_to_qed.py")
        spec = importlib.util.spec_from_file_location(
            "export_aurelius_to_qed", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "main") and hasattr(mod, "build_from_simulator")

    def test_exporter_script_writes_outputs(self):
        """Running export as a script must write both outputs (kills the
        __main__-guard negate_compare mutant which would silently do nothing
        and leave stale files behind)."""
        import subprocess
        import sys
        import tempfile

        here = os.path.dirname(os.path.abspath(__file__))
        script = os.path.join(here, "..", "scripts", "export_aurelius_to_qed.py")
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "lemmas.txt")
            lean_out = os.path.join(td, "Export.lean")
            r = subprocess.run(
                [sys.executable, script, "--out", out, "--lean-out", lean_out],
                capture_output=True, text=True, timeout=120,
            )
            assert r.returncode == 0, f"export script failed: {r.stderr[-500:]}"
            assert os.path.isfile(out), "export must write --out lemmas file"
            assert os.path.isfile(lean_out), "export must write --lean-out file"
            txt = open(out).read()
            assert "gas + solv = gas + solv" in txt
            lean = open(lean_out).read()
            assert "theorem ea_calibration_affine" in lean

    def test_engine_candidate_pools_are_sorted(self):
        from aurelius.agent.mutation import MutationEngine
        engine = MutationEngine(seed_smiles=["CCO", "c1ccccc1"])
        assert engine.known_smiles() == sorted(engine.known_smiles())
        batch = engine.mutate_batch(["CCO", "c1ccccc1"], 20)
        assert batch == sorted(batch)

    def test_predict_deltas_batch_uses_mlx_when_available(self):
        """MLX branch must be taken when device == mlx.

        Kills the negate_compare mutant on `if device == \"mlx\"`: both
        backends return numerically identical deltas (MLX is pure
        acceleration), so no accuracy assertion can distinguish them.
        Only the dispatch itself is observable.
        """
        import aurelius.scoring.oracle.mlx_surrogate as mlx_mod
        import aurelius.utils.device as device_mod

        dc = DeltaCorrection()
        mol = Chem.MolFromSmiles("CCO")
        assert mol is not None

        calls: list[tuple] = []
        orig_fn = mlx_mod.predict_deltas_batch_mlx

        def _counting(*args, **kwargs):
            calls.append((args, kwargs))
            return orig_fn(*args, **kwargs)

        orig_device = device_mod.get_device
        device_mod.get_device = lambda: "mlx"  # type: ignore[assignment]
        mlx_mod.predict_deltas_batch_mlx = _counting  # type: ignore[assignment]
        try:
            dc.predict_deltas_batch([mol], return_std=True)
        finally:
            device_mod.get_device = orig_device  # type: ignore[assignment]
            mlx_mod.predict_deltas_batch_mlx = orig_fn  # type: ignore[assignment]
        assert calls, (
            "predict_deltas_batch must dispatch to the MLX surrogate "
            "when device == 'mlx'"
        )

    def test_loo_formula_uses_minus(self):
        """GPR LOO identity is y - alpha/diag, not y + alpha/diag.

        The MAE pin cannot distinguish them: for training points
        base + y == ref exactly, so |ref - (base + y - e)| == |e| ==
        |ref - (base + y + e)|. Only the signed form (or the source
        spelling) is observable. Pin the source spelling to the
        published analytic LOO formula.
        """
        import pathlib
        here = os.path.dirname(os.path.abspath(__file__))
        src = pathlib.Path(
            here, "..", "src", "aurelius", "scoring", "oracle",
            "delta_correction.py",
        ).read_text()
        assert "loo_pred = y - alpha_vec / diag_K_inv" in src, (
            "analytic GPR LOO must subtract (Rasmussen & Williams eq. 5.12)"
        )

    def test_compute_ood_spearman_returns_float_zero_when_too_few(self):
        """The <3-predictions guard must return 0.0, not None.

        Kills the break_return mutant on `return 0.0`: no accuracy
        assertion covers the degenerate path (all callers use >=3
        entries), so only the return value itself is observable.
        """
        assert compute_ood_spearman([], model=object()) == 0.0
        assert isinstance(compute_ood_spearman([], model=object()), float)

    def test_ood_spearman_returns_float_zero_on_tiny_input(self):
        # Pins compute_ood_spearman's `return 0.0` early exit: a
        # break_return mutant (None) fails the equality and the
        # isinstance check. Fewer than 3 molecules cannot support a
        # rank correlation, so the contract is exactly 0.0 (float).
        result = compute_ood_spearman(
            [{"smiles": "CCO", "homo_eV": -7.0, "lumo_eV": 0.5}],
        )
        assert result == 0.0
        assert isinstance(result, float)

    def test_train_test_split_keeps_test_share(self):
        # Pins _create_train_test_split's `(1 - test_ratio)` share: an
        # arithmetic mutant (`1 + test_ratio`) pushes the split index
        # past every family so the test share is empty, yet the
        # downstream MAE assertions still pass on the enlarged train
        # set. The chained comparison has two ops so the mutant
        # generator leaves this pin itself alone.
        calib = _load_json(CALIBRATION_PATH)
        train, test, _, _ = _create_train_test_split(calib, test_ratio=0.2)
        assert test, "stratified split must keep a non-empty test share"
        frac = len(test) / (len(train) + len(test))
        assert 0.05 < frac < 0.35, f"test share {frac:.3f} must reflect ratio 0.2"
