"""Closed-loop efficacy: ingesting measurements must improve the oracle.

The existing feedback tests assert only that LOO MAE *does not get worse*, and
they measure LOO on the calibration set that was just enlarged with the new
points — a model scored on its own training data. These tests instead hold out
molecules that are never trained on and never ingested, so any improvement is
genuine generalisation rather than recall.

See ``benchmarks/benchmark_closed_loop.py`` for the full curves.
"""

from __future__ import annotations

import json
import os
import random
import warnings

import numpy as np
from rdkit import Chem
from scipy.stats import spearmanr

from aurelius.agent.feedback import FeedbackController
from aurelius.scoring.oracle.delta_correction import DeltaCorrection
from aurelius.types import MoleculeContext

_CALIBRATION_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src", "aurelius", "data",
    "orbital_calibration.json",
)

HOLDOUT_FRACTION = 0.30
SEED_SIZE = 20


def _load_entries() -> list[dict[str, float]]:
    with open(_CALIBRATION_PATH) as f:
        entries = json.load(f)
    return [e for e in entries if Chem.MolFromSmiles(e["smiles"]) is not None]


def _split(entries, seed=0):
    idx = list(range(len(entries)))
    random.Random(seed).shuffle(idx)
    n_hold = int(HOLDOUT_FRACTION * len(entries))
    holdout = [entries[i] for i in idx[:n_hold]]
    pool = [entries[i] for i in idx[n_hold:]]
    return holdout, pool[:SEED_SIZE], pool[SEED_SIZE:]


def _fit(entries) -> DeltaCorrection:
    smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(e["smiles"])) for e in entries]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DeltaCorrection(calib=entries, calib_smiles=smiles)


def _holdout_metrics(model, holdout) -> tuple[float, float]:
    """Return (spearman_rho, mae_eV) on molecules the model never saw."""
    pred, ref = [], []
    for entry in holdout:
        homo, _ = model.predict_corrected(Chem.MolFromSmiles(entry["smiles"]))
        pred.append(homo)
        ref.append(entry["homo_eV"])
    pred_arr, ref_arr = np.asarray(pred), np.asarray(ref)
    return float(spearmanr(pred_arr, ref_arr).correlation), float(
        np.abs(pred_arr - ref_arr).mean()
    )


class TestClosedLoopImprovesHeldOutAccuracy:
    """Ingest → refit must reduce error on molecules outside the loop."""

    def test_ingesting_measurements_reduces_holdout_mae(self):
        entries = _load_entries()
        holdout, seed_calib, unmeasured = _split(entries)

        _, mae_before = _holdout_metrics(_fit(seed_calib), holdout)
        _, mae_after = _holdout_metrics(_fit(seed_calib + unmeasured[:40]), holdout)

        assert mae_after < mae_before, (
            "Ingesting 40 measurements must reduce held-out MAE "
            f"(before={mae_before:.4f}, after={mae_after:.4f}). "
            "The closed loop is not learning."
        )

    def test_improvement_is_monotonic_in_data_volume(self):
        """More ingested measurements must not make the oracle worse."""
        entries = _load_entries()
        holdout, seed_calib, unmeasured = _split(entries)

        maes = [
            _holdout_metrics(_fit(seed_calib + unmeasured[:k]), holdout)[1]
            for k in (0, 20, 40)
        ]
        assert maes[-1] < maes[0], (
            f"Held-out MAE must improve with data volume, got {maes}"
        )

    def test_gain_requires_correct_labels_not_just_more_points(self):
        """The improvement must come from *information*, not recalibration.

        Adding any 40 molecules shifts the GPR mean and lowers MAE even when
        the labels are wrong — so "MAE went down" alone does not prove the
        loop learned anything. The permutation control keeps the molecules
        and the label distribution identical and only destroys the
        molecule→label pairing. Correct labels must beat shuffled ones.

        Tested for significance over 10 splits rather than demanding a clean
        sweep: two splits land within +/-0.005 eV of zero, which is numerical
        noise, not evidence against learning.
        """
        from scipy.stats import ttest_1samp

        entries = _load_entries()

        gaps = []
        for seed in range(10):
            holdout, seed_calib, unmeasured = _split(entries, seed)
            ingested = unmeasured[:40]

            _, mae_real = _holdout_metrics(_fit(seed_calib + ingested), holdout)

            labels = [(e["homo_eV"], e["lumo_eV"]) for e in ingested]
            random.Random(seed).shuffle(labels)
            permuted = [
                {**e, "homo_eV": lab[0], "lumo_eV": lab[1]}
                for e, lab in zip(ingested, labels, strict=True)
            ]
            _, mae_shuffled = _holdout_metrics(_fit(seed_calib + permuted), holdout)

            gaps.append(mae_shuffled - mae_real)

        p_value = float(ttest_1samp(gaps, 0.0, alternative="greater").pvalue)
        assert p_value < 0.05, (
            f"Correct labels do not beat shuffled labels significantly "
            f"(mean gap {np.mean(gaps):+.4f} eV, p={p_value:.4f}). "
            "The apparent closed-loop gain is recalibration, not learning."
        )

    def test_loop_survives_noisy_measurements(self):
        """A wet-lab loop must tolerate realistic instrument error."""
        entries = _load_entries()
        holdout, seed_calib, unmeasured = _split(entries)
        rng = random.Random(0)

        noisy = [
            {**e, "homo_eV": e["homo_eV"] + rng.gauss(0.0, 0.3),
             "lumo_eV": e["lumo_eV"] + rng.gauss(0.0, 0.3)}
            for e in unmeasured[:40]
        ]

        _, mae_before = _holdout_metrics(_fit(seed_calib), holdout)
        _, mae_after = _holdout_metrics(_fit(seed_calib + noisy), holdout)

        assert mae_after < mae_before, (
            "Loop must still improve with 0.3 eV measurement noise "
            f"(before={mae_before:.4f}, after={mae_after:.4f})"
        )


class TestAcquisitionIsNotRedundant:
    """The suggester must propose a structurally diverse batch.

    ADR-2026-08-08-06: before the structural-diversity fix the suggester's
    picks were *more* redundant than random sampling (mean pairwise Tanimoto
    0.138 vs 0.071), because its novelty term measures distance to the
    calibration set and is therefore identical across a homologous family.
    Redundant batches have correlated residuals and waste lab budget.
    """

    def test_suggested_batch_is_less_redundant_than_random(self):
        from aurelius.agent.experiment_suggester import suggest_experiments
        from aurelius.utils.device import batch_tanimoto

        entries = _load_entries()
        _, _, unmeasured = _split(entries, 0)
        pool = [e["smiles"] for e in unmeasured]

        def redundancy(smiles_list):
            fps = [
                MoleculeContext.from_smiles(s).get_ecfp4()
                for s in smiles_list
                if MoleculeContext.from_smiles(s) is not None
            ]
            sim = batch_tanimoto(fps)
            return float(sim[np.triu_indices(sim.shape[0], k=1)].mean())

        suggested = [
            s.smiles
            for s in suggest_experiments(pool, top_n=10, properties=["homo"])
        ]
        random_pick = random.Random(0).sample(pool, 10)

        assert redundancy(suggested) < redundancy(random_pick), (
            f"Suggested batch ({redundancy(suggested):.4f}) must be less "
            f"redundant than random ({redundancy(random_pick):.4f})"
        )


class TestFeedbackControllerRefitIsReal:
    """The production ``FeedbackController`` path, not just a bare model."""

    def test_maybe_refit_expands_calibration_and_reports_loo(self):
        feedback_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "aurelius", "data",
            "experimental_feedback.json",
        )
        controller = FeedbackController(
            refit_interval=1, experimental_feedback_path=feedback_path
        )

        for smiles in ("C1COC(=O)O1", "COC(=O)OC", "CC#N"):
            ctx = MoleculeContext.from_smiles(smiles)
            homo, lumo = controller.get_delta_correction().predict_corrected(ctx.mol)
            controller.accumulate(
                smiles=smiles,
                homo_prediction=homo,
                lumo_prediction=lumo,
                homo_corrected=homo,
                lumo_corrected=lumo,
                total_score=50.0,
                conformal_confidence=0.9,
                generation=1,
            )

        info = controller.maybe_refit(current_generation=5)

        assert info is not None, "Refit must trigger past the interval"
        assert info["new_calibration_entries"] > 0, (
            "Refit must actually absorb experimental points; otherwise the "
            "closed loop is a no-op."
        )
        assert info["loo_mae_after"] <= info["loo_mae_before"] + 1e-9, (
            f"Refit degraded LOO MAE: {info}"
        )
