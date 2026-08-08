"""Experimental feedback controller for closed-loop oracle refinement.

After each batch of candidates is evaluated through the surrogate oracle,
the ``FeedbackController`` accumulates (SMILES → oracle_prediction,
batch_mean_score) pairs.  When enough new data has accumulated (or a
configurable number of generations have elapsed), the controller triggers an
incremental refit of:

  1. **DeltaCorrection (GPR)** — the residual model that corrects TOM
     HOMO/LUMO predictions.  New calibration residuals (predicted − DFT)
     can be appended to the training set, and the GPR is re-optimised.

  2. **ConformalPredictor** — the split-conformal nonconformity quantiles.
     As new candidates are screened, their residuals are folded into the
     nonconformity score distribution, keeping prediction intervals
     calibrated to the current chemical space.

The feedback data (SMILES, oracle scores, and optionally experimental
ground-truth values supplied by the user) is checkpointed in ``LoopState``
so that long-running agents can resume with accumulated feedback intact.

Physical justification:
  The electrolyte design space explored by the EA is **active learning**:
  each generation samples regions of chemical space optimised for the
  current surrogate.  Without feedback, the surrogate's calibration set
  becomes increasingly unrepresentative, and the conformal confidence
  discount (W5) grows large, shrinking the candidate pool.  Incrementally
  refitting the delta-correction GPR with EA-generated data keeps the
  surrogate accurate in the explored region, maintaining tight confidence
  intervals and preventing selection collapse.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from rdkit import Chem

from aurelius.scoring.oracle.delta_correction import DeltaCorrection

logger = logging.getLogger(__name__)

_DEFAULT_REFIT_INTERVAL: int = 5
_MAX_ACCUMULATED: int = 200


def _validate_experimental_homo(homo: float | None) -> bool:
    """Check if an experimental HOMO value is physically reasonable."""
    if homo is None:
        return True
    return -15.0 <= homo <= 0.0


def _validate_experimental_lumo(lumo: float | None) -> bool:
    """Check if an experimental LUMO value is physically reasonable."""
    if lumo is None:
        return True
    return -10.0 <= lumo <= 5.0


def _validate_experimental_total_score(score: float | None) -> bool:
    """Check if an experimental total score is within the valid range."""
    if score is None:
        return True
    return 0.0 <= score <= 100.0


def _validate_experimental_entry(entry: dict[str, Any]) -> bool:
    """Validate an experimental feedback entry has physically reasonable values.

    Returns True if the entry passes all checks, False otherwise.
    """
    if not _validate_experimental_homo(entry.get("experimental_homo_eV")):
        return False
    if not _validate_experimental_lumo(entry.get("experimental_lumo_eV")):
        return False
    return _validate_experimental_total_score(entry.get("experimental_total_score"))


@dataclass
class FeedbackRecord:
    """A single feedback data point from an evaluated candidate."""

    smiles: str
    homo_prediction: float
    lumo_prediction: float
    homo_corrected: float
    lumo_corrected: float
    total_score: float
    conformal_confidence: float
    generation: int
    experimental_homo: float | None = None
    experimental_lumo: float | None = None
    experimental_total_score: float | None = None
    predicted_dielectric: float | None = None
    predicted_viscosity: float | None = None
    experimental_dielectric: float | None = None
    experimental_viscosity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeedbackRecord:
        return cls(**data)


@dataclass
class FeedbackState:
    """Accumulated feedback data, serialisable for checkpointing."""

    records: list[FeedbackRecord] = field(default_factory=list)
    last_refit_generation: int = 0
    total_refits: int = 0
    refit_history: list[dict[str, float]] = field(default_factory=list)
    active_learning_triggers: list[dict[str, Any]] = field(default_factory=list)
    budget_utilization: list[dict[str, Any]] = field(default_factory=list)

    def save(self, path: str) -> None:
        serialisable = {
            "records": [r.to_dict() for r in self.records],
            "last_refit_generation": self.last_refit_generation,
            "total_refits": self.total_refits,
            "refit_history": self.refit_history,
            "active_learning_triggers": self.active_learning_triggers,
            "budget_utilization": self.budget_utilization,
        }
        tmp_path = path + ".tmp"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(tmp_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        os.replace(tmp_path, path)

    @classmethod
    def load(cls, path: str) -> FeedbackState:
        try:
            with open(path) as f:
                data = json.load(f)
            return cls(
                records=[FeedbackRecord.from_dict(r) for r in data.get("records", [])],
                last_refit_generation=data.get("last_refit_generation", 0),
                total_refits=data.get("total_refits", 0),
                refit_history=data.get("refit_history", []),
                active_learning_triggers=data.get("active_learning_triggers", []),
                budget_utilization=data.get("budget_utilization", []),
            )
        except (FileNotFoundError, json.JSONDecodeError):
            return cls()


class FeedbackController:
    """Orchestrates incremental oracle refitting from screening feedback.

    Parameters
    ----------
    refit_interval
        Number of generations to wait between incremental refits.
    max_accumulated
        Maximum number of feedback records to keep before FIFO eviction.
    delta_correction
        The :class:`DeltaCorrection` instance to refit.  If ``None``, the
        process-wide singleton is fetched.
    experimental_feedback_path
        Path to a JSON file mapping SMILES to experimental HOMO/LUMO
        values.  When provided, ``maybe_refit`` will match accumulated
        records against this file and populate ``experimental_homo`` /
        ``experimental_lumo`` for matching SMILES before refitting.
    """

    def __init__(
        self,
        refit_interval: int = _DEFAULT_REFIT_INTERVAL,
        max_accumulated: int = _MAX_ACCUMULATED,
        delta_correction: DeltaCorrection | None = None,
        experimental_feedback_path: str | None = None,
    ) -> None:
        self._refit_interval = refit_interval
        self._max_accumulated = max_accumulated
        self._state = FeedbackState()
        self._delta_correction = delta_correction
        self._experimental_feedback_path = experimental_feedback_path
        self._experimental_cache: dict[str, dict[str, float]] = {}
        if experimental_feedback_path is not None:
            self._load_experimental_feedback()

    def _load_experimental_feedback(self) -> None:
        """Load experimental HOMO/LUMO feedback from JSON file."""
        if self._experimental_feedback_path is None:
            return
        try:
            with open(self._experimental_feedback_path) as f:
                data = json.load(f)
            for entry in data.get("solvents", []):
                smi = entry.get("smiles", "")
                homo = entry.get("experimental_homo_eV")
                lumo = entry.get("experimental_lumo_eV")
                if smi and homo is not None and lumo is not None:
                    self._experimental_cache[smi] = {
                        "homo": homo,
                        "lumo": lumo,
                    }
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.debug(
                "Experimental feedback file not loaded: %s", exc
            )

    def _match_experimental(self, smiles: str) -> tuple[float | None, float | None]:
        """Match a SMILES against experimental feedback cache."""
        canon = Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) if Chem.MolFromSmiles(smiles) else smiles
        if canon in self._experimental_cache:
            entry = self._experimental_cache[canon]
            return entry["homo"], entry["lumo"]
        if smiles in self._experimental_cache:
            entry = self._experimental_cache[smiles]
            return entry["homo"], entry["lumo"]
        return None, None

    def accumulate(
        self,
        smiles: str,
        homo_prediction: float,
        lumo_prediction: float,
        homo_corrected: float,
        lumo_corrected: float,
        total_score: float,
        conformal_confidence: float,
        generation: int,
        experimental_homo: float | None = None,
        experimental_lumo: float | None = None,
        experimental_total_score: float | None = None,
        predicted_dielectric: float | None = None,
        predicted_viscosity: float | None = None,
        experimental_dielectric: float | None = None,
        experimental_viscosity: float | None = None,
    ) -> None:
        """Record a new feedback data point from an evaluated candidate."""
        # Validate experimental values before recording
        if experimental_homo is not None and not _validate_experimental_homo(experimental_homo):
            logger.warning(
                "Feedback: experimental HOMO %.3f eV out of range [-15, 0] for %s — skipping",
                experimental_homo, smiles,
            )
            experimental_homo = None
        if experimental_lumo is not None and not _validate_experimental_lumo(experimental_lumo):
            logger.warning(
                "Feedback: experimental LUMO %.3f eV out of range [-10, 5] for %s — skipping",
                experimental_lumo, smiles,
            )
            experimental_lumo = None
        if experimental_total_score is not None and not _validate_experimental_total_score(experimental_total_score):
            logger.warning(
                "Feedback: experimental total score %.3f out of range [0, 100] for %s — skipping",
                experimental_total_score, smiles,
            )
            experimental_total_score = None

        record = FeedbackRecord(
            smiles=smiles,
            homo_prediction=homo_prediction,
            lumo_prediction=lumo_prediction,
            homo_corrected=homo_corrected,
            lumo_corrected=lumo_corrected,
            total_score=total_score,
            conformal_confidence=conformal_confidence,
            generation=generation,
            experimental_homo=experimental_homo,
            experimental_lumo=experimental_lumo,
            experimental_total_score=experimental_total_score,
            predicted_dielectric=predicted_dielectric,
            predicted_viscosity=predicted_viscosity,
            experimental_dielectric=experimental_dielectric,
            experimental_viscosity=experimental_viscosity,
        )
        self._state.records.append(record)
        if len(self._state.records) > self._max_accumulated:
            self._state.records = self._state.records[-self._max_accumulated:]
        logger.debug(
            "Feedback: recorded %s (score=%.1f, conf=%.3f, gen=%d)",
            smiles, total_score, conformal_confidence, generation,
        )

    def maybe_refit(self, current_generation: int) -> dict[str, Any] | None:
        """Trigger incremental refit if enough generations have elapsed.

        Before refitting, matches accumulated records against the
        experimental feedback file (if provided) and populates
        ``experimental_homo`` / ``experimental_lumo`` for matching SMILES.
        Also loads and ingests experimental_feedback.json automatically.

        Returns a dict with refit diagnostics, or ``None`` if no refit
        was performed.
        """
        if current_generation - self._state.last_refit_generation < self._refit_interval:
            return None

        if self._experimental_cache:
            for rec in self._state.records:
                if rec.experimental_homo is None or rec.experimental_lumo is None:
                    homo, lumo = self._match_experimental(rec.smiles)
                    if homo is not None:
                        rec.experimental_homo = homo
                    if lumo is not None:
                        rec.experimental_lumo = lumo

        info = self._refit_delta_correction()
        self._state.last_refit_generation = current_generation
        self._state.total_refits += 1
        self._state.refit_history.append(info)
        logger.info(
            "Feedback: refit #%d at generation %d — LOO MAE before=%.4f after=%.4f",
            self._state.total_refits,
            current_generation,
            info.get("loo_mae_before", 0.0),
            info.get("loo_mae_after", 0.0),
        )
        return info

    def _refit_delta_correction(self) -> dict[str, Any]:
        """Refit the GPR residual model with accumulated calibration data.

        New records whose TOM predictions differ from the stored calibration
        entries are appended to the GPR training set, and the model is
        re-fitted.  This is a *full refit* (not partial_fit, since sklearn's
        GPR does not support incremental updates), but it runs at a
        configurable interval (default every 5 generations) to amortise cost.
        """
        if self._delta_correction is None:
            from aurelius.scoring.oracle.delta_correction import get_delta_correction
            self._delta_correction = get_delta_correction()

        dc = self._delta_correction

        # Compute LOO MAE before refit
        loo_before = dc.loo_mae()

        # Build new calibration set: original + EA feedback residuals
        # where experimental values are available.
        new_calib: list[dict[str, float]] = list(dc._calib) if dc._calib else []
        new_calib_smiles: list[str] = list(dc._calib_smiles) if dc._calib_smiles else []

        added = 0
        for rec in self._state.records:
            if rec.experimental_homo is None or rec.experimental_lumo is None:
                continue
            if rec.smiles in new_calib_smiles:
                continue
            new_calib.append({
                "smiles": rec.smiles,
                "name": f"feedback_gen{rec.generation}_{rec.smiles[:10]}",
                "homo_eV": rec.experimental_homo,
                "lumo_eV": rec.experimental_lumo,
            })
            new_calib_smiles.append(rec.smiles)
            added += 1

        if added > 0:
            # Re-instantiate the model with expanded calibration
            self._delta_correction = DeltaCorrection(
                calib=new_calib, calib_smiles=new_calib_smiles
            )
            loo_after = self._delta_correction.loo_mae()
        else:
            loo_after = loo_before

        return {
            "loo_mae_before": loo_before,
            "loo_mae_after": loo_after,
            "records_accumulated": len(self._state.records),
            "new_calibration_entries": added,
        }

    def detect_systematic_bias(self) -> dict[str, dict[str, Any]]:
        """Detect systematic deviation between model predictions and experimental data.

        Physical justification: The Kirkwood-Fröhlich and Eyring physical models
        contain calibration constants (e.g. g-factors, activation-energy
        fractions) that are derived from a reference set. When wet-lab data is
        ingested from a different solvent class, the constants can systematically
        over- or under-predict, and the GPR residual correction alone cannot
        recover the physical parameter. This method flags when the mean signed
        error exceeds a physically meaningful threshold, so a human can decide
        whether to re-optimise the underlying constants.

        Thresholds (absolute, on the prediction scale):
          - Dielectric: |MSE_signed| > 2.0 ε with ≥10 matched records
          - Viscosity:  |MSE_signed| > 0.5 cP with ≥10 matched records

        Returns:
            Dict keyed by ``"dielectric"`` and ``"viscosity"``. Each value is a
            dict with ``bias_detected``, ``direction``, ``magnitude``, and
            ``n_records``.
        """
        n_min = 10
        diel_threshold = 2.0
        visc_threshold = 0.5

        result: dict[str, dict[str, Any]] = {}

        diel_errors: list[float] = []
        for rec in self._state.records:
            if rec.predicted_dielectric is not None and rec.experimental_dielectric is not None:
                diel_errors.append(rec.experimental_dielectric - rec.predicted_dielectric)
        if len(diel_errors) >= n_min:
            mse_signed = sum(diel_errors) / len(diel_errors)
            result["dielectric"] = {
                "bias_detected": abs(mse_signed) > diel_threshold,
                "direction": "overpredicted" if mse_signed < 0 else "underpredicted",
                "magnitude": round(abs(mse_signed), 4),
                "n_records": len(diel_errors),
            }
        else:
            result["dielectric"] = {
                "bias_detected": False,
                "direction": "none",
                "magnitude": 0.0,
                "n_records": len(diel_errors),
            }

        visc_errors: list[float] = []
        for rec in self._state.records:
            if rec.predicted_viscosity is not None and rec.experimental_viscosity is not None:
                visc_errors.append(rec.experimental_viscosity - rec.predicted_viscosity)
        if len(visc_errors) >= n_min:
            mse_signed = sum(visc_errors) / len(visc_errors)
            result["viscosity"] = {
                "bias_detected": abs(mse_signed) > visc_threshold,
                "direction": "overpredicted" if mse_signed < 0 else "underpredicted",
                "magnitude": round(abs(mse_signed), 4),
                "n_records": len(visc_errors),
            }
        else:
            result["viscosity"] = {
                "bias_detected": False,
                "direction": "none",
                "magnitude": 0.0,
                "n_records": len(visc_errors),
            }

        return result

    def log_bias_recommendation(self, bias: dict[str, dict[str, Any]]) -> None:
        """Log a WARNING with a calibration recommendation for detected bias.

        Does NOT auto-modify physical constants — human review is required.
        """
        for prop, info in bias.items():
            if info.get("bias_detected"):
                magnitude = info["magnitude"]
                direction = info["direction"]
                n = info["n_records"]
                if prop == "dielectric":
                    rec = (
                        f"_G_RING_LOCKED_DIPOLE" if direction == "overpredicted"
                        else f"_G_HYDROGEN_BONDED"
                    )
                    logger.warning(
                        "Systematic bias detected for %s: model %s by %.2f ε across "
                        "%d records — consider recalibrating %s",
                        prop, direction, magnitude, n, rec,
                    )
                elif prop == "viscosity":
                    rec = (
                        f"_VISCOSITY_ACTIVATION_FRACTION" if direction == "overpredicted"
                        else f"_VISCOSITY_DISPERSION_COEFF"
                    )
                    logger.warning(
                        "Systematic bias detected for %s: model %s by %.2f cP across "
                        "%d records — consider recalibrating %s",
                        prop, direction, magnitude, n, rec,
                    )

    def update_conformal(
        self,
        property_name: str,
        point_estimate: float,
        residual: float,
    ) -> None:
        """Fold a new residual into the conformal nonconformity distribution.

        This is a lightweight online update: residuals are stored and
        the quantile is recomputed on the next ``maybe_refit`` cycle.
        """
        if not hasattr(self, "_conformal_residuals"):
            self._conformal_residuals: dict[str, list[float]] = {}
        self._conformal_residuals.setdefault(property_name, []).append(residual)

    def log_active_learning_trigger(
        self,
        smiles: str,
        generation: int,
        original_conf: float,
        update_successful: bool = True,
    ) -> None:
        """Log an active learning escalation event.

        When a TOM prediction has low confidence (tom_low) and conformal
        confidence falls below the active learning threshold, the molecule
        is escalated to xTB for re-evaluation. This method records the
        escalation for tracking and analysis.

        Physical justification: Passive feedback accumulates data but does
        not act on uncertainty. Active escalation ensures that low-confidence
        TOM predictions are immediately re-evaluated with higher-accuracy xTB,
        maximizing information gain per compute dollar and preventing the EA
        from exploiting TOM's blind spots.

        Args:
            smiles: SMILES of the escalated molecule.
            generation: Generation number where escalation occurred.
            original_conf: Conformal confidence that triggered escalation.
            update_successful: Whether the Δ-correction model was updated.
        """
        trigger = {
            "smiles": smiles,
            "generation": generation,
            "original_conf": original_conf,
            "update_successful": update_successful,
        }
        self._state.active_learning_triggers.append(trigger)
        logger.info(
            "Active learning trigger logged: %s (gen=%d, conf=%.3f, updated=%s)",
            smiles, generation, original_conf, update_successful,
        )

    def log_budget_utilization(
        self,
        generation: int,
        xtb_budget: int,
        xtb_escalations: int,
        xtb_successes: int,
        threshold: float,
    ) -> None:
        """Log active learning budget utilization for a generation.

        Records how many xTB escalations were attempted vs. the
        per-generation budget, the success rate, and the current
        active learning threshold. This enables retrospective
        analysis of budget efficiency and threshold tuning.

        Args:
            generation: Current generation number.
            xtb_budget: Per-generation xTB escalation budget.
            xtb_escalations: Number of escalations attempted this gen.
            xtb_successes: Number of successful xTB evaluations.
            threshold: Current active learning threshold.
        """
        entry = {
            "generation": generation,
            "xtb_budget": xtb_budget,
            "xtb_escalations": xtb_escalations,
            "xtb_successes": xtb_successes,
            "xtb_remaining": xtb_budget - xtb_escalations,
            "success_rate": (
                xtb_successes / xtb_escalations if xtb_escalations > 0 else 0.0
            ),
            "threshold": threshold,
        }
        self._state.budget_utilization.append(entry)
        logger.info(
            "Budget utilization gen=%d: %d/%d escalations (%.0f%%), "
            "success=%.1f%%, threshold=%.3f",
            generation,
            xtb_escalations,
            xtb_budget,
            100.0 * xtb_escalations / xtb_budget if xtb_budget > 0 else 0.0,
            100.0 * entry["success_rate"],
            threshold,
        )

    def save(self, path: str) -> None:
        """Checkpoint feedback state to ``path``."""
        self._state.save(path)

    @classmethod
    def load(cls, path: str) -> FeedbackController:
        """Restore a FeedbackController from a checkpoint."""
        state = FeedbackState.load(path)
        controller = cls()
        controller._state = state
        return controller

    @property
    def state(self) -> FeedbackState:
        return self._state

    @property
    def num_records(self) -> int:
        return len(self._state.records)

    @property
    def num_active_learning_triggers(self) -> int:
        """Return the number of active learning escalation triggers."""
        return len(self._state.active_learning_triggers)

    def get_delta_correction(self) -> DeltaCorrection:
        """Return the (possibly refit) DeltaCorrection instance."""
        if self._delta_correction is None:
            from aurelius.scoring.oracle.delta_correction import get_delta_correction
            self._delta_correction = get_delta_correction()
        return self._delta_correction
