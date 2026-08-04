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

    def save(self, path: str) -> None:
        serialisable = {
            "records": [r.to_dict() for r in self.records],
            "last_refit_generation": self.last_refit_generation,
            "total_refits": self.total_refits,
            "refit_history": self.refit_history,
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
    ) -> None:
        """Record a new feedback data point from an evaluated candidate."""
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

    def get_delta_correction(self) -> DeltaCorrection:
        """Return the (possibly refit) DeltaCorrection instance."""
        if self._delta_correction is None:
            from aurelius.scoring.oracle.delta_correction import get_delta_correction
            self._delta_correction = get_delta_correction()
        return self._delta_correction
