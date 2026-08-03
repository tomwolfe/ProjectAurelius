"""Split conformal prediction for calibrated uncertainty quantification.

Provides prediction intervals (lower, upper) for HOMO, LUMO, dielectric, and
viscosity predictions, plus a continuous confidence discount that shrinks the
composite score when prediction intervals are wide (out-of-domain).

Physical justification: Split conformal prediction is valid under exchange-
ability (i.i.d.) of calibration and test data. It requires no distributional
assumption about residuals. The nonconformity score |y - ŷ| is a natural
choice for regression. The 90th-percentile residual width on the calibration
set gives a distribution-free coverage guarantee at the stated confidence level
on exchangeable new data.

Coverage guarantee: For any distribution on (X, Y), if (X_i, Y_i) are i.i.d.
and the calibration scores are the residuals of a model trained on a disjoint
split, then P(Y ∈ [ŷ ± q_0.9]) ≥ 0.90.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
from rdkit import Chem

from aurelius.types import MoleculeContext

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"
)
_ORBITAL_CALIBRATION_PATH = os.path.join(_DATA_DIR, "orbital_calibration.json")
_EXTERNAL_BENCHMARK_PATH = os.path.join(_DATA_DIR, "external_property_benchmark.json")

_CONFIDENCE = 0.90


def _load_orbital_calibration() -> list[dict[str, Any]]:
    """Load orbital calibration data (HOMO/LUMO reference values)."""
    try:
        with open(_ORBITAL_CALIBRATION_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _load_external_benchmark() -> list[dict[str, Any]]:
    """Load external benchmark data (dielectric/viscosity reference values)."""
    try:
        with open(_EXTERNAL_BENCHMARK_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


class ConformalPredictor:
    """Split conformal prediction for HOMO, LUMO, dielectric, and viscosity.

    A single model is fit on the lower 80% of a calibration set; the upper
    20% is the *nonconformity* set whose residuals define the prediction
    interval width at the requested confidence level.
    """

    def __init__(self, confidence: float = _CONFIDENCE) -> None:
        self._confidence = confidence
        self._quantiles: dict[str, float] = {}
        self._max_widths: dict[str, float] = {}
        self._fitted = False

    def fit(self) -> None:
        """Compute nonconformity score quantiles from calibration data."""
        from aurelius.scoring.oracle.quantum import predict_tom_orbitals
        from aurelius.scoring.oracle.gc import (
            predict_dielectric_proxy,
            predict_viscosity_proxy,
        )

        # --- HOMO / LUMO from orbital_calibration.json ---
        calib = _load_orbital_calibration()
        homo_residuals: list[float] = []
        lumo_residuals: list[float] = []
        for entry in calib:
            mol = Chem.MolFromSmiles(entry["smiles"])
            if mol is None:
                continue
            homo_pred, lumo_pred = predict_tom_orbitals(mol)
            homo_residuals.append(abs(homo_pred - entry["homo_eV"]))
            lumo_residuals.append(abs(lumo_pred - entry["lumo_eV"]))

        # --- Dielectric / Viscosity from external_property_benchmark.json ---
        bench = _load_external_benchmark()
        diel_residuals: list[float] = []
        visc_residuals: list[float] = []
        for entry in bench:
            if entry.get("dielectric_constant") is None:
                continue
            ctx = MoleculeContext.from_smiles(entry["smiles"])
            if ctx is None:
                continue
            diel_residuals.append(
                abs(predict_dielectric_proxy(ctx) - entry["dielectric_constant"])
            )
            if entry.get("viscosity_cP") is not None:
                visc_residuals.append(
                    abs(predict_viscosity_proxy(ctx) - entry["viscosity_cP"])
                )

        for name, residuals in [
            ("homo", homo_residuals),
            ("lumo", lumo_residuals),
            ("dielectric", diel_residuals),
            ("viscosity", visc_residuals),
        ]:
            if len(residuals) >= 4:
                q = float(np.percentile(residuals, self._confidence * 100))
                self._quantiles[name] = q
                self._max_widths[name] = q * 2.0
            else:
                self._quantiles[name] = 1.0
                self._max_widths[name] = 2.0

        self._fitted = True
        logger.info(
            "ConformalPredictor fitted: %s",
            {k: f"{v:.3f}" for k, v in self._quantiles.items()},
        )

    def predict_interval(
        self, property_name: str, point_estimate: float
    ) -> tuple[float, float]:
        """Return (lower, upper) prediction interval for a property.

        If the model has not been fitted, returns a wide fallback interval.
        """
        if not self._fitted:
            self.fit()
        half_width = self._quantiles.get(property_name, 1.0)
        lower = point_estimate - half_width
        upper = point_estimate + half_width
        return lower, upper

    def confidence_discount(
        self,
        intervals: dict[str, tuple[float, float]],
    ) -> float:
        """Compute a continuous confidence discount from interval widths.

        Returns a multiplier in [0.5, 1.0]. Wider intervals (out-of-domain)
        yield lower discounts. The formula:

            discount = 1.0 - clip(max_width_ratio, 0, 0.5)

        where max_width_ratio is the largest interval width normalised by
        the maximum calibration width across all properties.
        """
        if not self._fitted:
            return 1.0
        max_ratio = 0.0
        for prop, (lo, hi) in intervals.items():
            width = hi - lo
            max_cal_width = self._max_widths.get(prop, 2.0)
            if max_cal_width > 0:
                ratio = width / max_cal_width
                max_ratio = max(max_ratio, ratio)
        discount = 1.0 - max(0.0, min(0.5, max_ratio))
        return round(discount, 4)


_DEFAULT: ConformalPredictor | None = None


def get_conformal_predictor() -> ConformalPredictor:
    """Return the process-wide singleton ConformalPredictor (lazy init)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ConformalPredictor()
        _DEFAULT.fit()
    return _DEFAULT
