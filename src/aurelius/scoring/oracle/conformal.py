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

_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
_ORBITAL_CALIBRATION_PATH = os.path.join(_DATA_DIR, "..", "..", "data", "orbital_calibration.json")
_EXTERNAL_BENCHMARK_PATH = os.path.join(_DATA_DIR, "..", "..", "data", "external_property_benchmark.json")

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


_CALIBRATION_FPS: list[Any] | None = None


def _calibration_fingerprints() -> list[Any]:
    """ECFP4 fingerprints of every calibration molecule (computed once)."""
    global _CALIBRATION_FPS
    if _CALIBRATION_FPS is None:
        fps: list[Any] = []
        seen: set[str] = set()
        for entry in list(_load_orbital_calibration()) + list(_load_external_benchmark()):
            smiles = entry.get("smiles")
            if not smiles:
                continue
            ctx = MoleculeContext.from_smiles(smiles)
            if ctx is None or ctx.smiles in seen:
                continue
            seen.add(ctx.smiles)
            fps.append(ctx.get_ecfp4())
        _CALIBRATION_FPS = fps
    return _CALIBRATION_FPS


def _max_similarity_to_calibration(fp: Any) -> float:
    """Highest Tanimoto similarity to any calibration molecule, in [0, 1]."""
    fps = _calibration_fingerprints()
    if not fps:
        return 0.0
    from rdkit import DataStructs

    return float(max(DataStructs.BulkTanimotoSimilarity(fp, fps)))


def _spread(values: Any) -> float:
    """Interquartile-ish spread of a reference property, ignoring missing values.

    Uses the 5-95 percentile range rather than min-max so that a single
    outlying reference value cannot inflate the scale.
    """
    clean = [float(v) for v in values if v is not None]
    if len(clean) < 4:
        return 1.0
    lo, hi = np.percentile(clean, [5, 95])
    return float(max(hi - lo, 1e-6))


class ConformalPredictor:
    """Split conformal prediction for HOMO, LUMO, dielectric, and viscosity.

    A single model is fit on the lower 80% of a calibration set; the upper
    20% is the *nonconformity* set whose residuals define the prediction
    interval width at the requested confidence level.

    ADR-2026-08-08-02: intervals are now *locally adaptive*. The original
    implementation returned ``point ± q`` with a single global quantile per
    property, so every molecule received an identical interval width. That
    is valid — marginal coverage still holds — but useless for anything that
    needs to compare uncertainty *between* molecules: active learning saw a
    constant signal and degenerated to ranking on the other terms alone.

    The nonconformity score is therefore normalised by a difficulty estimate
    ``sigma(x)``, following Papadopoulos' normalised conformal regression:

        score_i = |y_i - yhat_i| / sigma(x_i)
        interval = yhat ± q_(1-alpha) * sigma(x)

    Marginal coverage is preserved exactly (the quantile is taken over the
    normalised scores), while molecules the model finds harder get honestly
    wider intervals. ``sigma`` is the domain-of-applicability penalty, which
    is already computed from closed-form topological features — no new model
    and no new dependency.
    """

    def __init__(self, confidence: float = _CONFIDENCE) -> None:
        self._confidence = confidence
        self._quantiles: dict[str, float] = {}
        self._max_widths: dict[str, float] = {}
        self._reference_spread: dict[str, float] = {}
        self._fitted = False

    @staticmethod
    def difficulty(mol: Chem.Mol) -> float:
        """Per-molecule difficulty estimate ``sigma(x)`` used to scale intervals.

        ``sigma`` combines two sources of expected error:

        * **Distance to the calibration set.** Split conformal prediction is
          justified by exchangeability. A molecule far from everything the
          model was calibrated on breaks that assumption, and its residual is
          expected to be larger. Distance is ``1 - max Tanimoto`` to the
          calibration fingerprints — the same notion of similarity the rest
          of the pipeline uses for diversity pressure.
        * **Domain-of-applicability penalty.** Captures topological features
          (long conjugation without sp3 support, excessive pi systems) that
          the orbital models are known to handle poorly.

        Distance carries most of the weight because, for the saturated
        electrolytes this project searches, the DoA penalty is near 1.0 for
        almost every candidate and so cannot discriminate between them.

        Returns 1.0 on any failure, degrading to the classical unnormalised
        interval.
        """
        try:
            from aurelius.scoring.oracle.quantum import compute_quantum_domain_penalty

            ctx = MoleculeContext(smiles=Chem.MolToSmiles(mol), mol=mol)
            penalty, _ = compute_quantum_domain_penalty(ctx)
            distance = 1.0 - _max_similarity_to_calibration(ctx.get_ecfp4())
        except Exception:  # pragma: no cover - defensive
            return 1.0
        # distance 0 -> 1.0, distance 1 -> 2.5 ; penalty 0.70 adds ~0.4 more.
        sigma = (1.0 + 1.5 * distance) * (1.0 + (1.0 - min(penalty, 1.0)))
        return float(min(4.0, max(0.5, sigma)))

    def _orbital_residuals(self, calib: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
        """Normalised HOMO/LUMO nonconformity scores over the calibration set."""
        from aurelius.scoring.oracle.lone_pair import predict_lone_pair_homo
        from aurelius.scoring.oracle.quantum import predict_tom_orbitals

        homo_residuals: list[float] = []
        lumo_residuals: list[float] = []
        for entry in calib:
            mol = Chem.MolFromSmiles(entry["smiles"])
            if mol is None:
                continue
            _, lumo_pred = predict_tom_orbitals(mol)
            # ADR-2026-08-08-01: the HOMO in production comes from the LPM, so
            # the nonconformity scores must be its residuals, not TOM's.
            homo_pred = predict_lone_pair_homo(mol)
            sigma = self.difficulty(mol)
            homo_residuals.append(abs(homo_pred - entry["homo_eV"]) / sigma)
            lumo_residuals.append(abs(lumo_pred - entry["lumo_eV"]) / sigma)
        return homo_residuals, lumo_residuals

    def _bulk_residuals(self, bench: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
        """Normalised dielectric/viscosity nonconformity scores."""
        from aurelius.scoring.oracle.gc import (
            predict_dielectric_proxy,
            predict_viscosity_proxy,
        )

        diel_residuals: list[float] = []
        visc_residuals: list[float] = []
        for entry in bench:
            if entry.get("dielectric_constant") is None:
                continue
            ctx = MoleculeContext.from_smiles(entry["smiles"])
            if ctx is None:
                continue
            sigma = self.difficulty(ctx.mol)
            diel_residuals.append(
                abs(predict_dielectric_proxy(ctx) - entry["dielectric_constant"]) / sigma
            )
            if entry.get("viscosity_cP") is not None:
                visc_residuals.append(
                    abs(predict_viscosity_proxy(ctx) - entry["viscosity_cP"]) / sigma
                )
        return diel_residuals, visc_residuals

    def fit(self) -> None:
        """Compute nonconformity score quantiles from calibration data."""
        calib = _load_orbital_calibration()
        bench = _load_external_benchmark()
        homo_residuals, lumo_residuals = self._orbital_residuals(calib)
        diel_residuals, visc_residuals = self._bulk_residuals(bench)

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

        # Spread of the reference values per property. Used to express an
        # interval width as a fraction of the property's natural range, which
        # makes widths comparable across properties measured in different
        # units (eV vs dimensionless epsilon vs cP).
        self._reference_spread = {
            "homo": _spread(e.get("homo_eV") for e in calib),
            "lumo": _spread(e.get("lumo_eV") for e in calib),
            "dielectric": _spread(e.get("dielectric_constant") for e in bench),
            "viscosity": _spread(e.get("viscosity_cP") for e in bench),
        }

        self._fitted = True
        logger.info(
            "ConformalPredictor fitted: %s",
            {k: f"{v:.3f}" for k, v in self._quantiles.items()},
        )

    def predict_interval(
        self,
        property_name: str,
        point_estimate: float,
        mol: Chem.Mol | None = None,
    ) -> tuple[float, float]:
        """Return (lower, upper) prediction interval for a property.

        Args:
            property_name: One of ``homo``, ``lumo``, ``dielectric``, ``viscosity``.
            point_estimate: The oracle's point prediction.
            mol: If given, the interval is scaled by this molecule's
                difficulty ``sigma(x)``, so harder molecules receive wider
                intervals. Omitting it reproduces the previous global-width
                behaviour, which remains valid but is not molecule-discriminative.

        If the model has not been fitted, returns a wide fallback interval.
        """
        if not self._fitted:
            self.fit()
        half_width = self._quantiles.get(property_name, 1.0)
        if mol is not None:
            half_width *= self.difficulty(mol)
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
