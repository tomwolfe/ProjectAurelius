"""Ingestion of wet-lab measurements into the feedback loop.

ADR-2026-08-07-11: ``FeedbackController`` could already accumulate
experimental values and refit the Δ-correction GPR, but nothing defined what
a wet-lab result file should look like, so every hand-off needed bespoke
glue. This module supplies the missing piece: one schema, one validator, and
one entry point that lands validated records in the feedback controller.

Design decisions worth stating, because each rejects a more permissive
alternative:

  * **Units are checked, not converted.** A record whose ``units`` do not
    match the canonical units of its ``measured_property`` is rejected. Silent
    unit conversion is how a Pa·s viscosity becomes a 1000x error in a
    calibration set, and calibration errors are invisible downstream — they
    just make the model quietly worse.
  * **Temperature is mandatory.** Dielectric constant and viscosity both vary
    strongly with temperature and the oracle models are referenced to
    298.15 K. A measurement without a temperature cannot be compared to a
    prediction. Records outside a tolerance of the reference temperature are
    accepted but flagged, since they are still useful provenance.
  * **Unparseable SMILES are rejected, not skipped silently.** They are
    returned in the report so the submitter can fix them.
  * **Ingestion never overwrites the calibration data files.** Records go to
    the feedback state, which is what ``maybe_refit`` consumes. Editing the
    verified benchmark from user input would destroy the property that makes
    it a trustworthy reference.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from rdkit import Chem

logger = logging.getLogger(__name__)

SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data",
    "experimental_results_schema.json",
)

_REQUIRED_FIELDS: tuple[str, ...] = (
    "smiles", "measured_property", "value", "units", "temperature_K", "method",
)

# Canonical units per property. A record must declare exactly one of these
# (case-insensitive, whitespace-stripped) or be rejected.
_CANONICAL_UNITS: dict[str, frozenset[str]] = {
    "dielectric_constant": frozenset({"", "1", "dimensionless", "none", "unitless"}),
    "viscosity_cP": frozenset({"cp", "mpa.s", "mpa*s", "mpas"}),
    "homo_eV": frozenset({"ev"}),
    "lumo_eV": frozenset({"ev"}),
    "donor_number": frozenset({"kcal/mol", "kcal mol-1"}),
    "ionic_conductivity_mS_cm": frozenset({"ms/cm", "ms cm-1", "mscm-1"}),
}

# Physically admissible ranges. A value outside these is far more likely to be
# a transcription or unit error than a real measurement.
_PLAUSIBLE_RANGE: dict[str, tuple[float, float]] = {
    "dielectric_constant": (1.0, 200.0),
    "viscosity_cP": (0.05, 10000.0),
    "homo_eV": (-15.0, 0.0),
    "lumo_eV": (-10.0, 5.0),
    "donor_number": (0.0, 60.0),
    "ionic_conductivity_mS_cm": (0.0, 100.0),
}

_REFERENCE_TEMPERATURE_K: float = 298.15
_TEMPERATURE_TOLERANCE_K: float = 5.0


def _predict_property_for_smiles(smiles: str, prop: str) -> float | None:
    """Get the model's predicted value for a property from GC oracle.

    Used by bias detection to compare model predictions against experimental
    measurements. Returns None if prediction fails.
    """
    try:
        from aurelius.scoring.oracle.gc import (
            predict_dielectric_proxy,
            predict_viscosity_proxy,
        )
        from aurelius.types import MoleculeContext

        ctx = MoleculeContext.from_smiles(smiles)
        if ctx is None:
            return None
        if prop == "dielectric":
            return predict_dielectric_proxy(ctx)
        if prop == "viscosity":
            return predict_viscosity_proxy(ctx)
    except Exception:
        return None
    return None


@dataclass
class IngestionReport:
    """Outcome of an ingestion run."""

    accepted: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[tuple[dict[str, Any], str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    refit: dict[str, Any] | None = None

    @property
    def n_accepted(self) -> int:
        return len(self.accepted)

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_accepted": self.n_accepted,
            "n_rejected": self.n_rejected,
            "accepted": self.accepted,
            "rejected": [{"record": r, "reason": why} for r, why in self.rejected],
            "warnings": self.warnings,
            "refit": self.refit,
        }


def _normalise_units(units: str) -> str:
    return str(units).strip().lower().replace(" ", "")


def validate_record(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Validate and canonicalise one measurement.

    Returns ``(canonicalised_record, "")`` on success or ``(None, reason)``
    on rejection.
    """
    # `units` is checked for presence only, not for emptiness: the canonical
    # unit of a dielectric constant is the empty string, since it is
    # dimensionless. Treating "" as absent would reject every valid
    # dielectric measurement.
    missing = [
        f for f in _REQUIRED_FIELDS
        if (record.get(f) is None) or (f != "units" and record.get(f) == "")
    ]
    if missing:
        return None, f"missing required field(s): {', '.join(missing)}"

    prop = str(record["measured_property"])
    if prop not in _CANONICAL_UNITS:
        return None, (
            f"unknown measured_property '{prop}'; expected one of "
            f"{', '.join(sorted(_CANONICAL_UNITS))}"
        )

    mol = Chem.MolFromSmiles(str(record["smiles"]))
    if mol is None:
        return None, f"unparseable SMILES: {record['smiles']!r}"

    try:
        value = float(record["value"])
        temperature = float(record["temperature_K"])
    except (TypeError, ValueError):
        return None, "value and temperature_K must be numeric"

    if temperature <= 0:
        return None, f"temperature_K must be positive, got {temperature}"

    if _normalise_units(record["units"]) not in _CANONICAL_UNITS[prop]:
        return None, (
            f"units {record['units']!r} are not the canonical units for {prop}; "
            f"expected one of {sorted(_CANONICAL_UNITS[prop])}. Convert the "
            f"value before ingesting — units are never converted automatically."
        )

    low, high = _PLAUSIBLE_RANGE[prop]
    if not low <= value <= high:
        return None, (
            f"{prop} = {value} lies outside the physically plausible range "
            f"[{low}, {high}]; check for a unit or transcription error"
        )

    canonical = dict(record)
    canonical["smiles"] = Chem.MolToSmiles(mol)
    canonical["value"] = value
    canonical["temperature_K"] = temperature
    canonical["measured_property"] = prop
    return canonical, ""


def load_records(path: str) -> list[dict[str, Any]]:
    """Read measurements from a JSON envelope or a CSV file.

    JSON must match ``experimental_results_schema.json``. CSV must have a
    header row whose column names match the measurement fields.
    """
    if path.lower().endswith(".csv"):
        with open(path, newline="") as f:
            return list(csv.DictReader(f))

    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "measurements" in data:
        return list(data["measurements"])
    raise ValueError(
        "JSON input must be a list of measurements or an object with a "
        "'measurements' array; see experimental_results_schema.json"
    )


def _partition_records(
    records: list[dict[str, Any]],
    report: IngestionReport,
    orbital: dict[str, dict[str, float]],
    bulk: dict[str, dict[str, float]],
) -> None:
    """Validate each record and route it to the orbital or bulk bucket.

    Records measured far from the model reference temperature are accepted but
    flagged: a viscosity at 60 °C is not comparable to a 25 °C prediction, and
    silently mixing them would masquerade as model error.
    """
    for raw in records:
        canonical, reason = validate_record(raw)
        if canonical is None:
            report.rejected.append((raw, reason))
            continue

        delta_t = abs(canonical["temperature_K"] - _REFERENCE_TEMPERATURE_K)
        if delta_t > _TEMPERATURE_TOLERANCE_K:
            report.warnings.append(
                f"{canonical['smiles']}: {canonical['measured_property']} measured at "
                f"{canonical['temperature_K']:.1f} K, {delta_t:.1f} K from the "
                f"{_REFERENCE_TEMPERATURE_K:.2f} K model reference; comparison to "
                f"predictions will carry a temperature bias"
            )

        prop = canonical["measured_property"]
        if prop in orbital:
            orbital[prop][canonical["smiles"]] = canonical["value"]
        elif prop in bulk:
            bulk[prop][canonical["smiles"]] = canonical["value"]
        report.accepted.append(canonical)


# Bulk property key -> the predictor name understood by
# ``_predict_property_for_smiles`` and the controller kwarg pair it feeds.
_BULK_PREDICTORS: tuple[tuple[str, str, str, str], ...] = (
    ("dielectric_constant", "dielectric", "predicted_dielectric", "experimental_dielectric"),
    ("viscosity_cP", "viscosity", "predicted_viscosity", "experimental_viscosity"),
)


def _accumulate_bulk(
    controller: Any,
    bulk: dict[str, dict[str, float]],
    generation: int,
) -> int:
    """Pair bulk measurements with model predictions for bias detection.

    Physical justification: The Kirkwood-Fröhlich and Eyring models make
    scale-dependent predictions. Comparing experimental measurements to the
    closed-form predictions reveals whether the calibration constants
    systematically over- or under-predict for the ingested solvent class.

    Returns the number of records accumulated.
    """
    n_new = 0
    for key, predictor, pred_kwarg, exp_kwarg in _BULK_PREDICTORS:
        for smiles, exp_val in bulk[key].items():
            pred = _predict_property_for_smiles(smiles, predictor)
            if pred is None:
                continue
            controller.accumulate(
                smiles=smiles,
                homo_prediction=0.0,
                lumo_prediction=0.0,
                homo_corrected=0.0,
                lumo_corrected=0.0,
                total_score=0.0,
                conformal_confidence=1.0,
                generation=generation,
                **{pred_kwarg: pred, exp_kwarg: exp_val},
            )
            n_new += 1
    return n_new


def ingest_experimental_results(
    path: str,
    controller: Any | None = None,
    generation: int = 0,
    trigger_refit: bool = True,
) -> IngestionReport:
    """Validate a results file and push the accepted records into feedback.

    Args:
        path: JSON or CSV file of measurements.
        controller: A ``FeedbackController``. Created with defaults if omitted.
        generation: Generation number to attribute the records to.
        trigger_refit: Whether to call ``maybe_refit`` after accumulating.

    Returns:
        An :class:`IngestionReport` recording what was accepted, what was
        rejected and why, and any refit diagnostics.
    """
    from aurelius.agent.feedback import FeedbackController

    report = IngestionReport()
    records = load_records(path)

    if controller is None:
        controller = FeedbackController()

    # Orbital energies are the only properties the Delta-correction GPR
    # consumes. Bulk properties are validated and recorded for provenance but
    # cannot currently drive a refit, and saying so is better than implying
    # they do.
    orbital = {"homo_eV": {}, "lumo_eV": {}}
    # Bulk properties tracked for systematic-bias detection (ADR-2026-08-07-09)
    bulk: dict[str, dict[str, float]] = {
        "dielectric_constant": {},
        "viscosity_cP": {},
    }

    _partition_records(records, report, orbital, bulk)

    paired = set(orbital["homo_eV"]) & set(orbital["lumo_eV"])
    for smiles in sorted(paired):
        controller.accumulate(
            smiles=smiles,
            homo_prediction=0.0,
            lumo_prediction=0.0,
            homo_corrected=0.0,
            lumo_corrected=0.0,
            total_score=0.0,
            conformal_confidence=1.0,
            generation=generation,
            experimental_homo=orbital["homo_eV"][smiles],
            experimental_lumo=orbital["lumo_eV"][smiles],
        )

    n_new_bulk = _accumulate_bulk(controller, bulk, generation)

    unpaired = (set(orbital["homo_eV"]) ^ set(orbital["lumo_eV"]))
    if unpaired:
        report.warnings.append(
            f"{len(unpaired)} molecule(s) have only one of HOMO/LUMO measured; "
            f"the Delta-correction refit needs both, so these were recorded but "
            f"not used for refitting: {', '.join(sorted(unpaired)[:5])}"
        )

    bulk_only = report.n_accepted - sum(len(v) for v in orbital.values())
    if bulk_only > 0:
        report.warnings.append(
            f"{bulk_only} bulk-property measurement(s) (dielectric, viscosity, "
            f"donor number, conductivity) were validated and recorded, but the "
            f"refit path currently consumes orbital energies only, so they do "
            f"not yet change any model"
        )

    # Systematic bias detection (ADR-2026-08-07-09)
    total_new_records = len(paired) + n_new_bulk
    if total_new_records >= 10:
        bias = controller.detect_systematic_bias()
        controller.log_bias_recommendation(bias)
        report.warnings.extend(
            f"Bias detected for {prop}: {info['direction']} by {info['magnitude']} "
            f"(n={info['n_records']})"
            for prop, info in bias.items()
            if info.get("bias_detected")
        )

    if trigger_refit and paired:
        report.refit = controller.maybe_refit(current_generation=generation)

    logger.info(
        "Experimental ingestion: %d accepted, %d rejected, %d refit-eligible pairs",
        report.n_accepted, report.n_rejected, len(paired),
    )
    return report
