"""Experimental feedback ingestion pipeline.

This module provides standardized tools for parsing experimental results
from CSV and SDF files, validating feedback schemas, and integrating
experimental data into the active-learning loop.

Usage:
    from aurelius.feedback.parser import (
        parse_experimental_csv,
        parse_experimental_sdf,
        validate_feedback_schema,
    )

    # Parse CSV feedback
    feedback = parse_experimental_csv("experimental_results.csv")

    # Parse SDF feedback
    feedback = parse_experimental_sdf("experimental_results.sdf")

    # Validate a feedback entry
    is_valid, errors = validate_feedback_schema(entry)
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)


class FeedbackEntry(Protocol):
    """Protocol for experimental feedback entries."""

    smiles: str
    dielectric: float
    viscosity: float
    cycle_life: float
    dielectric_constant: float
    viscosity_cP: float


def parse_experimental_csv(file_path: str) -> list[dict[str, Any]]:
    """Read a CSV file and return a list of experiment entries.

    Required columns: ``smiles``, ``dielectric``, ``viscosity``,
    ``cycle_life`` (case-insensitive).

    Args:
        file_path: Path to a CSV file.

    Returns:
        List of dicts with keys: smiles, dielectric, viscosity, cycle_life.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    with open(file_path) as fh:
        content = fh.read()

    reader = csv.DictReader(content.splitlines())
    fieldnames = [fn.lower().strip() for fn in reader.fieldnames or []]

    required = {"smiles", "dielectric", "viscosity", "cycle_life"}
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    results: list[dict[str, Any]] = []
    for row in reader:
        normalized_row = {k.strip().lower(): v.strip() for k, v in row.items()}
        entry: dict[str, Any] = {}
        for col in required:
            val = normalized_row.get(col, "")
            if not val:
                entry[col] = 0.0
            else:
                try:
                    entry[col] = float(val)
                except ValueError:
                    entry[col] = 0.0
        entry["smiles"] = normalized_row.get("smiles", "")
        if entry["smiles"]:
            results.append(entry)
    return results


def parse_experimental_sdf(file_path: str) -> list[dict[str, Any]]:
    """Read an SDF file and return a list of experiment entries.

    Required properties in each molecule block:
    - ``SMILES``: the SMILES string
    - ``dielectric_constant``: experimental dielectric constant
    - ``viscosity_cP``: experimental viscosity in cP
    - ``cycle_life``: experimental cycle life

    Args:
        file_path: Path to an SDF file.

    Returns:
        List of dicts with keys: smiles, dielectric_constant, viscosity_cP, cycle_life.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    from rdkit import Chem

    results: list[dict[str, Any]] = []
    with open(file_path, "rb") as fh:
        for mol_block in Chem.ForwardSDMolSupplier(fh):
            if mol_block is None:
                continue
            entry: dict[str, Any] = {}
            entry["smiles"] = mol_block.GetProp("SMILES") if mol_block.HasProp("SMILES") else ""
            entry["dielectric_constant"] = float(mol_block.GetProp("dielectric_constant")) if mol_block.HasProp("dielectric_constant") else 0.0
            entry["viscosity_cP"] = float(mol_block.GetProp("viscosity_cP")) if mol_block.HasProp("viscosity_cP") else 0.0
            entry["cycle_life"] = float(mol_block.GetProp("cycle_life")) if mol_block.HasProp("cycle_life") else 0.0
            if entry["smiles"]:
                results.append(entry)
    return results


def parse_feedback_file(file_path: str) -> list[dict[str, Any]]:
    """Auto-detect file type (CSV vs SDF) and parse accordingly.

    Args:
        file_path: Path to either a CSV or SDF file.

    Returns:
        List of experiment entries.
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".csv":
        return parse_experimental_csv(file_path)
    elif ext in {".sdf", ".mol", ".mol2"}:
        return parse_experimental_sdf(file_path)
    else:
        raise ValueError(f"Unsupported file extension '{ext}'. Use .csv or .sdf.")


def validate_feedback_schema(entry: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate that a feedback entry has required fields.

    Args:
        entry: A dict with keys like 'smiles', 'dielectric', 'viscosity', etc.

    Returns:
        Tuple of (is_valid, list_of_error_messages).
    """
    errors: list[str] = []
    required_keys = {"smiles", "dielectric", "viscosity", "cycle_life"}

    for key in required_keys:
        if key not in entry:
            errors.append(f"Missing required field: {key}")
        elif entry.get(key) is not None:
            if key == "smiles":
                if not isinstance(entry[key], str) or not entry[key].strip():
                    errors.append(f"Invalid SMILES value: {entry[key]!r}")
            elif isinstance(entry[key], (int, float)):
                if entry[key] < 0:
                    errors.append(f"Negative value for {key}: {entry[key]}")
            else:
                try:
                    float(entry[key])
                except (ValueError, TypeError):
                    errors.append(f"Non-numeric value for {key}: {entry[key]!r}")

    return len(errors) == 0, errors


def ingest_feedback(
    feedback_data: list[dict[str, Any]],
    pipeline: Any = None,
) -> dict[str, Any]:
    """Ingest experimental feedback and optionally trigger model retraining.

    This function:
    1. Validates all feedback entries using validate_feedback_schema.
    2. If a pipeline is provided, feeds data into the GcUqEnsemble.
    3. Returns summary statistics of the ingested data.

    Args:
        feedback_data: List of experiment entries (dicts).
        pipeline: Optional AureliusPipeline for retraining.

    Returns:
        Summary dict with n_valid, n_invalid, and mean values.
    """
    valid_entries: list[dict[str, Any]] = []
    invalid_entries: list[dict[str, Any]] = []

    for entry in feedback_data:
        is_valid, errors = validate_feedback_schema(entry)
        if is_valid:
            valid_entries.append(entry)
        else:
            invalid_entries.append({"entry": entry, "errors": errors})
            log.warning("Invalid feedback entry: %s", errors)

    dielectrics = [e.get("dielectric", 0.0) for e in valid_entries if e.get("dielectric")]
    viscosities = [e.get("viscosity", 0.0) for e in valid_entries if e.get("viscosity")]

    summary = {
        "n_valid": len(valid_entries),
        "n_invalid": len(invalid_entries),
        "mean_dielectric": sum(dielectrics) / len(dielectrics) if dielectrics else 0.0,
        "mean_viscosity": sum(viscosities) / len(viscosities) if viscosities else 0.0,
    }

    if pipeline is not None:
        gc_uq = getattr(pipeline, "_oracle", None)
        if gc_uq is not None:
            feedback_for_retrain = [
                {
                    "smiles": e.get("smiles", ""),
                    "dielectric_constant": e.get("dielectric", 0.0),
                    "viscosity_cP": e.get("viscosity", 0.0),
                }
                for e in valid_entries
            ]
            gc_uq.append_empirical_data(feedback_for_retrain)
            summary["retrained"] = True

    return summary
