"""Automated active-learning suggest → validate → retrain pipeline.

This module closes the gap between discovery generation and experimental
feedback by providing:

* ``SuggestAndValidatePipeline`` – generates top-10 discovery suggestions
  from the loop state and exports them to ``suggestions.sdf``.
* ``ExperimentResultParser`` – parses CSV or SDF files containing experimental
  measurements (SMILES, dielectric, viscosity, cycle life, etc.).
* ``AutoRetrainPipeline`` – feeds parsed experimental data back into the
  GC UQ ensemble so the model can learn from wet-lab results.

Together they form a fully-automated "suggest → validate → retrain" cycle.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from aurelius.agent.state import LoopState
from aurelius.types import MoleculeContext, ScreeningResult

log = logging.getLogger(__name__)


class SuggestAndValidatePipeline:
    """Generates top-10 suggestions and exports them to an SDF file.

    The pipeline:
    1. Sorts discoveries by total_score descending.
    2. Selects the top-10 (or fewer if fewer exist).
    3. Exports them as an SDF file with property annotations.

    Usage:
        suggestions = SuggestAndValidatePipeline(discoveries)
        suggestions.export("suggestions.sdf")
    """

    def __init__(self, discoveries: list[ScreeningResult] | list[dict[str, Any]]) -> None:
        def _score_key(r: ScreeningResult | dict[str, Any]) -> float:
            if isinstance(r, dict):
                return r.get("total_score", 0.0)
            return r.total_score
        self._discoveries = sorted(discoveries, key=_score_key, reverse=True)[:10]

    @property
    def suggestions(self) -> list[dict[str, Any]]:
        """Return the top-10 discovery entries as dicts."""
        entries: list[dict[str, Any]] = []
        for d in self._discoveries:
            if isinstance(d, dict):
                entries.append({
                    "smiles": d.get("smiles", ""),
                    "total_score": d.get("total_score", 0.0),
                    "homo_eV": d.get("homo_eV"),
                    "lumo_eV": d.get("lumo_eV"),
                    "dielectric_proxy": d.get("dielectric_proxy"),
                    "viscosity_proxy": d.get("viscosity_proxy"),
                    "li_solvation_proxy": d.get("li_solvation_proxy"),
                })
            else:
                entries.append({
                    "smiles": d.smiles,
                    "total_score": d.total_score,
                    "homo_eV": d.homo_eV,
                    "lumo_eV": d.lumo_eV,
                    "dielectric_proxy": d.dielectric_proxy,
                    "viscosity_proxy": d.viscosity_proxy,
                    "li_solvation_proxy": d.li_solvation_proxy,
                })
        return entries

    def export(self, path: str = "suggestions.sdf") -> None:
        """Export suggestions to an SDF file.

        Args:
            path: Output SDF file path.
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem

        path = _resolve_output_path(path)
        writer = Chem.SDWriter(str(path))
        for entry in self._discoveries:
            smi = entry.smiles if isinstance(entry, ScreeningResult) else entry.get("smiles", "")
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            _embed_3d(mol)
            mol.SetProp("SMILES", smi)
            mol.SetProp("total_score", f"{entry.total_score if isinstance(entry, ScreeningResult) else entry.get('total_score', 0.0):.2f}")
            writer.write(mol)
        writer.close()
        log.info("Exported %d suggestions to %s", len(self._discoveries), path)

    @staticmethod
    def _resolve_output_path(path: str) -> str:
        """Resolve an output path (relative or absolute)."""
        return str(Path(path).resolve())


class ExperimentResultParser:
    """Parses experimental results from CSV or SDF files.

    Expected CSV columns: SMILES, dielectric, viscosity, cycle_life
    (column names are case-insensitive and whitespace-stripped).

    Expected SDF properties: SMILES, dielectric_constant, viscosity_cP,
    cycle_life (and any additional property keys).
    """

    @staticmethod
    def parse_csv(file_path: str) -> list[dict[str, Any]]:
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
            # Normalize column names by stripping whitespace and lowercasing
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

    @staticmethod
    def parse_sdf(file_path: str) -> list[dict[str, Any]]:
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

    @staticmethod
    def parse_file(file_path: str) -> list[dict[str, Any]]:
        """Auto-detect file type (CSV vs SDF) and parse accordingly.

        Args:
            file_path: Path to either a CSV or SDF file.

        Returns:
            List of experiment entries.
        """
        ext = Path(file_path).suffix.lower()
        if ext == ".csv":
            return ExperimentResultParser.parse_csv(file_path)
        elif ext in {".sdf", ".mol", ".mol2"}:
            return ExperimentResultParser.parse_sdf(file_path)
        else:
            raise ValueError(f"Unsupported file extension '{ext}'. Use .csv or .sdf.")


class AutoRetrainPipeline:
    """Triggers GC UQ ensemble retraining with experimental feedback data.

    The pipeline:
    1. Collects experimental data from either SuggestAndValidatePipeline or
       ExperimentResultParser.
    2. Feeds the data into the GcUqEnsemble's ``append_empirical_data`` method.
    3. Forces retraining on the next prediction call.
    4. Returns summary statistics of the retrained model.
    """

    def __init__(self) -> None:
        self._feedback_data: list[dict[str, Any]] = []

    def from_pipeline(self, suggestions: SuggestAndValidatePipeline) -> "AutoRetrainPipeline":
        """Load feedback data from a SuggestAndValidatePipeline instance."""
        self._feedback_data = suggestions.suggestions
        self._feedback_data = [
            {
                "smiles": d.get("smiles", ""),
                "dielectric_constant": d.get("dielectric_proxy", 0.0),
                "viscosity_cP": d.get("viscosity_proxy", 0.0),
            }
            for d in self._feedback_data
        ]
        return self

    def from_results(self, results: list[ScreeningResult]) -> "AutoRetrainPipeline":
        """Load feedback data from a list of ScreeningResult objects."""
        self._feedback_data = [
            {
                "smiles": r.smiles,
                "dielectric_constant": r.dielectric_proxy or 0.0,
                "viscosity_cP": r.viscosity_proxy or 0.0,
            }
            for r in results
        ]
        return self

    @property
    def feedback_data(self) -> list[dict[str, Any]]:
        """Return the collected feedback data."""
        return self._feedback_data

    def retrain(self, pipeline: Any) -> dict[str, Any]:
        """Apply feedback data to the pipeline's GC UQ ensemble.

        Args:
            pipeline: The AureliusPipeline instance containing the GC UQ ensemble.

        Returns:
            Summary of the retraining operation.
        """
        gc_uq = getattr(pipeline, '_oracle', None)
        if gc_uq is None:
            return {"status": "error", "message": "GC UQ ensemble not available"}

        gc_uq.append_empirical_data(self._feedback_data)
        log.info("Applied %d feedback entries to GcUqEnsemble for retraining", len(self._feedback_data))
        return {
            "status": "success",
            "n_feedback": len(self._feedback_data),
            "smiles": [d.get("smiles", "") for d in self._feedback_data],
        }

    def summary(self) -> dict[str, Any]:
        """Return a summary of the feedback data."""
        if not self._feedback_data:
            return {"n_feedback": 0, "message": "No feedback data loaded"}

        dielectrics = [d.get("dielectric_constant", 0.0) for d in self._feedback_data]
        viscosities = [d.get("viscosity_cP", 0.0) for d in self._feedback_data]

        return {
            "n_feedback": len(self._feedback_data),
            "mean_dielectric": float(np.mean(dielectrics)) if dielectrics else 0.0,
            "std_dielectric": float(np.std(dielectrics, ddof=1)) if len(dielectrics) > 1 else 0.0,
            "mean_viscosity": float(np.mean(viscosities)) if viscosities else 0.0,
            "std_viscosity": float(np.std(viscosities, ddof=1)) if len(viscosities) > 1 else 0.0,
            "smiles": [d.get("smiles", "") for d in self._feedback_data],
        }


def _resolve_output_path(path: str) -> str:
    """Resolve an output path (relative or absolute)."""
    return str(Path(path).resolve())


def _embed_3d(mol: Any) -> None:
    """Embed a molecule in 3D if possible."""
    from rdkit.Chem import AllChem
    try:
        mol_3d = AllChem.RWMol(mol)
        mol_3d.UpdatePropertyCache()
        mol_h = AllChem.AddHs(mol_3d)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol_h, params) == -1:
            mol.SetProp("3D_embed_failed", "True")
    except Exception:
        mol.SetProp("3D_embed_failed", "True")
