"""Tier 0: Activation energy predictor wrapper.

Provides both GNN-based and linear-fallback prediction interfaces
for molecule-specific activation energies.
"""

from __future__ import annotations
from typing import Any

import os

import numpy as np

from aurelius.screening.tier0.data import (
    _build_molecular_graph,
)
from aurelius.screening.tier0.models import Tier0MPNN

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore[assignment, unused-ignore]


class Tier0ActivationPredictor:
    """Wrapper that supports both the old linear model and the new MPNN.

    When MPNN weights are available, uses the GNN for predictions.
    Falls back to the original linear predictor (literature defaults)
    if the GNN model is not loaded.

    Maintains backward compatibility with the original
    Tier0ActivationPredictor interface.
    """

    def __init__(self, model_path: str | None = None) -> None:
        """Initialize the predictor.

        Args:
            model_path: Optional path to MPNN weights. If provided and
                the file exists, loads the GNN model. Otherwise falls
                back to the linear predictor.
        """
        self._use_gnn = False
        self._gnn_model: Tier0MPNN | None = None

        if model_path and os.path.isfile(model_path):
            if HAS_TORCH:
                try:
                    self._gnn_model = Tier0MPNN(node_dim=4, edge_dim=0, hidden_dim=64, output_dim=4)
                    self._gnn_model.load_weights(model_path)
                    self._gnn_model.eval()
                    self._use_gnn = True
                except Exception as e:
                    print(f"[Tier0] Failed to load MPNN model from {model_path}: {e}. "
                          "Falling back to linear predictor.")
                    self._gnn_model = None
                    self._use_gnn = False
        elif model_path is None or not os.path.isfile(model_path):
            if model_path is not None:
                print(f"[Tier0] WARNING: Model path '{model_path}' is invalid or file not found. "
                      "Loading default linear predictor. For better accuracy, train via "
                      "`aurelius train --task tier0`.")
            else:
                print("[Tier0] WARNING: No model path provided. "
                      "Loading default linear predictor. For better accuracy, train via "
                      "`aurelius train --task tier0`.")

        self._linear_predictor = _LinearFallbackPredictor()

    def predict(
        self,
        descriptors: dict[str, float] | None = None,
        smiles: str | None = None,
    ) -> dict[str, float]:
        """Predict molecule-specific activation energies.

        Uses MPNN if available and SMILES is provided, otherwise
        falls back to the linear predictor.

        Args:
            descriptors: Optional molecular descriptors dict.
            smiles: Optional SMILES string.

        Returns:
            Dictionary with predicted activation energies:
                - ec_reduction: EC solvent reduction Ea (eV)
                - dm_reduction: DMC solvent reduction Ea (eV)
                - pf6_decomposition: Salt decomposition Ea (eV)
                - polymerization: Polymerization Ea (eV)
        """
        if self._use_gnn and smiles is not None and self._gnn_model is not None:
            try:
                nf, ei = _build_molecular_graph(smiles)
                with torch.no_grad():
                    preds = self._gnn_model(nf, ei)
                return {
                    "ec_reduction": float(preds[0].item()),
                    "dm_reduction": float(preds[1].item()),
                    "pf6_decomposition": float(preds[2].item()),
                    "polymerization": float(preds[3].item()),
                }
            except Exception:
                pass

        return self._linear_predictor.predict(descriptors=descriptors, smiles=smiles)

    def set_gnn_model(self, model: Tier0MPNN, model_path: str) -> None:
        """Set the GNN model explicitly.

        Args:
            model: The trained MPNN model.
            model_path: Path to the model weights file.
        """
        self._gnn_model = model
        self._use_gnn = True


class _LinearFallbackPredictor:
    """Original linear predictor (kept for backward compatibility).

    This is the v5.2 heuristic model that uses normalized descriptors
    with hardcoded weights. Used when MPNN is unavailable.
    """

    _SOLVENT_WEIGHTS = np.array([
        0.002, 0.08, -0.02, -0.03, -0.003, 0.01, 0.15, 0.005,
    ])
    _SOLVENT_BIAS = 0.70
    _SALT_WEIGHTS = np.array([
        0.001, 0.05, 0.01, 0.02, 0.002, 0.005, 0.10, 0.003,
    ])
    _SALT_BIAS = 1.15
    _POLY_WEIGHTS = np.array([
        0.001, 0.06, -0.01, -0.02, -0.002, 0.015, 0.20, 0.004,
    ])
    _POLY_BIAS = 0.45

    _MW_RANGE = (50, 500)
    _LOGP_RANGE = (-2, 5)
    _HBA_RANGE = (0, 10)
    _HBD_RANGE = (0, 5)
    _TPSA_RANGE = (0, 200)
    _ROT_RANGE = (0, 10)
    _ARO_RANGE = (0, 1)
    _HEAVY_RANGE = (5, 50)

    def predict(
        self,
        descriptors: dict[str, float] | None = None,
        smiles: str | None = None,
    ) -> dict[str, float]:
        """Predict using the original linear model.

        Args:
            descriptors: Optional molecular descriptors dict.
            smiles: Optional SMILES string.

        Returns:
            Dictionary with predicted activation energies.
        """
        if descriptors is None and smiles is None:
            return {
                "ec_reduction": 0.65,
                "dm_reduction": 0.75,
                "pf6_decomposition": 1.20,
                "polymerization": 0.40,
            }

        if descriptors is None:
            assert smiles is not None
            descriptors = _generate_molecular_descriptors(smiles)

        def _predict_single(desc: dict[str, float], weights: np.ndarray[Any, Any], bias: float) -> float:
            normalized = np.array([
                (desc.get("mw", 250) - self._MW_RANGE[0]) / (self._MW_RANGE[1] - self._MW_RANGE[0]),
                (desc.get("logp", 1.5) - self._LOGP_RANGE[0]) / (self._LOGP_RANGE[1] - self._LOGP_RANGE[0]),
                (desc.get("hba", 5) - self._HBA_RANGE[0]) / (self._HBA_RANGE[1] - self._HBA_RANGE[0]),
                (desc.get("hbd", 2) - self._HBD_RANGE[0]) / (self._HBD_RANGE[1] - self._HBD_RANGE[0]),
                (desc.get("tpsa", 100) - self._TPSA_RANGE[0]) / (self._TPSA_RANGE[1] - self._TPSA_RANGE[0]),
                (desc.get("rot_bonds", 5) - self._ROT_RANGE[0]) / (self._ROT_RANGE[1] - self._ROT_RANGE[0]),
                (desc.get("aromatic_ratio", 0.5) - self._ARO_RANGE[0]) / (self._ARO_RANGE[1] - self._ARO_RANGE[0]),
                (desc.get("heavy_atom_count", 25) - self._HEAVY_RANGE[0]) / (self._HEAVY_RANGE[1] - self._HEAVY_RANGE[0]),
            ])
            raw_ea = float(np.dot(normalized, weights) + bias)
            return float(np.clip(raw_ea, 0.30, 1.50))

        return {
            "ec_reduction": float(_predict_single(descriptors, self._SOLVENT_WEIGHTS, self._SOLVENT_BIAS)),
            "dm_reduction": float(_predict_single(descriptors, self._SOLVENT_WEIGHTS, self._SOLVENT_BIAS) * 1.15),
            "pf6_decomposition": float(_predict_single(descriptors, self._SALT_WEIGHTS, self._SALT_BIAS)),
            "polymerization": float(_predict_single(descriptors, self._POLY_WEIGHTS, self._POLY_BIAS)),
        }


def _generate_molecular_descriptors(smiles: str) -> dict[str, float]:
    """Generate molecular descriptors from SMILES (delegated to shared module).

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        Dictionary of descriptor name -> value.
    """
    from aurelius.utils.descriptors import _generate_molecular_descriptors as _gen
    return _gen(smiles)
