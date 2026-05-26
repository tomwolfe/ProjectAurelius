"""Tier 0: Activation energy predictor wrapper.

Provides both GNN-based and linear-fallback prediction interfaces
for molecule-specific activation energies.
"""

from __future__ import annotations

import os

from aurelius.screening.tier0.data import (
    _build_molecular_graph,
)
from aurelius.screening.tier0.models import PyTorchBackend
from aurelius.utils.dependencies import HAS_TORCH

if HAS_TORCH:
    import torch  # noqa: F401


class Tier0ActivationPredictor:
    """Predictor for molecule-specific activation energies using MPNN.

    When MPNN weights are available, uses the GNN for predictions.
    Raises RuntimeError if model weights cannot be loaded, ensuring
    failures are visible rather than silently hidden.
    """

    def __init__(self, model_path: str | None = None) -> None:
        """Initialize the predictor.

        Args:
            model_path: Optional path to MPNN weights. If provided and
                the file exists, loads the GNN model. Otherwise raises
                a clear error.

        Raises:
            RuntimeError: If model_path is provided but invalid, or if
                the GNN model fails to load.
        """
        self._use_gnn = False
        self._gnn_model: PyTorchBackend | None = None

        if model_path and os.path.isfile(model_path):
            if HAS_TORCH:
                try:
                    self._gnn_model = PyTorchBackend(node_dim=4, edge_dim=0, hidden_dim=64, output_dim=4)
                    self._gnn_model.load_weights(model_path)
                    self._gnn_model.eval()  # type: ignore[attr-defined]
                    self._use_gnn = True
                except (FileNotFoundError, RuntimeError) as e:
                    raise RuntimeError(
                        "Tier 0 MPNN weights not found. Run: aurelius train --task tier0"
                    ) from e
        else:
            raise RuntimeError(
                "Tier 0 MPNN weights not found. Run: aurelius train --task tier0"
            )

    def predict(
        self,
        descriptors: dict[str, float] | None = None,
        smiles: str | None = None,
    ) -> dict[str, float]:
        """Predict molecule-specific activation energies.

        Uses MPNN if available and SMILES is provided.

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
            except (ImportError, ValueError, RuntimeError):
                raise RuntimeError(
                    "MPNN prediction failed. Run: aurelius train --task tier0"
                ) from None

        raise RuntimeError(
            "Tier 0 MPNN weights not found. Run: aurelius train --task tier0"
        )

    def set_gnn_model(self, model: PyTorchBackend, model_path: str) -> None:
        """Set the GNN model explicitly.

        Args:
            model: The trained MPNN model.
            model_path: Path to the model weights file.
        """
        self._gnn_model = model
        self._use_gnn = True
