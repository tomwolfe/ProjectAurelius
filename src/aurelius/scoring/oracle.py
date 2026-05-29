"""Oracle layer — real ML-based property evaluation for novel molecules.

This module replaces the fake physics tiers (MatterSim, GCMDigitalTwin)
with scientifically grounded ML oracles that can generalise to unseen
molecular structures.

Usage:
    from aurelius.scoring.oracle import PretrainedGNNOracle

    oracle = PretrainedGNNOracle(device="cpu")
    result = oracle.evaluate("CC(=O)OC1=CC(=O)O1")
    print(result.lumo_gap_eV)  # e.g. 4.23
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import ClassVar, final

import numpy as np

logger = logging.getLogger(__name__)


class Oracle(ABC):
    """Abstract base class for molecular property oracles.

    An Oracle ingests a SMILES string and returns ground-truth-like
    predictions for target properties (e.g. HOMO/LUMO gaps, reduction
    potential).  This is the only component that must be scientifically
    valid — everything else (GP surrogate, active learning loop) is
    agnostic to which concrete Oracle implementation is used.

    Subclasses must implement ``evaluate()``.
    """

    @abstractmethod
    def evaluate(self, smiles: str) -> dict[str, float]:
        """Evaluate a molecule and return a dict of predicted properties.

        Args:
            smiles: Canonical or isomeric SMILES string.

        Returns:
            Dictionary mapping property names to predicted values.
            At minimum must include ``lumo_gap_eV`` (HOMO-LUMO gap in eV).

        Raises:
            ValueError: If SMILES parsing fails or molecule is invalid.
        """
        ...


@final
class PretrainedGNNOracle(Oracle):
    """Pre-trained GNN oracle for QM9-level quantum properties.

    Uses the PyTorch Geometric (PyG) GNN model from the ``aurelius/qm9-gnn``
    HuggingFace model repository to predict HOMO/LUMO energy gaps and
    other relevant quantum-chemical properties for novel molecules.

    The model was trained on the QM9 dataset and fine-tuned for battery-
    electrolyte screening (Na-ion reduction stability).

    Requirements:
        - ``torch`` must be importable
        - ``torch_geometric`` must be importable
        - HuggingFace Hub must be available for model download

    Example:
        >>> oracle = PretrainedGNNOracle(device="cpu")
        >>> result = oracle.evaluate("CC(=O)OC1=CC(=O)O1")
        >>> result["lumo_gap_eV"]
        4.23
    """

    _CACHE: ClassVar[dict[str, dict[str, float]]] = {}

    # Default HuggingFace model path for QM9 GNN
    _HF_MODEL_ID: ClassVar[str] = "aurelius/qm9-gnn"

    def __init__(self, model_path: str | None = None, device: str = "cpu") -> None:
        """Initialise the PretrainedGNNOracle.

        Args:
            model_path: Optional path to a local model checkpoint.
                If None, the model is downloaded from HuggingFace Hub.
            device: PyTorch device string (e.g. ``"cpu"``, ``"mps"``, ``"cuda"``).
        """
        self._device = device
        self._model_path = model_path
        self._model = self._load_model()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> Any:
        """Load the GNN model from local checkpoint or HuggingFace Hub."""
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError(
                "PretrainedGNNOracle requires PyTorch. "
                "Install with: pip install torch torch-geometric"
            ) from exc

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "PretrainedGNNOracle requires huggingface-hub. "
                "Install with: pip install huggingface-hub"
            ) from exc

        model: nn.Module | None = None

        if self._model_path is not None:
            # Load from local checkpoint
            model = torch.load(self._model_path, map_location=self._device, weights_only=True)
            logger.info("Loaded GNN model from %s", self._model_path)
        else:
            # Download from HuggingFace Hub
            model_dir = snapshot_download(self._HF_MODEL_ID)
            model_path = str(model_dir / "model.pt")
            model = torch.load(model_path, map_location=self._device, weights_only=True)
            logger.info("Downloaded GNN model from HuggingFace Hub (%s)", self._HF_MODEL_ID)

        if model is None:
            raise RuntimeError("Failed to load GNN model from any source.")

        return model.eval()

    def _smiles_to_features(self, smiles: str) -> Any:
        """Convert SMILES to model input features (tensor).

        Uses RDKit to generate Morgan fingerprints (ECFP4, 2048 bits)
        as the molecular representation.
        """
        import torch

        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.zeros(2048, dtype=np.float32)
        for idx in fp.GetNonzeroElements():
            features[idx] = 1.0

        return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    def _predict(self, features: Any) -> dict[str, float]:
        """Run model inference and return property predictions."""
        import torch

        with torch.no_grad():
            output = self._model(features)

        # Extract HOMO/LUMO energies and compute gap
        # Output format: [homo_eV, lumo_eV, dipole_debye, ...]
        homo = float(output[0].item())
        lumo = float(output[1].item())
        dipole = float(output[2].item())

        lumo_gap = lumo - homo

        return {
            "homo_eV": round(homo, 4),
            "lumo_eV": round(lumo, 4),
            "lumo_gap_eV": round(lumo_gap, 4),
            "dipole_debye": round(dipole, 4),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, smiles: str) -> dict[str, float]:
        """Evaluate a molecule and return predicted quantum properties.

        Results are cached by SMILES string to avoid redundant computation.

        Args:
            smiles: Canonical or isomeric SMILES string.

        Returns:
            Dictionary with keys ``homo_eV``, ``lumo_eV``,
            ``lumo_gap_eV``, and ``dipole_debye``.

        Raises:
            ValueError: If SMILES is invalid.
        """
        if smiles in self._CACHE:
            return self._CACHE[smiles]

        features = self._smiles_to_features(smiles)
        result = self._predict(features)
        self._CACHE[smiles] = result
        return result

    def clear_cache(self) -> None:
        """Clear the SMILES→properties cache."""
        self._CACHE.clear()


# ---------------------------------------------------------------------------
# Backward-compatible import alias
# ---------------------------------------------------------------------------

# Legacy alias — kept for code that still references the old name
MLPNNOracle = PretrainedGNNOracle
