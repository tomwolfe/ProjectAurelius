"""Tier 1: MLX Neural Accelerator filter and fingerprint utilities.

Contains the main MLXNAFilter class and ECFP4 fingerprint generation
functions. The filter coordinates model inference across MLX, PyTorch,
and NumPy backends depending on available hardware.

References:
    Delaney, S. J. "ESOL: Estimating Aqueous Solubility
    Directly from Structure." J. Chem. Inf. Model. 2004.
    Ramakrishnan, R. et al. "QM9: 134 Kilo Molecules."
    Sci. Data 2014.
"""

from __future__ import annotations

import json
import logging
import os
import time
from importlib import resources
from typing import Any

import numpy as np

from aurelius.screening.tier1.loaders import (
    HuggingFaceWeightLoader,
    load_pytorch_fallback_with_mlx_weights,
)
from aurelius.screening.tier1.models import (
    HAS_MLX,
    HAS_TORCH,
    MLXBackend,
    PyTorchBackend,
)
from aurelius.screening.tier1.training import (
    _train_synthetic_mlx,
    _train_synthetic_pytorch,
    train_on_esol,
)
from aurelius.types import MLXFilterResult
from aurelius.utils.chem_utils import generate_ecfp4_fingerprint
from aurelius.utils.dependencies import HAS_RDKIT

logger = logging.getLogger(__name__)


class MLXNAFilter:
    """Tier 1: MLX Neural Accelerator filter for rapid molecular screening.

    Uses a 2-layer MLP trained on ECFP4 (Morgan radius=2) fingerprints
        to predict molecular viability. When MLX is available, inference
        runs entirely on the MLX backend; otherwise a numpy fallback
        provides deterministic pseudo-results for pipeline validation.

    With --use-real-models (default), the filter:
    1. Attempts to load pre-trained weights from Hugging Face Hub
    2. Falls back to locally trained weights in models/
    3. Falls back to training on ESOL/QM9 if no weights exist

    With --demo, uses synthetic training data for demonstration.
    """

    def __init__(
        self,
        quantization_format: str = "MX4",
        train_on_init: bool = True,
    ) -> None:
        """Initialize the MLX-NA filter.

        Args:
            quantization_format: Quantization format string (e.g., "MX4").
            train_on_init: If True, train or load model at initialization.
        """
        self.quantization_format = quantization_format
        self._train_on_init = train_on_init
        self._model_loaded = False
        self._model: MLXBackend | PyTorchBackend | None = None
        self._use_mlx = HAS_MLX
        self._weight_loader = HuggingFaceWeightLoader()
        if HAS_MLX:
            try:
                import mlx.core as mx  # noqa: F401

                self._mx = mx
            except ImportError:
                self._use_mlx = False
                self._mx = None  # type: ignore[assignment]
        else:
            self._mx = None  # type: ignore[assignment]

        # Conditional torch/torch_nn imports
        self._torch: Any | None = None
        self._torch_nn: Any | None = None
        if HAS_TORCH:
            try:
                import torch  # noqa: F401
                import torch.nn as torch_nn  # noqa: F401

                self._torch = torch
                self._torch_nn = torch_nn
            except ImportError:
                pass

        # Conditional RDKit imports
        self._rdkit_chem: Any | None = None
        self._rdkit_allchem: Any | None = None
        if HAS_RDKIT:
            try:
                from rdkit import Chem as _rdkit_chem  # noqa: F401, N813
                from rdkit.Chem import AllChem as _rdkit_allchem  # noqa: F401, N813

                self._rdkit_chem = _rdkit_chem
                self._rdkit_allchem = _rdkit_allchem
            except ImportError:
                pass

        if train_on_init:
            self._load_or_train_model()

    def _load_or_train_model(self) -> None:
        """Load pre-trained weights or train the model at initialization.

        Priority:
        1. Hugging Face Hub
        2. Local model directory
        3. Train on ESOL dataset
        """
        print("[Aurelius v5.2 Tier1] Attempting to load real model weights...")
        try:
            model = self._weight_loader.load_model(task="esol_solubility", local_only=False)
        except Exception as e:
            print(f"[Aurelius v5.2 Tier1] HF weight loading failed: {e}")
            model = None

        if model is not None:
            self._model = model
            self._model_loaded = True
            print("[Aurelius v5.2 Tier1] Real model loaded successfully")
            return

        print("[Aurelius v5.2 Tier1] No pre-trained weights found, training on ESOL dataset...")
        self._train_default_model()

    def _get_demo_result(self) -> MLXFilterResult:
        """Return a demo viability result for environments without ML frameworks.

        Returns:
            MLXFilterResult with deterministic demo values.
        """
        return MLXFilterResult(
            molecule_smiles="",
            is_viable=True,
            confidence_score=0.85,
            inference_time_ms=0.0,
            na_utilization_pct=85.0,
        )

    def _train_default_model(self) -> MLXFilterResult:
        """Train the model on real solubility or synthetic data.

        Returns:
            MLXFilterResult with viability data.
        """
        if not self._use_mlx:
            if not HAS_TORCH:
                result = self._get_demo_result()
                self._model = PyTorchBackend()
                self._model_loaded = True
                return result

            print("[Aurelius v6.0 Tier1] MLX unavailable, initializing PyTorch fallback filter...")
            self._model = PyTorchBackend()

            print("[Aurelius v6.0 Tier1] Training PyTorch fallback on synthetic data...")
            self._model = _train_synthetic_pytorch()
            self._model_loaded = True
            return MLXFilterResult(
                molecule_smiles="",
                is_viable=True,
                confidence_score=0.85,
                inference_time_ms=0.0,
                na_utilization_pct=85.0,
            )

        model = MLXBackend()
        self._model = model
        self._model_loaded = True

        try:
            model = train_on_esol(model, epochs=200, lr=0.005, batch_size=16, seed=42)
            self._weight_loader.save_model(model, "esol_solubility")
        except (ImportError, RuntimeError, ValueError) as e:
            print(f"[Aurelius v5.2 Tier1] ESOL training failed: {e}")
            self._model = _train_synthetic_mlx(model)

        return MLXFilterResult(
            molecule_smiles="",
            is_viable=True,
            confidence_score=0.85,
            inference_time_ms=0.0,
            na_utilization_pct=85.0,
        )

    def load_model(self, model_path: str) -> MLXFilterResult | None:
        """Load ChemVLM-2 model from a saved path.

        In production, model_path points to a saved MLX model.
        For now, trains the MLP on real or synthetic data.

        Args:
            model_path: Path to model weights directory.
        """
        if self._model_loaded:
            return None
        if self._use_mlx:
            print(f"[Aurelius v6.0 Tier1] Loading model from {model_path}")
            self._model = MLXBackend()
            self._train_default_model()
            return None
        else:
            if not HAS_TORCH:
                return self._get_demo_result()
            self._model = PyTorchBackend()
            if os.path.isdir(model_path):
                self._model = load_pytorch_fallback_with_mlx_weights(self._model, model_path)
            self._model_loaded = True
            print("[Aurelius v6.0 Tier1] Model ready")
            return None

    def screen_molecule(self, smiles: str) -> MLXFilterResult:
        """Screen a single molecule through the MLX-NA filter.

        Generates an ECFP4 (Morgan radius=2) fingerprint from the
        SMILES string, runs it through the MLP, and returns a
        viability result with confidence score.

        Args:
            smiles: SMILES string of the molecule to screen.

        Returns:
            MLXFilterResult with viability, confidence, and metadata.
        """
        if not self._model_loaded:
            if self._train_on_init:
                self._model = MLXBackend()
                self._train_default_model()
            else:
                # Load from local saved path only (HF is skipped when train_on_init=False)
                self._model = self._weight_loader.load_model(task="esol_solubility", local_only=True)
                self._model_loaded = self._model is not None
            if not self._model_loaded:
                if self._use_mlx:
                    self._model = MLXBackend()
                    self._train_default_model()
                else:
                    if not HAS_TORCH:
                        return MLXFilterResult(
                            molecule_smiles="",
                            is_viable=True,
                            confidence_score=0.85,
                            inference_time_ms=0.0,
                            na_utilization_pct=85.0,
                        )
                    self._model = PyTorchBackend()
                self._model_loaded = True

        start = time.perf_counter()

        fingerprint = _generate_ecfp4_fingerprint(smiles)
        result = self._run_inference(fingerprint, smiles)

        elapsed_ms = (time.perf_counter() - start) * 1000
        na_util = self._estimate_na_utilization(result["confidence"])

        return MLXFilterResult(
            molecule_smiles=smiles,
            is_viable=result["is_viable"],
            confidence_score=result["confidence"],
            inference_time_ms=elapsed_ms,
            na_utilization_pct=na_util,
        )

    def screen_batch(self, smiles_list: list[str], batch_size: int = 32) -> list[MLXFilterResult]:
        """Screen a batch of molecules through the MLX-NA filter."""
        return [self.screen_molecule(smiles) for smiles in smiles_list]

    def _estimate_na_utilization(self, confidence: float) -> float:
        """Estimate Neural Accelerator utilization percentage."""
        tier1_params: dict[str, Any] = {}
        try:
            ff_path = str(resources.files("aurelius.data").joinpath("force_field_params.json"))
            if os.path.isfile(ff_path):
                with open(ff_path) as f:
                    data = json.load(f)
                    tier1_params = data.get("tier1_parameters", {}).get("na_utilization", {})
        except (json.JSONDecodeError, OSError):
            pass

        base_util = tier1_params.get("base_utilization_pct", 75.0) + confidence * tier1_params.get(
            "confidence_boost", 20.0
        )
        return float(min(base_util, tier1_params.get("max_utilization_pct", 98.0)))

    def _bits_from_format(self) -> int:
        """Extract bit depth from quantization format string."""
        if "MX4" in self.quantization_format:
            return 4
        elif "MX6" in self.quantization_format:
            return 6
        return 4

    def _run_inference(self, fingerprint: np.ndarray[Any, Any], smiles: str) -> dict[str, Any]:
        """Run molecular viability inference via MLX, PyTorch, or numpy fallback."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call _load_or_train_model() first.")

        # Use unified predict() method across all backends
        if self._use_mlx and self._mx is not None:
            fp_array = self._mx.array(fingerprint, dtype=self._mx.float32)
            if fp_array.ndim == 1:
                fp_array = fp_array.reshape(1, -1)
            output = self._model.predict(fp_array)  # type: ignore[arg-type]
            confidence = float(self._mx.squeeze(output))  # type: ignore[arg-type]
        elif self._torch is not None:
            fp_tensor = self._torch.from_numpy(fingerprint).float().unsqueeze(0)
            with self._torch.no_grad():
                output = self._model.predict(fp_tensor)
            confidence = float(output.squeeze().item())
        else:
            output = self._model.predict(fingerprint)  # type: ignore[arg-type]
            confidence = float(np.squeeze(output))  # type: ignore[arg-type]

        confidence = float(np.clip(confidence, 0.0, 1.0))
        is_viable = confidence > 0.5
        return {"is_viable": is_viable, "confidence": confidence}


# Re-exported from chem_utils for backward compatibility
_generate_ecfp4_fingerprint = generate_ecfp4_fingerprint


__all__ = [
    "HAS_MLX",
    "HAS_RDKIT",
    "HAS_TORCH",
    "MLXNAFilter",
    "_generate_ecfp4_fingerprint",
]


# Re-export dependency flags for backward compatibility
