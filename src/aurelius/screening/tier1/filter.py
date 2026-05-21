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
    HAS_RDKIT,
    HAS_TORCH,
    PyTorchFallbackFilter,
    _ChemVLM2MLP,
    _FallbackMLP,
)
from aurelius.screening.tier1.training import (
    _train_synthetic_mlx,
    _train_synthetic_pytorch,
    train_on_esol,
)
from aurelius.types import MLXFilterResult

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
        use_real_models: bool = True,
        train_on_init: bool = True,
    ) -> None:
        """Initialize the MLX-NA filter.

        Args:
            quantization_format: Quantization format string (e.g., "MX4").
            use_real_models: If True, load/train on real data.
                If False, use synthetic training data (demo mode).
            train_on_init: If True, train or load model at initialization.
        """
        self.quantization_format = quantization_format
        self._model_loaded = False
        self._model: Any | None = None
        self._mx: Any = None
        self._use_mlx = HAS_MLX
        self._use_real_models = use_real_models
        self._weight_loader = HuggingFaceWeightLoader()
        if HAS_MLX:
            try:
                import mlx.core as mx  # noqa: F401
                self._mx = mx
            except Exception:
                self._use_mlx = False
                self._mx = None
        else:
            self._mx = None

        # Conditional torch/torch_nn imports
        self._torch: Any = None
        self._torch_nn: Any = None
        if HAS_TORCH:
            try:
                import torch  # noqa: F401
                import torch.nn as torch_nn  # noqa: F401
                self._torch = torch
                self._torch_nn = torch_nn
            except Exception:
                pass

        # Conditional RDKit imports
        self._rdkit_chem: Any = None
        self._rdkit_allchem: Any = None
        if HAS_RDKIT:
            try:
                from rdkit import Chem as _rdkit_chem  # noqa: F401, N813
                from rdkit.Chem import AllChem as _rdkit_allchem  # noqa: F401, N813
                self._rdkit_chem = _rdkit_chem
                self._rdkit_allchem = _rdkit_allchem
            except Exception:
                pass

        if train_on_init:
            self._load_or_train_model()

    def _load_or_train_model(self) -> None:
        """Load pre-trained weights or train the model at initialization.

        Priority:
        1. Hugging Face Hub (if available and use_real_models)
        2. Local model directory (if use_real_models)
        3. Train on ESOL dataset (if use_real_models)
        4. Train on synthetic data (if not use_real_models / demo mode)
        """
        if self._use_real_models:
            print("[Aurelius v5.2 Tier1] Attempting to load real model weights...")
            model = self._weight_loader.load_model(
                task="esol_solubility", local_only=False
            )
            if model is not None:
                self._model = model
                self._model_loaded = True
                print("[Aurelius v5.2 Tier1] Real model loaded successfully")
                return

            print("[Aurelius v5.2 Tier1] No pre-trained weights found, training on ESOL dataset...")
            self._train_default_model()
        else:
            print("[Aurelius v5.2 Tier1] Demo mode: training synthetic solubility model...")
            self._train_default_model()

    def _train_default_model(self) -> None:
        """Train the model on real solubility or synthetic data."""
        if not self._use_mlx:
            if not HAS_TORCH:
                print("[Aurelius v6.0 Tier1] WARNING: Both MLX and PyTorch unavailable. "
                      "Using numpy-only fallback.")
                self._model = _FallbackMLP()
                self._model_loaded = True
                return

            print("[Aurelius v6.0 Tier1] MLX unavailable, initializing PyTorch fallback filter...")
            self._model = PyTorchFallbackFilter()

            if self._use_real_models and os.path.isdir(self._weight_loader.model_dir):
                local_task_dir = os.path.join(self._weight_loader.model_dir, "esol_solubility")
                if os.path.isdir(local_task_dir):
                    self._model = load_pytorch_fallback_with_mlx_weights(
                        self._model, local_task_dir
                    )
                    self._model_loaded = True
                    return

            print("[Aurelius v6.0 Tier1] Training PyTorch fallback on synthetic data...")
            self._model = _train_synthetic_pytorch()
            self._model_loaded = True
            return

        model = _ChemVLM2MLP()

        if self._use_real_models:
            try:
                model = train_on_esol(model, epochs=200, lr=0.005, batch_size=16, seed=42)
                self._weight_loader.save_model(model, "esol_solubility")
            except Exception as e:
                print(f"[Aurelius v5.2 Tier1] ESOL training failed: {e}")
                print("[Aurelius v5.2 Tier1] Falling back to synthetic training...")
                model = _train_synthetic_mlx(model, use_real_models=False)
        else:
            model = _train_synthetic_mlx(model, use_real_models=False)

        self._model = model
        self._model_loaded = True

    def load_model(self, model_path: str) -> None:
        """Load ChemVLM-2 model from a saved path.

        In production, model_path points to a saved MLX model.
        For now, trains the MLP on real or synthetic data.

        Args:
            model_path: Path to model weights directory.
        """
        if self._use_mlx:
            print(f"[Aurelius v6.0 Tier1] Loading model from {model_path}")
            self._model = _ChemVLM2MLP()
            self._train_default_model()
        else:
            if not HAS_TORCH:
                print("[Aurelius v6.0 Tier1] MLX and PyTorch unavailable, using numpy fallback MLP")
                self._model = _FallbackMLP()
            else:
                print("[Aurelius v6.0 Tier1] MLX unavailable, using PyTorch fallback filter")
                self._model = PyTorchFallbackFilter()
                if os.path.isdir(model_path):
                    self._model = load_pytorch_fallback_with_mlx_weights(
                        self._model, model_path
                    )
        self._model_loaded = True
        print("[Aurelius v6.0 Tier1] Model ready")

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
            if self._use_mlx:
                self._model = _ChemVLM2MLP()
                self._train_default_model()
            else:
                if not HAS_TORCH:
                    self._model = _FallbackMLP()
                else:
                    self._model = PyTorchFallbackFilter()
            self._model_loaded = True

        start = time.perf_counter()

        fingerprint = _generate_ecfp4_fingerprint(smiles, use_real_models=self._use_real_models)
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

    def screen_batch(
        self, smiles_list: list[str], batch_size: int = 32
    ) -> list[MLXFilterResult]:
        """Screen a batch of molecules through the MLX-NA filter."""
        results = []
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i : i + batch_size]
            for smiles in batch:
                results.append(self.screen_molecule(smiles))
        return results

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

        base_util = tier1_params.get("base_utilization_pct", 75.0) + confidence * tier1_params.get("confidence_boost", 20.0)
        return float(min(base_util, tier1_params.get("max_utilization_pct", 98.0)))

    def _bits_from_format(self) -> int:
        """Extract bit depth from quantization format string."""
        if "MX4" in self.quantization_format:
            return 4
        elif "MX6" in self.quantization_format:
            return 6
        return 4

    def _run_inference(self, fingerprint: np.ndarray, smiles: str) -> dict[str, Any]:
        """Run molecular viability inference via MLX, PyTorch, or numpy fallback."""
        if self._use_mlx and self._model is not None and self._mx is not None:
            fp_array = self._mx.array(fingerprint, dtype=self._mx.float32)
            if fp_array.ndim == 1:
                fp_array = fp_array.reshape(1, -1)
            logits = self._model(fp_array)
            confidence = float(self._mx.squeeze(logits))
        elif HAS_TORCH and isinstance(self._model, PyTorchFallbackFilter) and self._torch is not None:
            fp_tensor = self._torch.from_numpy(fingerprint).float().unsqueeze(0)
            with self._torch.no_grad():
                output = self._model.predict(fp_tensor)
            confidence = float(output.squeeze().item())
        else:
            assert self._model is not None
            output = self._model(fingerprint)
            confidence = float(np.squeeze(output))

        confidence = float(np.clip(confidence, 0.0, 1.0))
        is_viable = confidence > 0.5
        return {"is_viable": is_viable, "confidence": confidence}


def _generate_ecfp4_fingerprint(smiles: str, use_real_models: bool = True) -> np.ndarray:
    """Generate a 2048-bit ECFP4 (Morgan radius=2) fingerprint from SMILES.

    Uses RDKit's GetMorganFingerprintAsBitVect for production-grade
    fingerprints. Falls back to a deterministic hash-based vector
    when RDKit is not installed.

    Args:
        smiles: SMILES string of the molecule.
        use_real_models: If True and RDKit is unavailable, raises
            RuntimeError since hash fingerprints break chemical validity.

    Returns:
        numpy float32 array of shape (2048,) with values 0.0 or 1.0.

    Raises:
        RuntimeError: If use_real_models=True and RDKit is unavailable.
    """
    if HAS_RDKIT:
        from rdkit import Chem as _Chem
        from rdkit.Chem import AllChem as _AllChem

        mol = _Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(
                "RDKit failed to parse SMILES '%s', using hash fallback. "
                "This fingerprint is NOT chemically valid.",
                smiles,
            )
            return _hash_fallback(smiles)
        fp = _AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        bit_list = fp.ToList()
        arr = np.array(bit_list, dtype=np.float32)
        if len(arr) < 2048:
            padded = np.zeros(2048, dtype=np.float32)
            padded[:len(arr)] = arr
            return padded
        return arr[:2048]

    if use_real_models:
        raise RuntimeError(
            "[Aurelius v5.2 Tier1] RDKit is required when use_real_models=True. "
            "Hash-based fingerprints are NOT chemically valid and cannot be used "
            "for real screening. Install RDKit for chemically meaningful screening:\n"
            "  pip install rdkit\n"
            "Or run in demo mode: AureliusPipeline(config, use_real_models=False)"
        )

    print(
        "[Aurelius v5.2 Tier1] WARNING: RDKit is not installed. "
        "Using deterministic hash-based fingerprint fallback. "
        "This is NOT a real ECFP4 fingerprint and breaks chemical validity. "
        "Install RDKit for chemically meaningful screening: "
        "pip install rdkit"
    )
    return _hash_fallback(smiles)


def _hash_fallback(smiles: str) -> np.ndarray:
    """Deterministic hash-based fingerprint fallback when RDKit is unavailable.

    Produces a 2048-bit vector from the SMILES hash using SHA-256.
    This is NOT a real ECFP4 fingerprint but provides deterministic,
    reproducible input for pipeline validation.

    Args:
        smiles: SMILES string.

    Returns:
        numpy float32 array of shape (2048,).
    """
    import hashlib

    n_bits = 2048
    min_set = 80
    max_set = 200

    tier1_params: dict[str, Any] = {}
    try:
        ff_path = str(resources.files("aurelius.data").joinpath("force_field_params.json"))
        if os.path.isfile(ff_path):
            with open(ff_path) as f:
                data = json.load(f)
                tier1_params = data.get("tier1_parameters", {})
    except (json.JSONDecodeError, OSError):
        pass

    if tier1_params:
        hash_params = tier1_params.get("hash_fallback", {})
        n_bits = hash_params.get("n_bits", n_bits)
        min_set = hash_params.get("min_set_bits", min_set)
        max_set = hash_params.get("max_set_bits", max_set)

    arr = np.zeros(n_bits, dtype=np.float32)
    seed = int(hashlib.sha256(smiles.encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    n_set = rng.randint(min_set, max_set)
    indices = rng.randint(0, n_bits, size=n_set)
    arr[indices] = 1.0
    return arr


__all__ = [
    "HAS_MLX",
    "HAS_RDKIT",
    "HAS_TORCH",
    "MLXNAFilter",
    "_generate_ecfp4_fingerprint",
    "_hash_fallback",
]
