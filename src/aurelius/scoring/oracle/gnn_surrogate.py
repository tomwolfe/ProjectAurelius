"""GNNQuantumOracle — Lightweight Equivariant GNN Surrogate for HOMO/LUMO.

Provides a scikit-learn-compatible wrapper that loads a pre-trained,
lightweight graph model (e.g., a small NequIP or MACE model exported to
ONNX) for predicting HOMO/LUMO energies. If the ONNX binary is not
present, gracefully falls back to None (no-op).

The ONNX runtime (onnxruntime) is the only additional dependency.
No PyTorch, TensorFlow, or JAX is required at inference time.

Physical justification: Equivariant GNNs (e.g., NequIP, MACE) respect
the physical symmetries of molecular systems (SE(3) equivariance) and
can achieve sub-0.5 eV MAE on HOMO/LUMO prediction with minimal training
data when pre-trained on large QM datasets. This provides a higher-fidelity
alternative to the TOM fallback without requiring the xTB binary.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from aurelius.types import MoleculeContext

logger = logging.getLogger(__name__)


def _check_onnx() -> bool:
    """Check if onnxruntime is available."""
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


_HAS_ONNX: bool = _check_onnx()


class GNNQuantumOracle:
    """Lightweight GNN surrogate for HOMO/LUMO via ONNX runtime.

    Loads a pre-trained, lightweight equivariant GNN model exported to
    ONNX format. The model should accept ECFP4-based feature vectors
    (2048-bit Morgan fingerprint + molecular descriptors) and return
    (homo_eV, lumo_eV).

    If no model file is found or onnxruntime is unavailable, this class
    gracefully degrades: ``predict()`` returns (None, None) and
    ``is_available`` returns False.

    Usage:
        gnn = GNNQuantumOracle()
        if gnn.is_available:
            homo, lumo = gnn.predict(ctx)
    """

    def __init__(
        self,
        model_path: str | None = None,
    ) -> None:
        self._model_path = model_path
        self._session: Any = None
        self._available = False

        if not _HAS_ONNX:
            logger.info(
                "GNNQuantumOracle: onnxruntime not installed — GNN surrogate disabled."
            )
            return

        if model_path is None:
            import os
            module_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(
                module_dir, "..", "..", "..", "..", "models",
                "gnn_quantum_oracle.onnx",
            )

        try:
            import onnxruntime
            if os.path.exists(model_path):
                self._session = onnxruntime.InferenceSession(model_path)
                self._available = True
                logger.info("GNNQuantumOracle: loaded model from %s", model_path)
            else:
                logger.info(
                    "GNNQuantumOracle: model not found at %s — GNN surrogate disabled.",
                    model_path,
                )
        except Exception as exc:
            logger.info("GNNQuantumOracle: failed to load model (%s)", exc)
            self._session = None
            self._available = False

    @property
    def is_available(self) -> bool:
        """Return True if the ONNX model is loaded and ready for inference."""
        return self._available

    def predict(self, ctx: MoleculeContext) -> tuple[float | None, float | None]:
        """Predict (homo_eV, lumo_eV) using the ONNX GNN model.

        Returns (None, None) if the model is unavailable.

        The feature vector is the same 2053-dim vector used by the
        SurrogateQuantumOracle (2048-bit ECFP4 + MW + LogP + TPSA +
        RingCount + RotatableBonds).
        """
        if not self._available or self._session is None:
            return None, None

        try:
            features = ctx.get_feature_vector().astype(np.float32).reshape(1, -1)
            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: features})
            homo = float(outputs[0][0][0]) if len(outputs) > 0 else None
            lumo = float(outputs[0][0][1]) if len(outputs[0][0]) > 1 else None
            return homo, lumo
        except Exception as exc:
            logger.debug("GNNQuantumOracle inference failed: %s", exc)
            return None, None

    def compute_penalty(self, homo_eV: float | None) -> float:
        """Return 0.5x penalty if GNN predicts unstable HOMO (> -5.0 eV).

        Matches the SurrogateQuantumOracle interface for use as a drop-in
        replacement.
        """
        if homo_eV is not None and homo_eV > -5.0:
            return 0.5
        return 1.0
