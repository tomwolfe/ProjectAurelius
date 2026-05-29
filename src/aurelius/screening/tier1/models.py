"""Tier 1: Neural network model definitions.

Contains all MLP model classes used for molecular viability screening:
- MLXBackend: MLX-compatible 2-layer MLP for ECFP4 fingerprints
- PyTorchBackend: PyTorch MLP replicating ChemVLM2MLP architecture
- ModelFactory: Returns the appropriate backend based on framework availability

References:
    Delaney, S. J. "ESOL: Estimating Aqueous Solubility
    Directly from Structure." J. Chem. Inf. Model. 2004.
    Glorot, X. & Bengio, Y. "Understanding the difficulty
    of training deep feedforward neural networks." AISTATS 2010.
"""

from __future__ import annotations

import os
from importlib import resources

from aurelius.screening.tier1.backend_mlx import (
    MLXBackend,
    MLXBackendUnavailableError,
)
from aurelius.screening.tier1.backend_mlx import (
    _model_factory as _mlx_model_factory,
)
from aurelius.screening.tier1.backend_torch import (
    PyTorchBackend,
    PyTorchBackendUnavailableError,
)
from aurelius.screening.tier1.backend_torch import (
    _model_factory as _torch_model_factory,
)
from aurelius.utils.dependencies import HAS_MLX, HAS_TORCH

__all__ = [
    "DEFAULT_MODEL_DIR",
    "HAS_MLX",
    "HAS_TORCH",
    "MLXBackend",
    "MLXBackendUnavailableError",
    "PyTorchBackend",
    "PyTorchBackendUnavailableError",
    "HUGGINGFACE_MODELS",
    "ModelFactory",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default model weight paths
DEFAULT_MODEL_DIR = os.environ.get(
    "AURELIUS_MODEL_DIR",
    str(resources.files("aurelius").joinpath("models")),
)

HUGGINGFACE_MODELS: dict[str, str] = {
    "esol_solubility": "aurelius/tier1-esol-mlp",
    "qm9_energy": "aurelius/tier1-qm9-mlp",
}


def ModelFactory() -> MLXBackend | PyTorchBackend:  # noqa: N802
    """Return the appropriate backend based on framework availability.

    Returns:
        MLXBackend or PyTorchBackend depending on availability.

    Raises:
        MLXBackendUnavailableError: If MLX is not available.
        PyTorchBackendUnavailableError: If PyTorch is not available.
    """
    if HAS_MLX:
        return _mlx_model_factory()
    if HAS_TORCH:
        return _torch_model_factory()
    raise PyTorchBackendUnavailableError(
        "Neither MLX nor PyTorch is available. Install mlx or torch to use model backends."
    )
