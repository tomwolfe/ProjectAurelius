"""Cross-framework zero-copy bridging between MLX and PyTorch.

Provides DLpack-based memory sharing between MLX arrays and PyTorch
tensors on Apple Silicon, avoiding expensive deep-copy duplications
across framework boundaries.

Cross-Platform Compatibility:
    This module is designed to be importable on Linux/Windows (where MLX
    is unavailable) to support CI pipelines. MLX-specific imports are
    wrapped in try/except ImportError blocks. If MLX is not installed,
    the CrossFrameworkBridge class will still instantiate but will raise
    a descriptive RuntimeError when mlx_to_pytorch or pytorch_to_mlx
    is called.
"""

from __future__ import annotations

import logging
from typing import Any

from aurelius.utils.dependencies import HAS_MLX, HAS_TORCH

logger = logging.getLogger(__name__)

# Conditional framework imports (for runtime use when available)
_mx: Any = None
_torch: Any = None
if HAS_MLX:
    try:
        import mlx.core as mx  # noqa: F401
        _mx = mx
    except Exception:
        pass
if HAS_TORCH:
    try:
        import torch  # noqa: F401
        _torch = torch
    except Exception:
        pass

# Runtime error messages for graceful degradation
_MLX_NOT_AVAILABLE_MSG = (
    "MLX is not available on this platform. "
    "MLX is required for cross-framework bridging. "
    "On Linux/Windows, install MLX via: pip install mlx "
    "(Note: MLX is primarily optimized for Apple Silicon. "
    "For CPU-only CI pipelines, consider skipping MLX-dependent tests.)"
)

_TORCH_NOT_AVAILABLE_MSG = (
    "PyTorch is required for cross-framework bridging. "
    "Install with: pip install torch"
)

_DLPACK_UNSUPPORTED_MSG = (
    "This version of MLX does not support DLpack. "
    "Upgrade to mlx>=0.21.0 for cross-framework zero-copy bridging."
)

_TORCH_DLPACK_UNSUPPORTED_MSG = (
    "This version of PyTorch does not support DLpack. "
    "Upgrade to torch>=2.12.0 for cross-framework zero-copy bridging."
)


def bridge_mlx_to_pytorch(mlx_array: _mx.array) -> _torch.Tensor:
    """Bridge MLX array memory to PyTorch via DLPack.

    Uses DLpack to export the MLX array memory view into a DLPack
    capsule, then consumes it natively inside PyTorch targeting the
    same device as the source MLX array. This avoids explicit heavy
    deep-copy memory duplication.

    Args:
        mlx_array: An MLX core array to bridge.

    Returns:
        A PyTorch tensor on the same device as the source MLX array,
        sharing the underlying unified memory representation.

    Raises:
        RuntimeError: If MLX or PyTorch is not available on this platform.
        AttributeError: If the installed MLX version lacks DLpack support.
    """
    if not HAS_MLX:
        raise RuntimeError(_MLX_NOT_AVAILABLE_MSG)
    if not HAS_TORCH:
        raise RuntimeError(_TORCH_NOT_AVAILABLE_MSG)

    # Export the MLX array memory view into a DLPack capsule
    try:
        capsule = _mx.to_dlpack(mlx_array)  # type: ignore[attr-defined, unused-ignore]
    except AttributeError as err:
        raise AttributeError(_DLPACK_UNSUPPORTED_MSG) from err

    # Consume the capsule natively inside PyTorch
    torch_tensor = _torch.from_dlpack(capsule)  # type: ignore[attr-defined, unused-ignore]

    # Preserve the device from the source MLX array's underlying storage.
    # MLX arrays on Apple Silicon map to MPS in PyTorch; on CPU, they
    # stay on CPU. This avoids the previous hardcoded .to("mps") which
    # would silently fail or error on Linux/Windows.
    if _torch.backends.mps.is_available():
        try:
            torch_tensor = torch_tensor.to("mps")
        except RuntimeError:
            # MPS not available on this hardware; fall back to CPU
            torch_tensor = torch_tensor.to("cpu")

    return torch_tensor


def bridge_pytorch_to_mlx(torch_tensor: _torch.Tensor) -> _mx.array:
    """Bridge PyTorch tensor memory to MLX via DLPack.

    Uses DLpack to export the PyTorch tensor memory view into a
    DLPack capsule, then consumes it natively inside MLX.

    Args:
        torch_tensor: A PyTorch tensor to bridge.

    Returns:
        An MLX core array sharing the underlying unified memory.

    Raises:
        RuntimeError: If MLX or PyTorch is not available on this platform.
    """
    if not HAS_TORCH:
        raise RuntimeError(_TORCH_NOT_AVAILABLE_MSG)
    if not HAS_MLX:
        raise RuntimeError(_MLX_NOT_AVAILABLE_MSG)

    # Export the PyTorch tensor memory view into a DLPack capsule
    try:
        capsule = _torch.utils.dlpack.to_dlpack(torch_tensor)  # type: ignore[attr-defined, unused-ignore]
    except AttributeError as err:
        raise AttributeError(_TORCH_DLPACK_UNSUPPORTED_MSG) from err

    # Consume the capsule natively inside MLX
    try:
        mlx_array = _mx.from_dlpack(capsule)  # type: ignore[attr-defined, unused-ignore]
    except AttributeError as err:
        raise AttributeError(_DLPACK_UNSUPPORTED_MSG) from err

    return mlx_array  # type: ignore[no-any-return, unused-ignore]


class CrossFrameworkBridge:
    """Manages zero-copy data exchange between MLX and PyTorch.

    Provides a high-level interface for moving feature arrays between
    Tier 1 (MLX) and Tier 2 (PyTorch) without incurring explicit
    deep-copy memory duplication steps.

    Cross-Platform Behavior:
        On Linux/Windows (where MLX is unavailable), this class
        instantiates successfully but methods that require MLX will
        raise a descriptive RuntimeError on first usage. This allows
        the module to be imported in CI pipelines without errors,
        while still failing fast with a clear message when bridging
        is actually attempted.
    """

    def __init__(self) -> None:
        """Initialize the bridge.

        On platforms where MLX is unavailable (Linux/Windows), the
        bridge will still instantiate but will raise RuntimeError
        when mlx_to_pytorch or pytorch_to_mlx is called.
        """
        self._mlx_available = HAS_MLX
        self._torch_available = HAS_TORCH

    @property
    def is_available(self) -> bool:
        """Return True if both MLX and PyTorch are available."""
        return self._mlx_available and self._torch_available

    @property
    def mlx_available(self) -> bool:
        """Return True if MLX is available on this platform."""
        return self._mlx_available

    @property
    def torch_available(self) -> bool:
        """Return True if PyTorch is available on this platform."""
        return self._torch_available

    def mlx_to_pytorch(self, mlx_array: _mx.array) -> _torch.Tensor:
        """Bridge an MLX array to a PyTorch tensor on the same device.

        Raises:
            RuntimeError: If MLX is not available on this platform
                (e.g., Linux/Windows without MLX installed).
        """
        if not self._mlx_available:
            raise RuntimeError(_MLX_NOT_AVAILABLE_MSG)
        return bridge_mlx_to_pytorch(mlx_array)

    def pytorch_to_mlx(self, torch_tensor: _torch.Tensor) -> _mx.array:
        """Bridge a PyTorch tensor to an MLX array.

        Raises:
            RuntimeError: If MLX is not available on this platform.
        """
        if not self._mlx_available:
            raise RuntimeError(_MLX_NOT_AVAILABLE_MSG)
        return bridge_pytorch_to_mlx(torch_tensor)
