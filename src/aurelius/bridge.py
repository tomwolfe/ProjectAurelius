"""Cross-framework zero-copy bridging between MLX and PyTorch.

Provides DLpack-based memory sharing between MLX arrays and PyTorch
tensors on Apple Silicon, avoiding expensive deep-copy duplications
across framework boundaries.
"""


try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    mx = None  # type: ignore
    HAS_MLX = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    HAS_TORCH = False


def bridge_mlx_to_pytorch(mlx_array: "mx.array") -> "torch.Tensor":
    """Bridge MLX array memory to PyTorch via DLPack.

    Uses DLpack to export the MLX array memory view into a DLPack
    capsule, then consumes it natively inside PyTorch targeting the
    MPS device. This avoids explicit heavy deep-copy memory duplication.

    Args:
        mlx_array: An MLX core array to bridge.

    Returns:
        A PyTorch tensor on the MPS device sharing the underlying
        unified memory representation.

    Raises:
        RuntimeError: If MLX or PyTorch is not available.
        AttributeError: If the installed MLX version lacks DLpack support.
    """
    if not HAS_MLX:
        raise RuntimeError("MLX is required for MLX-to-PyTorch bridging.")
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for MLX-to-PyTorch bridging.")

    # Export the MLX array memory view into a DLPack capsule
    try:
        capsule = mx.to_dlpack(mlx_array)
    except AttributeError as err:
        raise AttributeError(
            "This version of MLX does not support DLpack. "
            "Upgrade to mlx>=0.21.0 for cross-framework zero-copy bridging."
        ) from err

    # Consume the capsule natively inside PyTorch targeting the MPS device
    torch_tensor = torch.from_dlpack(capsule)

    # Ensure the tensor is assigned to the MPS device space
    if torch.backends.mps.is_available():
        torch_tensor = torch_tensor.to("mps")

    return torch_tensor


def bridge_pytorch_to_mlx(torch_tensor: "torch.Tensor") -> "mx.array":
    """Bridge PyTorch tensor memory to MLX via DLPack.

    Uses DLpack to export the PyTorch tensor memory view into a
    DLPack capsule, then consumes it natively inside MLX.

    Args:
        torch_tensor: A PyTorch tensor to bridge.

    Returns:
        An MLX core array sharing the underlying unified memory.

    Raises:
        RuntimeError: If MLX or PyTorch is not available.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for PyTorch-to-MLX bridging.")
    if not HAS_MLX:
        raise RuntimeError("MLX is required for PyTorch-to-MLX bridging.")

    # Export the PyTorch tensor memory view into a DLPack capsule
    try:
        capsule = torch.utils.dlpack.to_dlpack(torch_tensor)
    except AttributeError as err:
        raise AttributeError(
            "This version of PyTorch does not support DLpack. "
            "Upgrade to torch>=2.12.0 for cross-framework zero-copy bridging."
        ) from err

    # Consume the capsule natively inside MLX
    try:
        mlx_array = mx.from_dlpack(capsule)
    except AttributeError as err:
        raise AttributeError(
            "This version of MLX does not support DLpack. "
            "Upgrade to mlx>=0.21.0 for cross-framework zero-copy bridging."
        ) from err

    return mlx_array


class CrossFrameworkBridge:
    """Manages zero-copy data exchange between MLX and PyTorch.

    Provides a high-level interface for moving feature arrays between
    Tier 1 (MLX) and Tier 2 (PyTorch) without incurring explicit
    deep-copy memory duplication steps.
    """

    def __init__(self) -> None:
        self._mlx_available = HAS_MLX
        self._torch_available = HAS_TORCH

    @property
    def is_available(self) -> bool:
        """Return True if both MLX and PyTorch are available."""
        return self._mlx_available and self._torch_available

    def mlx_to_pytorch(self, mlx_array: "mx.array") -> "torch.Tensor":
        """Bridge an MLX array to a PyTorch MPS tensor."""
        return bridge_mlx_to_pytorch(mlx_array)

    def pytorch_to_mlx(self, torch_tensor: "torch.Tensor") -> "mx.array":
        """Bridge a PyTorch MPS tensor to an MLX array."""
        return bridge_pytorch_to_mlx(torch_tensor)
