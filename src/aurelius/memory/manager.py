"""Phase 1: Zero-Copy Memory Management.

Manages zero-copy memory between MLX and PyTorch on M-series NPUs.
Uses dynamic RAM detection via psutil for hardware-agnostic allocation.
Replaces private torch._C APIs with safe wrappers.
"""

from __future__ import annotations
