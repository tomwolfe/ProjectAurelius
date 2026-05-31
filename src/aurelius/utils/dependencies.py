"""Centralized dependency detection for Project Aurelius.

Provides simple boolean availability flags for optional frameworks.
"""

from __future__ import annotations

HAS_TORCH: bool = False
HAS_RDKIT: bool = False
HAS_HF_HUB: bool = False
HAS_DATASETS: bool = False

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    pass

try:
    from rdkit import Chem  # noqa: F401

    HAS_RDKIT = True
except ImportError:
    pass

try:
    __import__("huggingface_hub")
    HAS_HF_HUB = True
except ImportError:
    pass

try:
    __import__("datasets")
    HAS_DATASETS = True
except ImportError:
    pass
