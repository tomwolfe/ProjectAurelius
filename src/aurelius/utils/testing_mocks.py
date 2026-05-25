"""Testing-only hash-based fingerprint fallback.

This module provides hash-based fingerprint generation for testing
and demo purposes. It is NOT suitable for production use because
hash-based fingerprints are NOT chemically valid.

For production, use real ECFP4 fingerprints from RDKit instead.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def _hash_fallback(smiles: str, n_bits: int = 2048) -> np.ndarray[Any, Any]:
    """Deterministic hash-based fingerprint fallback for testing.

    Produces a 2048-bit vector from the SMILES hash using SHA-256.
    This is NOT a real ECFP4 fingerprint.

    Args:
        smiles: SMILES string.
        n_bits: Number of bits in the output vector (default: 2048).

    Returns:
        numpy float32 array of shape (n_bits,) with values 0.0 or 1.0.
    """
    n_set = n_bits // 8
    seed = int(hashlib.sha256(smiles.encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, n_bits, size=n_set)
    arr = np.zeros(n_bits, dtype=np.float32)
    arr[indices] = 1.0
    return arr
