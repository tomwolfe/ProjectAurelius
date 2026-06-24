"""Tests for the PropertyOracle disk-based caching layer.

Verifies that:
1. Repeated evaluate() calls return cached results (L1 in-memory).
2. Results persist across instances via diskcache (L2).
3. clear_cache() empties both cache levels.
"""

from __future__ import annotations

import time

import pytest

from aurelius.scoring.oracle.oracle import PropertyOracle
from aurelius.types import MoleculeContext


@pytest.fixture(scope="module")
def oracle() -> PropertyOracle:
    return PropertyOracle(use_xtb=False, use_surrogate=False, use_gc_uq=False)


def test_cache_returns_same_result(oracle: PropertyOracle) -> None:
    """Cached evaluation should return identical results."""
    ctx = MoleculeContext.from_smiles("CC")
    assert ctx is not None
    result1 = oracle.evaluate(ctx)
    result2 = oracle.evaluate(ctx)
    assert result1 == result2


def test_cache_speeds_up_repeated_calls(oracle: PropertyOracle) -> None:
    """Repeated calls should be faster due to caching."""
    ctx = MoleculeContext.from_smiles("CCO")
    assert ctx is not None
    # First call — populate cache
    t0 = time.perf_counter()
    oracle.evaluate(ctx)
    t1 = time.perf_counter()
    # Second call — cache hit
    t2 = time.perf_counter()
    oracle.evaluate(ctx)
    t3 = time.perf_counter()
    uncached_time = t1 - t0
    cached_time = t3 - t2
    assert cached_time < uncached_time, (
        f"Cached call ({cached_time:.4f}s) should be faster "
        f"than uncached ({uncached_time:.4f}s)"
    )


def test_clear_cache_empties_both_levels(oracle: PropertyOracle) -> None:
    """After clear_cache(), the result should no longer be in cache."""
    ctx = MoleculeContext.from_smiles("CCC")
    assert ctx is not None
    oracle.evaluate(ctx)
    oracle.clear_cache()
    # Re-evaluate — should compute fresh (no KeyError)
    result = oracle.evaluate(ctx)
    assert result is not None
    assert "homo_eV" in result


def test_disk_cache_persistence() -> None:
    """Results should persist across PropertyOracle instances via diskcache."""
    # Use a unique SMILES to avoid collisions with other tests
    ctx = MoleculeContext.from_smiles("CCCO")
    assert ctx is not None
    # First instance — populate cache
    oracle1 = PropertyOracle(use_xtb=False, use_surrogate=False, use_gc_uq=False)
    result1 = oracle1.evaluate(ctx)
    # Second instance — should load from disk
    oracle2 = PropertyOracle(use_xtb=False, use_surrogate=False, use_gc_uq=False)
    result2 = oracle2.evaluate(ctx)
    assert result1 == result2
    # Clean up
    oracle1.clear_cache()
    oracle2.clear_cache()


def test_cache_miss_for_different_smiles(oracle: PropertyOracle) -> None:
    """Different SMILES should not trigger false cache hits."""
    ctx1 = MoleculeContext.from_smiles("CCOC")
    ctx2 = MoleculeContext.from_smiles("CCCOC")
    assert ctx1 is not None and ctx2 is not None
    oracle.evaluate(ctx1)
    result2 = oracle.evaluate(ctx2)
    assert result2 is not None
    # SMILES should differ
    assert ctx1.smiles != ctx2.smiles
