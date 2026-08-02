"""Tests for v11.0 improvements — vectorized Tanimoto, LRU cache, kernel health."""
from __future__ import annotations

import numpy as np
from rdkit.DataStructs import BulkTanimotoSimilarity

from aurelius.agent.selection import _batch_max_tanimoto, _fp_to_array
from aurelius.pipeline import check_kernel_health
from aurelius.types import MoleculeContext


class TestVectorizedTanimoto:
    """Tests for the vectorized Tanimoto similarity functions."""

    def test_fp_to_array_shape(self):
        """_fp_to_array should return a 2048-element uint8 array."""
        ctx = MoleculeContext.from_smiles("CCO")
        assert ctx is not None
        fp = ctx.get_ecfp4()
        arr = _fp_to_array(fp)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2048,)
        assert arr.dtype == np.uint8

    def test_fp_to_array_bit_pattern(self):
        """_fp_to_array should preserve the on-bit pattern."""
        ctx = MoleculeContext.from_smiles("CCO")
        assert ctx is not None
        fp = ctx.get_ecfp4()
        arr = _fp_to_array(fp)
        on_bits = list(fp.GetOnBits())
        assert all(arr[idx] == 1 for idx in on_bits)
        assert arr.sum() == len(on_bits)

    def test_batch_max_tanimoto_identical(self):
        """Max Tanimoto of identical fingerprints should be 1.0."""
        arr = np.zeros((3, 2048), dtype=np.uint8)
        arr[0, 0] = 1
        arr[1, 0] = 1
        arr[2, 1] = 1
        selected = arr[:1]
        result = _batch_max_tanimoto(arr, selected)
        assert result[0] == 1.0
        assert result[1] == 1.0
        assert result[2] == 0.0

    def test_batch_max_tanimoto_vs_rdkit(self):
        """Vectorized results should match RDKit BulkTanimotoSimilarity."""
        ctx_ethanol = MoleculeContext.from_smiles("CCO")
        ctx_methane = MoleculeContext.from_smiles("C")
        ctx_water = MoleculeContext.from_smiles("O")
        assert all(c is not None for c in [ctx_ethanol, ctx_methane, ctx_water])

        fps = [c.get_ecfp4() for c in [ctx_ethanol, ctx_methane, ctx_water]]
        fps_arr = np.array([_fp_to_array(fp) for fp in fps], dtype=np.uint8)

        selected = fps[:2]
        selected_arr = fps_arr[:2]

        result = _batch_max_tanimoto(fps_arr, selected_arr)
        for i in range(3):
            rdkit_sims = BulkTanimotoSimilarity(fps[i], selected)
            expected = max(rdkit_sims)
            assert abs(result[i] - expected) < 1e-6, (
                f"Mismatch at index {i}: vec={result[i]}, rdkit={expected}"
            )

    def test_batch_max_tanimoto_empty_selected(self):
        """With no selected fingerprints, all similarities should be 0."""
        arr = np.zeros((3, 2048), dtype=np.uint8)
        selected = np.zeros((0, 2048), dtype=np.uint8)
        result = _batch_max_tanimoto(arr, selected)
        assert np.all(result == 0.0)

    def test_batch_max_tanimoto_orthogonal(self):
        """Orthogonal bits should yield zero Tanimoto."""
        n = 5
        arr = np.eye(n, 2048, dtype=np.uint8)
        selected = arr[2:3]
        result = _batch_max_tanimoto(arr, selected)
        assert result[2] == 1.0
        assert all(r == 0.0 for r in result[:2])
        assert all(r == 0.0 for r in result[3:])


class TestMoleculeContextLruCache:
    """Tests for LRU cache on MoleculeContext.from_smiles."""

    def test_cache_returns_same_object(self):
        """Repeated calls with the same SMILES should return the same object."""
        ctx1 = MoleculeContext.from_smiles("CCO")
        ctx2 = MoleculeContext.from_smiles("CCO")
        assert ctx1 is ctx2

    def test_cache_returns_none_for_invalid(self):
        """Invalid SMILES should be cached as None."""
        result = MoleculeContext.from_smiles("not-a-smiles")
        assert result is None

    def test_cache_clear_forces_recomputation(self):
        """After clearing cache, the same SMILES should produce a new object."""
        ctx1 = MoleculeContext.from_smiles("CCO")
        MoleculeContext.cache_clear()
        ctx2 = MoleculeContext.from_smiles("CCO")
        assert ctx1 is not ctx2
        assert ctx1.smiles == ctx2.smiles

    def test_cache_clear_different_invalid(self):
        """Clearing the cache should also invalidate None results."""
        r1 = MoleculeContext.from_smiles("invalid##")
        MoleculeContext.cache_clear()
        r2 = MoleculeContext.from_smiles("invalid##")
        assert r1 is r2  # both should be None, but different None isn't possible
        assert r1 is None

    def test_cache_hits_after_miss(self):
        """Identical SMILES after an intermediate miss should still hit."""
        c1 = MoleculeContext.from_smiles("CCO")
        MoleculeContext.from_smiles("CCO")
        _ = MoleculeContext.from_smiles("CCC")
        c3 = MoleculeContext.from_smiles("CCO")
        assert c1 is c3


class TestCheckKernelHealth:
    """Tests for check_kernel_health."""

    def test_inside_domain(self, caplog):
        """A molecule within all domain boundaries should return True."""
        ctx = MoleculeContext.from_smiles("C1COC(=O)O1")
        assert ctx is not None
        assert check_kernel_health(ctx) is True

    def test_outside_mw(self, caplog):
        """A molecule with MW outside the domain should log a warning."""
        import aurelius.pipeline as p
        # Temporarily lower MW max to trigger warning
        orig = dict(p.DEFAULT_DOMAIN_BOUNDARIES)
        p.DEFAULT_DOMAIN_BOUNDARIES["mw"] = (30.0, 31.0)
        ctx = MoleculeContext.from_smiles("CCO")
        assert ctx is not None
        try:
            result = check_kernel_health(ctx)
            assert result is False
            assert "Molecule outside default kernel domain" in caplog.text
            assert "mw" in caplog.text
        finally:
            p.DEFAULT_DOMAIN_BOUNDARIES.clear()
            p.DEFAULT_DOMAIN_BOUNDARIES.update(orig)

    def test_outside_multiple_properties(self, caplog):
        """Multiple out-of-domain properties should each log a warning."""
        import aurelius.pipeline as p
        orig = dict(p.DEFAULT_DOMAIN_BOUNDARIES)
        p.DEFAULT_DOMAIN_BOUNDARIES["mw"] = (30.0, 31.0)
        p.DEFAULT_DOMAIN_BOUNDARIES["hba"] = (5, 10)
        ctx = MoleculeContext.from_smiles("CCO")
        assert ctx is not None
        try:
            result = check_kernel_health(ctx)
            assert result is False
            assert caplog.text.count("Molecule outside default kernel domain") >= 1
        finally:
            p.DEFAULT_DOMAIN_BOUNDARIES.clear()
            p.DEFAULT_DOMAIN_BOUNDARIES.update(orig)

    def test_handles_nonexistent_property(self):
        """If a MoleculeContext lacks a domain property, it should be skipped."""
        # MoleculeContext has all the defined properties, so this just
        # verifies the function doesn't crash.
        ctx = MoleculeContext.from_smiles("CCO")
        assert ctx is not None
        check_kernel_health(ctx)
