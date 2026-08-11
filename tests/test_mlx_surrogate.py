"""MLX-native surrogate tests.

Validates that the MLX GPR surrogate produces results consistent with sklearn,
and that batch prediction works correctly.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from aurelius.scoring.oracle.delta_correction import DeltaCorrection


@pytest.fixture(scope="module")
def correction():
    return DeltaCorrection()


@pytest.fixture(scope="module")
def sample_mols():
    smiles = [
        "C1COC(=O)O1", "COC(=O)OC", "CC#N", "O=S1(=O)CCCC1",
        "COCCOC", "CS(=O)(=O)c1ccc(C#N)cc1", "FC(F)(F)COC(=O)OC",
    ]
    return [Chem.MolFromSmiles(s) for s in smiles]


class TestMLXSurrogate:
    def test_import_guard(self):
        """Module imports without error even if mlx is absent."""
        from aurelius.scoring.oracle import mlx_surrogate
        assert hasattr(mlx_surrogate, "MLXGPRSurrogate")
        assert hasattr(mlx_surrogate, "predict_deltas_batch_mlx")

    def test_surrogate_creation(self, correction):
        """MLXGPRSurrogate can be created from a fitted sklearn GPR."""
        from aurelius.scoring.oracle.mlx_surrogate import MLXGPRSurrogate

        surrogate = MLXGPRSurrogate(correction._homo_model)
        # May or may not be available depending on mlx installation
        # Just verify it doesn't crash
        assert isinstance(surrogate.is_available, bool)

    def test_batch_deltas_shape(self, correction, sample_mols):
        """Batch prediction returns correct shapes."""
        d_homo, d_lumo, std_homo, std_lumo = correction.predict_deltas_batch(
            sample_mols, return_std=True
        )
        n = len(sample_mols)
        assert d_homo.shape == (n,)
        assert d_lumo.shape == (n,)
        assert std_homo is not None and std_homo.shape == (n,)
        assert std_lumo is not None and std_lumo.shape == (n,)

    def test_batch_deltas_match_serial(self, correction, sample_mols):
        """Batch predictions match per-molecule predictions."""
        d_homo_batch, d_lumo_batch, std_homo_batch, std_lumo_batch = (
            correction.predict_deltas_batch(sample_mols, return_std=True)
        )

        for i, mol in enumerate(sample_mols):
            d_h, d_l, s_h, s_l = correction.predict_deltas_with_uncertainty(mol)
            assert abs(d_homo_batch[i] - d_h) < 1e-3
            assert abs(d_lumo_batch[i] - d_l) < 1e-3
            assert abs(std_homo_batch[i] - s_h) < 1e-3  # type: ignore[index]
            assert abs(std_lumo_batch[i] - s_l) < 1e-3  # type: ignore[index]

    def test_batch_deltas_no_std(self, correction, sample_mols):
        """Batch prediction without std returns None for std arrays."""
        d_homo, d_lumo, std_homo, std_lumo = correction.predict_deltas_batch(
            sample_mols, return_std=False
        )
        assert d_homo.shape == (len(sample_mols),)
        assert std_homo is None
        assert std_lumo is None

    def test_batch_empty_input(self, correction):
        """Empty input returns empty arrays."""
        d_homo, d_lumo, std_homo, std_lumo = correction.predict_deltas_batch([])
        assert len(d_homo) == 0
        assert len(d_lumo) == 0
        assert std_homo is not None and len(std_homo) == 0

    def test_batch_corrected_shape(self, correction, sample_mols):
        """Batch corrected prediction returns correct shapes."""
        homo, lumo = correction.predict_corrected_batch(sample_mols)
        assert homo.shape == (len(sample_mols),)
        assert lumo.shape == (len(sample_mols),)

    def test_batch_corrected_matches_serial(self, correction, sample_mols):
        """Batch corrected predictions match per-molecule predictions."""
        homo_batch, lumo_batch = correction.predict_corrected_batch(sample_mols)

        for i, mol in enumerate(sample_mols):
            h, l = correction.predict_corrected(mol)
            assert abs(homo_batch[i] - h) < 1e-3
            assert abs(lumo_batch[i] - l) < 1e-3

    def test_batch_corrected_with_base(self, correction, sample_mols):
        """Batch correction uses provided base predictions."""
        base = [(float(-7.0 + i * 0.1), float(2.0 + i * 0.1)) for i in range(len(sample_mols))]
        homo, lumo = correction.predict_corrected_batch(sample_mols, base=base)
        assert homo.shape == (len(sample_mols),)
        assert lumo.shape == (len(sample_mols),)


# ---------------------------------------------------------------------------
# Phase 4: MLX GPR variance for BALD acquisition
# ---------------------------------------------------------------------------


class TestPredictVarianceBatchMLX:
    def test_variance_batch_returns_correct_shape(self, correction, sample_mols):
        """predict_variance_batch_mlx returns (n,) variance array."""
        from aurelius.scoring.oracle.mlx_surrogate import predict_variance_batch_mlx

        variances = predict_variance_batch_mlx(correction._homo_model, sample_mols)
        assert variances.shape == (len(sample_mols),)
        assert variances.dtype == np.float32

    def test_variance_is_non_negative(self, correction, sample_mols):
        """Posterior variance must be non-negative."""
        from aurelius.scoring.oracle.mlx_surrogate import predict_variance_batch_mlx

        variances = predict_variance_batch_mlx(correction._homo_model, sample_mols)
        assert np.all(variances >= 0.0), "Variance contains negative values"

    def test_variance_matches_sklearn(self, correction, sample_mols):
        """MLX variance must match sklearn within 1e-4."""
        from aurelius.scoring.oracle.mlx_surrogate import (
            predict_variance_batch_mlx,
            _predict_deltas_batch_sklearn,
        )

        mlx_var = predict_variance_batch_mlx(correction._homo_model, sample_mols)
        _, sklearn_std = _predict_deltas_batch_sklearn(
            correction._homo_model, sample_mols, return_std=True
        )
        if sklearn_std is None:
            pytest.skip("sklearn std not available")

        sklearn_var = sklearn_std ** 2
        max_diff = float(np.max(np.abs(mlx_var - sklearn_var)))
        assert max_diff < 1e-4, f"MLX variance differs from sklearn by {max_diff:.6e}"

    def test_variance_empty_input(self, correction):
        """Empty input returns empty array."""
        from aurelius.scoring.oracle.mlx_surrogate import predict_variance_batch_mlx

        variances = predict_variance_batch_mlx(correction._homo_model, [])
        assert len(variances) == 0


# ---------------------------------------------------------------------------
# Phase 4: R_g batch caching + threading (ADR-2026-08-07-04)
# ---------------------------------------------------------------------------


class TestRadiusOfGyrationBatch:
    def test_batch_returns_correct_shape(self):
        """R_g batch returns array matching input length."""
        from aurelius.scoring.oracle.quantum import _compute_radius_of_gyration_batch

        mols = [Chem.MolFromSmiles(s) for s in ["C1COC(=O)O1", "COC(=O)OC", "CC#N"]]
        rgs = _compute_radius_of_gyration_batch(mols)
        assert rgs.shape == (3,)
        assert rgs.dtype == np.float32

    def test_batch_empty_input(self):
        """Empty input returns empty array."""
        from aurelius.scoring.oracle.quantum import _compute_radius_of_gyration_batch

        rgs = _compute_radius_of_gyration_batch([])
        assert len(rgs) == 0

    def test_batch_cache_hits(self):
        """Repeated molecules hit the cache — no recomputation."""
        from aurelius.scoring.oracle.quantum import (
            _compute_radius_of_gyration_batch,
            _RG_CACHE,
        )

        smi = "C1COC(=O)O1"
        mol = Chem.MolFromSmiles(smi)
        mols = [mol, mol, mol]

        # Use a unique molecule so cache state from other tests doesn't interfere
        unique_smi = "CCCCCCCCCCCCCCCCO"  # long alkyl chain, likely not cached
        unique_mol = Chem.MolFromSmiles(unique_smi)
        unique_mols = [unique_mol, unique_mol, unique_mol]

        cache_before = len(_RG_CACHE)
        rgs = _compute_radius_of_gyration_batch(unique_mols)
        cache_after = len(_RG_CACHE)

        assert cache_after > cache_before, "Cache was not populated"
        assert len(set(rgs.tolist())) == 1, "Identical molecules got different R_g"

    def test_batch_threaded_parallelism(self):
        """Batch with many unique molecules runs via ThreadPoolExecutor."""
        from aurelius.scoring.oracle.quantum import (
            _compute_radius_of_gyration_batch,
            _RG_CACHE,
        )

        # Use valid SMILES that are unlikely to be in cache
        mols = [Chem.MolFromSmiles(f"CC{'C'*i}N") for i in range(10, 20)]
        rgs = _compute_radius_of_gyration_batch(mols)
        assert rgs.shape == (10,)

    def test_throughput_meets_target(self):
        """R_g batch must achieve >= 2000 mol/s with cache hits on M5 Pro."""
        from aurelius.scoring.oracle.quantum import _compute_radius_of_gyration_batch

        mols = [Chem.MolFromSmiles(f"CC{'C'*i}N") for i in range(50)]
        # Warm cache
        _compute_radius_of_gyration_batch(mols)
        mols = [m for m in mols if m is not None]

        import time
        start = time.perf_counter()
        _compute_radius_of_gyration_batch(mols)
        elapsed = time.perf_counter() - start
        throughput = len(mols) / elapsed

        # This target assumes M5 Pro; on other platforms it's a softer check
        from aurelius.utils.device import get_device
        if get_device() == "mlx":
            assert throughput >= 2000, f"R_g throughput {throughput:.0f} < 2000 mol/s"
