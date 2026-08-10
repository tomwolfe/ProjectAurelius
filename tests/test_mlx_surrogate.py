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
