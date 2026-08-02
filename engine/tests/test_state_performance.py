"""Tests for LoopState performance — verifying O(1) exact lookup via _fingerprint_dict."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from aurelius.agent.state import LoopState
from aurelius.types import MoleculeContext


class TestStatePerformance:
    """Verify that find_nearest_screened uses O(1) dict lookup for exact SMILES matches."""

    @patch.object(MoleculeContext, 'get_ecfp4', return_value=MagicMock())
    def test_exact_smiles_lookup_is_o1(self, mock_ecfp4) -> None:
        """Adding and then finding an exact SMILES match must complete in O(1) time.

        We verify that _fingerprint_dict is populated and that find_nearest_screened
        returns the cached result for an exact SMILES match without computing Tanimoto.
        """
        state = LoopState(path="/tmp/test_perf.json")
        smi = "CC(=O)OC"
        result = {"score": 85.0, "is_viable": True}
        fp = MagicMock()

        state.add_screened_fingerprint(smi, fp, result)

        # Verify the dict is populated
        assert smi in state._fingerprint_dict

        # Create a context with the same SMILES
        ctx = MoleculeContext.from_smiles(smi)
        assert ctx is not None

        # Mock get_ecfp4 to return our dummy fp
        with patch.object(MoleculeContext, 'get_ecfp4', return_value=fp):
            cached = state.find_nearest_screened(ctx)

        assert cached is not None
        assert cached["score"] == 85.0

    def test_exact_lookup_uses_dict_not_list(self) -> None:
        """When exact SMILES match exists, _fingerprint_dict must be used."""
        state = LoopState(path="/tmp/test_perf2.json")
        smi = "CCO"
        result = {"score": 90.0}
        fp = MagicMock()

        state.add_screened_fingerprint(smi, fp, result)

        # Verify dict has the entry
        assert smi in state._fingerprint_dict
        assert state._fingerprint_dict[smi][1] == result

    def test_duplicate_smiles_updates_dict_only(self) -> None:
        """Adding the same SMILES twice must update the dict, not duplicate in list."""
        state = LoopState(path="/tmp/test_perf3.json")
        smi = "CC(=O)OC"
        result1 = {"score": 80.0}
        result2 = {"score": 85.0}
        fp1 = MagicMock()
        fp2 = MagicMock()

        state.add_screened_fingerprint(smi, fp1, result1)
        state.add_screened_fingerprint(smi, fp2, result2)

        # Dict should have the latest entry
        assert state._fingerprint_dict[smi][1] == result2

        # List should have both entries (for backward compatibility)
        assert len(state.screened_fingerprints) == 2

    @patch.object(MoleculeContext, 'get_ecfp4', return_value=MagicMock())
    def test_find_nearest_exact_match_returns_immediately(self, mock_ecfp4) -> None:
        """Exact SMILES match should return result without computing Tanimoto."""
        state = LoopState(path="/tmp/test_perf4.json")
        smi = "CC(C)OC(C)=O"
        result = {"score": 95.0}
        fp = MagicMock()

        state.add_screened_fingerprint(smi, fp, result)

        ctx = MoleculeContext.from_smiles(smi)
        assert ctx is not None

        with patch.object(MoleculeContext, 'get_ecfp4', return_value=fp):
            cached = state.find_nearest_screened(ctx, threshold=0.95)

        assert cached is not None
        assert cached["score"] == 95.0

    @patch('rdkit.DataStructs.BulkTanimotoSimilarity', return_value=[0.5, 0.6, 0.7])
    @patch.object(MoleculeContext, 'get_ecfp4', return_value=MagicMock())
    def test_nonexistent_smiles_falls_back_to_similarity(self, mock_ecfp4, mock_bulk) -> None:
        """When SMILES not in dict, fallback to Tanimoto similarity search."""
        state = LoopState(path="/tmp/test_perf5.json")
        smi_exact = "CC(=O)OC"
        result = {"score": 85.0}
        fp = MagicMock()

        state.add_screened_fingerprint(smi_exact, fp, result)

        # Query with different SMILES — should not find exact match in dict
        different_smi = "CCO"
        ctx = MoleculeContext.from_smiles(different_smi)
        assert ctx is not None

        # Should return None since max similarity (0.7) < threshold (0.95)
        cached = state.find_nearest_screened(ctx, threshold=0.95)
        assert cached is None

    @patch.object(MoleculeContext, 'get_ecfp4', return_value=MagicMock())
    def test_performance_benchmark_sub_millisecond(self, mock_ecfp4) -> None:
        """Exact SMILES lookup must complete in < 1ms (measured with time)."""
        import time

        state = LoopState(path="/tmp/test_perf6.json")
        smi = "CC(C)OC(C)=O"
        result = {"score": 88.0}
        fp = MagicMock()

        state.add_screened_fingerprint(smi, fp, result)

        ctx = MoleculeContext.from_smiles(smi)
        assert ctx is not None

        with patch.object(MoleculeContext, 'get_ecfp4', return_value=fp):
            start = time.perf_counter()
            cached = state.find_nearest_screened(ctx)
            elapsed_ms = (time.perf_counter() - start) * 1000

        assert cached is not None
        assert elapsed_ms < 1.0, f"Exact lookup took {elapsed_ms:.3f}ms, expected < 1ms"
