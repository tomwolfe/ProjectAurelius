"""Tests for the GNNQuantumOracle surrogate interface.

Verifies that the GNN surrogate gracefully handles missing model files
and missing onnxruntime, and returns (None, None) when unavailable.
Also verifies the compute_penalty interface matches SurrogateQuantumOracle.
"""

from __future__ import annotations

from aurelius.scoring.oracle.gnn_surrogate import GNNQuantumOracle
from aurelius.types import MoleculeContext


def test_gnn_unavailable_by_default() -> None:
    """GNN surrogate should not be available without a model file."""
    gnn = GNNQuantumOracle()
    assert gnn.is_available is False


def test_gnn_predict_returns_none_when_unavailable() -> None:
    """predict() should return (None, None) when model is not loaded."""
    gnn = GNNQuantumOracle()
    ctx = MoleculeContext.from_smiles("CCO")
    assert ctx is not None
    homo, lumo = gnn.predict(ctx)
    assert homo is None
    assert lumo is None


def test_gnn_predict_with_nonexistent_path() -> None:
    """A non-existent model path should not crash."""
    gnn = GNNQuantumOracle(model_path="/tmp/nonexistent_model.onnx")
    assert gnn.is_available is False


def test_gnn_compute_penalty_above_threshold() -> None:
    """compute_penalty should return 0.5 for HOMO > -5.0 eV."""
    gnn = GNNQuantumOracle()
    penalty = gnn.compute_penalty(-4.5)
    assert penalty == 0.5


def test_gnn_compute_penalty_below_threshold() -> None:
    """compute_penalty should return 1.0 for HOMO <= -5.0 eV."""
    gnn = GNNQuantumOracle()
    penalty = gnn.compute_penalty(-6.0)
    assert penalty == 1.0


def test_gnn_compute_penalty_none() -> None:
    """compute_penalty should return 1.0 when HOMO is None."""
    gnn = GNNQuantumOracle()
    penalty = gnn.compute_penalty(None)
    assert penalty == 1.0


def test_gnn_in_property_oracle_fallback() -> None:
    """PropertyOracle should gracefully handle GNN surrogate being unavailable."""
    from aurelius.scoring.oracle import PropertyOracle

    oracle = PropertyOracle(use_xtb=False, use_gnn=True)
    assert oracle._gnn is not None
    assert oracle._gnn.is_available is False

    # Should still evaluate via TOM fallback
    ctx = MoleculeContext.from_smiles("CCO")
    assert ctx is not None
    result = oracle.evaluate(ctx)
    assert "homo_eV" in result
    assert "lumo_eV" in result
    assert result["homo_eV"] < result["lumo_eV"]

