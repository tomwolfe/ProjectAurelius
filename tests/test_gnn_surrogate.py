"""Tests verifying GNN surrogate removal in v10.1.

The GNNQuantumOracle class was removed because no ONNX model was
available. PropertyOracle now uses a two-tier architecture
(xTB → TOM) instead of three-tier (xTB → GNN → TOM).
"""

from __future__ import annotations


def test_gnn_module_stub_exists() -> None:
    """The gnn_surrogate module should still exist as a stub."""
    import aurelius.scoring.oracle.gnn_surrogate as gnn_mod
    assert gnn_mod.__doc__ is not None


def test_gnn_not_in_property_oracle() -> None:
    """PropertyOracle should no longer have _gnn or use_gnn attributes."""
    from aurelius.scoring.oracle import PropertyOracle

    oracle = PropertyOracle(use_xtb=False)
    assert not hasattr(oracle, "_gnn"), "GNN surrogate should be removed"
    assert not hasattr(oracle, "_use_gnn"), "use_gnn should be removed"

    # Should still evaluate via TOM fallback
    from aurelius.types import MoleculeContext
    ctx = MoleculeContext.from_smiles("CCO")
    assert ctx is not None
    result = oracle.evaluate(ctx)
    assert "homo_eV" in result
    assert "lumo_eV" in result
    assert result["homo_eV"] < result["lumo_eV"]
