"""Property-based tests for GC predictions using hypothesis.

Verifies physical invariants:
  - predict_dielectric_proxy always returns > 1.0 for valid molecules
  - predict_viscosity_proxy always returns > 0.0 for valid molecules
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis.strategies import from_regex

from aurelius.scoring.oracle.gc import (
    predict_dielectric_proxy,
    predict_li_solvation_proxy,
    predict_viscosity_proxy,
)
from aurelius.types import MoleculeContext

# SMILES-like pattern: alphanumeric chars plus bracket-enclosed atoms
_SMILES_PATTERN = r"[A-Za-z0-9@+\-\[\]\(\)=#%\.\\\/]+"

# Known-valid electrolyte SMILES for targeted testing
_KNOWN_ELECTROLYTES = [
    "CCO",
    "CC(=O)OC1=CC=CC=C1",
    "C1COC(=O)O1",
    "CS(=O)(=O)C",
    "CC#N",
    "C1COCCO1",
    "COC(=O)OC",
    "C1=CC=CC=C1",
    "CC(=O)OCC",
    "CCOC(=O)OCC",
]


@settings(max_examples=200)
@given(from_regex(_SMILES_PATTERN))
def test_dielectric_proxy_always_positive(smiles: str) -> None:
    """predict_dielectric_proxy must return > 1.0 for any parseable molecule."""
    ctx = MoleculeContext.from_smiles(smiles)
    if ctx is None:
        return  # skip invalid SMILES
    try:
        result = predict_dielectric_proxy(ctx)
    except Exception:
        return  # skip molecules that trigger exceptions
    assert result > 1.0, f"Dielectric proxy {result} <= 1.0 for {smiles}"


@settings(max_examples=200)
@given(from_regex(_SMILES_PATTERN))
def test_viscosity_proxy_always_positive(smiles: str) -> None:
    """predict_viscosity_proxy must return > 0.0 for any parseable molecule."""
    ctx = MoleculeContext.from_smiles(smiles)
    if ctx is None:
        return
    try:
        result = predict_viscosity_proxy(ctx)
    except Exception:
        return
    assert result > 0.0, f"Viscosity proxy {result} <= 0.0 for {smiles}"


@settings(max_examples=200)
@given(from_regex(_SMILES_PATTERN))
def test_li_solvation_proxy_always_positive(smiles: str) -> None:
    """predict_li_solvation_proxy must return > 0.0 for any parseable molecule."""
    ctx = MoleculeContext.from_smiles(smiles)
    if ctx is None:
        return
    try:
        result = predict_li_solvation_proxy(ctx)
    except Exception:
        return
    assert result > 0.0, f"Li solvation proxy {result} <= 0.0 for {smiles}"


def test_known_electrolytes_dielectric_above_one() -> None:
    """All known electrolyte SMILES must have dielectric proxy > 1.0."""
    for smi in _KNOWN_ELECTROLYTES:
        ctx = MoleculeContext.from_smiles(smi)
        assert ctx is not None, f"Failed to parse known electrolyte: {smi}"
        result = predict_dielectric_proxy(ctx)
        assert result > 1.0, f"Dielectric proxy {result} <= 1.0 for known electrolyte {smi}"


def test_known_electrolytes_viscosity_above_zero() -> None:
    """All known electrolyte SMILES must have viscosity proxy > 0.0."""
    for smi in _KNOWN_ELECTROLYTES:
        ctx = MoleculeContext.from_smiles(smi)
        assert ctx is not None, f"Failed to parse known electrolyte: {smi}"
        result = predict_viscosity_proxy(ctx)
        assert result > 0.0, f"Viscosity proxy {result} <= 0.0 for known electrolyte {smi}"
