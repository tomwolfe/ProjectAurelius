"""Tests for the GC fragment loader in aurelius.scoring.oracle.gc.

Verifies that fragments are correctly loaded from the JSON data file,
compiled into valid RDKit Mol objects, and match expected properties.
"""

from __future__ import annotations

import json

from rdkit import Chem

from aurelius.scoring.oracle.gc import (
    _FRAGMENT_COSTS,
    _FRAGMENTS_DATA,
    _GC_FRAGMENTS,
    _load_fragment_data,
)


def test_fragment_count_matches_json() -> None:
    """Number of loaded fragments must match the JSON source."""
    raw = _load_fragment_data()
    assert len(_GC_FRAGMENTS) == len(raw)
    assert len(_FRAGMENTS_DATA) == len(raw)


def test_all_smarts_compile_to_valid_mol() -> None:
    """Every SMARTS pattern in _GC_FRAGMENTS must compile to a valid Mol."""
    for pattern, name, *_ in _GC_FRAGMENTS:
        assert pattern is not None, f"Failed to compile SMARTS for fragment '{name}'"
        assert pattern.GetNumAtoms() > 0, f"Empty pattern for fragment '{name}'"


def test_fragment_names_match_json() -> None:
    """Fragment names in _GC_FRAGMENTS must correspond to JSON data."""
    raw = _load_fragment_data()
    json_names = {item["name"] for item in raw}
    loaded_names = {name for _, name, *_ in _GC_FRAGMENTS}
    assert loaded_names == json_names, (
        f"Fragment names differ. "
        f"Missing from JSON: {loaded_names - json_names}. "
        f"Missing from loader: {json_names - loaded_names}."
    )


def test_fragment_contributions_type_and_range() -> None:
    """All property contributions must be finite floats within plausible bounds."""
    for _, name, dd, dv, ls, dc in _GC_FRAGMENTS:
        assert isinstance(dd, float), f"dielectric not float for '{name}': {type(dd)}"
        assert isinstance(dv, float), f"viscosity not float for '{name}': {type(dv)}"
        assert isinstance(ls, float), f"li_solvation not float for '{name}': {type(ls)}"
        assert isinstance(dc, float), f"ced not float for '{name}': {type(dc)}"
        assert -10.0 <= dd <= 20.0, f"dielectric out of range for '{name}': {dd}"
        assert -10.0 <= dv <= 20.0, f"viscosity out of range for '{name}': {dv}"
        assert -10.0 <= ls <= 20.0, f"li_solvation out of range for '{name}': {ls}"
        assert -10.0 <= dc <= 20.0, f"ced out of range for '{name}': {dc}"


def test_fragment_costs_loaded() -> None:
    """Every fragment must have an associated cost value."""
    raw = _load_fragment_data()
    assert len(_FRAGMENT_COSTS) == len(raw)
    for item in raw:
        name = str(item["name"])
        assert name in _FRAGMENT_COSTS, f"Missing cost for fragment '{name}'"
        cost = _FRAGMENT_COSTS[name]
        assert isinstance(cost, float), f"Cost not float for '{name}': {type(cost)}"
        assert cost > 0.0, f"Cost must be positive for '{name}': {cost}"


def test_cyclic_carbonate_fragment() -> None:
    """Verify the cyclic_carbonate fragment has expected properties."""
    for pattern, name, dd, dv, ls, dc in _GC_FRAGMENTS:
        if name == "cyclic_carbonate":
            assert dd == 8.0, f"Expected dielectric=8.0, got {dd}"
            assert dv == 0.8, f"Expected viscosity=0.8, got {dv}"
            assert ls == 0.0, f"Expected li_solvation=0.0, got {ls}"
            assert dc == 4.0, f"Expected ced=4.0, got {dc}"
            # Verify it matches ethylene carbonate
            ec = Chem.MolFromSmiles("C1COC(=O)O1")
            assert ec is not None
            matches = ec.GetSubstructMatches(pattern)
            assert len(matches) == 1, (
                f"cyclic_carbonate should match EC once, got {len(matches)} matches"
            )
            return
    assert False, "cyclic_carbonate fragment not found in _GC_FRAGMENTS"


def test_fragments_data_immutable_types() -> None:
    """All JSON fields must have the correct Python types after loading."""
    for item in _FRAGMENTS_DATA:
        assert isinstance(item["smarts"], str)
        assert isinstance(item["name"], str)
        assert isinstance(item["dielectric"], (int, float))
        assert isinstance(item["viscosity"], (int, float))
        assert isinstance(item["li_solvation"], (int, float))
        assert isinstance(item["ced"], (int, float))
