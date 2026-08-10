"""Conductivity proxy must not saturate on real electrolytes (ADR-2026-08-10-04).

The Walden-product proxy previously hard-clamped at 10.0. Once the dielectric
model moved onto the true epsilon scale (Kirkwood-Fröhlich), 15 of 51 known
electrolytes pinned to exactly 10.000 — EC, FEC, PC and sulfolane all became
indistinguishable, which is precisely the region a discovery campaign cares
about. The clamp was replaced by a smooth saturating map.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from aurelius.scoring.oracle.gc import (
    _CONDUCTIVITY_MAX,
    predict_dielectric_proxy,
    predict_ionic_conductivity_proxy,
    predict_ionic_conductivity_proxy_batch,
    predict_li_solvation_proxy,
    predict_viscosity_proxy,
)
from aurelius.types import MoleculeContext

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "src", "aurelius", "data")


@pytest.fixture(scope="module")
def known_properties():
    with open(os.path.join(DATA_DIR, "known_electrolytes.json")) as fh:
        smiles = json.load(fh)
    rows = []
    for smi in smiles:
        ctx = MoleculeContext.from_smiles(smi)
        if ctx is None:
            continue
        rows.append((
            predict_dielectric_proxy(ctx),
            predict_viscosity_proxy(ctx),
            predict_li_solvation_proxy(ctx),
        ))
    return rows


def test_does_not_saturate_on_known_electrolytes(known_properties):
    values = np.array([predict_ionic_conductivity_proxy(*r) for r in known_properties])
    pinned = int(np.sum(values >= _CONDUCTIVITY_MAX - 1e-6))
    assert pinned == 0, f"{pinned}/{len(values)} electrolytes pinned at the ceiling"


def test_retains_resolution_at_the_top(known_properties):
    """The best candidates must still be distinguishable from each other."""
    values = np.array([predict_ionic_conductivity_proxy(*r) for r in known_properties])
    top = np.sort(values)[-10:]
    assert len(set(np.round(top, 3))) >= 8, "top-10 conductivities are nearly tied"


def test_output_stays_in_range(known_properties):
    for row in known_properties:
        value = predict_ionic_conductivity_proxy(*row)
        assert 0.0 <= value < _CONDUCTIVITY_MAX


def test_monotone_in_the_walden_product():
    """The saturating map must never reorder two candidates.

    This is what makes the change safe: no prior ranking claim can be affected
    by replacing the clamp, because rank order is preserved exactly.
    """
    dielectrics = np.linspace(2.0, 120.0, 60)
    values = [predict_ionic_conductivity_proxy(d, 1.5, 3.5) for d in dielectrics]
    assert all(b > a for a, b in zip(values, values[1:], strict=False))

    viscosities = np.linspace(0.3, 20.0, 60)
    values = [predict_ionic_conductivity_proxy(60.0, v, 3.5) for v in viscosities]
    assert all(b < a for a, b in zip(values, values[1:], strict=False))


def test_batch_matches_scalar(known_properties):
    arr = np.array(known_properties, dtype=float)
    scalar = np.array([predict_ionic_conductivity_proxy(*r) for r in known_properties])
    batch = predict_ionic_conductivity_proxy_batch(arr[:, 0], arr[:, 1], arr[:, 2])
    assert np.allclose(scalar, batch)


def test_degenerate_inputs_are_safe():
    assert predict_ionic_conductivity_proxy(0.0, 1.0, 3.5) == 0.0
    assert predict_ionic_conductivity_proxy(50.0, 0.0, 3.5) == 0.0
    assert predict_ionic_conductivity_proxy(-1.0, 1.0, 3.5) == 0.0
