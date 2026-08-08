"""Validation tests for calibrated GC cross-terms.

Verifies that:
1. Cross-terms load from gc_cross_terms.json as valid tuples
2. No coefficient exceeds |2.0|
3. Dielectric rank correlation is non-negative on benchmark data
"""

from __future__ import annotations

import json
import os

import pytest
from scipy.stats import spearmanr

from aurelius.scoring.oracle.gc import _load_cross_terms, predict_dielectric_proxy
from aurelius.types import MoleculeContext

_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "src", "aurelius", "data"
)


@pytest.fixture(scope="module")
def cross_terms_data():
    path = os.path.join(_DATA_DIR, "gc_cross_terms.json")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def benchmark_data():
    path = os.path.join(_DATA_DIR, "external_property_benchmark.json")
    with open(path) as f:
        return json.load(f)


class TestGCCrossTerms:
    def test_cross_terms_loaded_from_json(self):
        terms = _load_cross_terms()
        assert len(terms) == 9, f"Expected 9 cross-terms, got {len(terms)}"
        assert all(isinstance(t, tuple) and len(t) == 4 for t in terms)

    def test_no_coefficient_exceeds_bounds(self, cross_terms_data):
        for entry in cross_terms_data["cross_terms"]:
            assert abs(entry["coefficient"]) <= 2.0, (
                f"Coefficient for {entry['frag_a']}+{entry['frag_b']} "
                f"= {entry['coefficient']} exceeds |2.0|"
            )

    def test_all_terms_have_source_field(self, cross_terms_data):
        for entry in cross_terms_data["cross_terms"]:
            assert entry.get("source") in ("fitted", "default"), (
                f"Coefficient for {entry['frag_a']}+{entry['frag_b']} "
                f"has invalid source: {entry.get('source')}"
            )

    def test_all_nine_pairs_present(self, cross_terms_data):
        pairs = {(t["frag_a"], t["frag_b"]) for t in cross_terms_data["cross_terms"]}
        assert len(pairs) == 9, f"Expected 9 cross-term pairs, got {len(pairs)}"

    def test_dielectric_spearman_nonneg(self, benchmark_data):
        preds = []
        exp = []
        for entry in benchmark_data:
            diel = entry.get("dielectric_constant")
            if diel is None:
                continue
            ctx = MoleculeContext.from_smiles(entry["smiles"])
            if ctx is None:
                continue
            preds.append(predict_dielectric_proxy(ctx))
            exp.append(diel)
        rho, _ = spearmanr(preds, exp)
        assert rho >= 0.0, (
            f"Dielectric Spearman rho = {rho:.4f}, expected >= 0.0"
        )
