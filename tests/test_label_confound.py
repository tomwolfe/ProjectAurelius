"""Guard against ranking claims made on provenance-confounded targets.

A benchmark can pass every per-entry integrity check and still be unusable as
a *ranking* target, if its labels were compiled from many sources whose
methods disagree. The labels then encode which paper a value came from rather
than what the molecule is.

This is not hypothetical. On the 45 "unseen" LUMO entries of
``external_property_benchmark.json``, a predictor given only the citation
string — no structure at all — scores Spearman rho = 0.84, while no physics
or ML model exceeded 0.09. See ``benchmarks/audit_label_confound.py``.

These tests pin the diagnostic itself (so it cannot silently stop detecting)
and document which targets are currently affected.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks")
)

from audit_label_confound import (  # noqa: E402
    BETWEEN_SOURCE_LIMIT,
    CITATION_RHO_LIMIT,
    analyse,
)

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "aurelius", "data",
)


def _load(name: str) -> list[dict]:
    import json

    with open(os.path.join(DATA_DIR, name)) as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("solvents", [])


class TestDetectorBehaviour:
    """The diagnostic must separate methodology-driven labels from real ones."""

    def test_flags_per_source_offsets(self):
        """Labels that are a per-source constant must be flagged."""
        rng = np.random.default_rng(0)
        entries = []
        for source in range(8):
            offset = rng.normal(0.0, 1.5)
            for _ in range(10):
                entries.append(
                    {
                        "value": float(offset + rng.normal(0.0, 0.1)),
                        "reference": f"paper{source}",
                    }
                )

        result = analyse(entries, "value")
        assert result["confounded"], result
        assert result["citation_rho"] > 0.9
        assert result["between_source_fraction"] > 0.9

    def test_does_not_flag_randomly_attributed_labels(self):
        """Chemistry-driven labels with arbitrary sources must pass."""
        rng = np.random.default_rng(0)
        entries = [
            {
                "value": float(rng.normal(0.0, 1.0)),
                "reference": f"paper{rng.integers(0, 8)}",
            }
            for _ in range(80)
        ]

        result = analyse(entries, "value")
        assert not result["confounded"], result
        assert result["between_source_fraction"] < BETWEEN_SOURCE_LIMIT

    def test_single_source_target_is_reported_as_clean(self):
        """A one-method dataset is the ideal case, not an unanalysable one.

        Regression guard: an early version silently skipped these, which
        removed the only clean control from the report.
        """
        rng = np.random.default_rng(0)
        entries = [
            {"value": float(rng.normal()), "reference": "NIST"} for _ in range(40)
        ]

        result = analyse(entries, "value")
        assert result is not None, "single-source targets must not be skipped"
        assert not result["confounded"]
        assert result["citation_rho"] == 0.0


class TestKnownTargetStatus:
    """Document the current state of the shipped datasets."""

    def test_experimental_ionization_is_clean(self):
        """The LPM's validation target must stay free of the confound.

        The published LPM figure (rho 0.91) is trustworthy *because* it was
        measured on a single-method dataset. If this ever fails, that claim
        needs re-examination.
        """
        result = analyse(_load("experimental_ionization.json"), "ip_eV")
        assert result is not None
        assert not result["confounded"], (
            f"The NIST IP set has become provenance-confounded: {result}. "
            "The LPM accuracy claim rests on this dataset being clean."
        )

    def test_dft_lumo_target_is_confounded(self):
        """Pins the finding that motivated halting the LUMO work.

        If this ever passes, the dataset has been rebuilt with a consistent
        method and a LUMO *ranking* target becomes meaningful again.
        """
        result = analyse(_load("external_property_benchmark.json"), "lumo_eV")
        assert result is not None
        assert result["confounded"], (
            "external_property_benchmark LUMO is no longer confounded — "
            "re-evaluate whether a ranking target is now supportable."
        )
        assert result["citation_rho"] > CITATION_RHO_LIMIT
