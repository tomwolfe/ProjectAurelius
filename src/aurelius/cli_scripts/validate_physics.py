#!/usr/bin/env python3
"""Physics validation for a single molecule.

Validates that the physical properties of a molecule are consistent with
known chemistry (e.g., charge balance, energy conservation, etc.).

Usage:
    python validate_physics.py --smiles CC(=O)OC1=CC(=O)O1
"""

from __future__ import annotations

import argparse
import sys

from aurelius.config import get_config
from aurelius.pipeline import AureliusPipeline


def main() -> None:
    """Run physics validation on a molecule."""
    parser = argparse.ArgumentParser(
        description="Run physics validation on a molecule.",
    )
    parser.add_argument("--smiles", default="CC(=O)OC1=CC(=O)O1", help="Molecule to validate")
    args = parser.parse_args()

    config = get_config()
    pipeline = AureliusPipeline(config)
    try:
        pipeline.initialize()
    except Exception as exc:
        print(f"[Aurelius Pipeline] Initialization failed: {exc}")
        print("No score computed.")
        sys.exit(1)

    results = pipeline.screen_molecule(
        args.smiles,
        solvent_type="ec:dmc",
        salt_type="NaPF6",
        ion_type="Na+",
    )

    score = results.get("score") if isinstance(results, dict) else getattr(results, "score", None)
    if score:
        total = score["total_score"] if isinstance(score, dict) else score.total_score
        viable = score["is_viable"] if isinstance(score, dict) else score.is_viable
        print(f"\nAurelius Score v9.0: {total:.1f}/100 {'VIABLE' if viable else 'REJECTED'}")
    else:
        print("No score computed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
