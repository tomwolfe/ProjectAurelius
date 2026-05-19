#!/usr/bin/env python3
"""prep_discovery.py — Automated model preparation pipeline.

Ensures all ML components are trained and ready before the autonomous
screening agent runs.  Checks for existence of trained model weights
and, if missing, triggers automated training using existing modules:

    Tier 1 (ESOL/QM9 MLP)                → models/tier1/esol_solubility/
    Tier 0 (MPNN activation energy)      → models/tier0/mpnn_weights.pth

Validation: after training, loads the saved models and runs a
deterministic inference check on Ethylene Carbonate (O=C1OCCO1) to
verify integrity.

Usage:
    python scripts/prep_discovery.py
    python scripts/prep_discovery.py --dataset esol --epochs 200
    python scripts/prep_discovery.py --tier0-epochs 200 --tier1-epochs 200

If RDKit is not available the script exits with a clear error message
because all downstream training and inference depends on RDKit.
"""

from __future__ import annotations

from aurelius.cli_scripts.prep_discovery import prep_discovery as _main

if __name__ == "__main__":
    _main()
