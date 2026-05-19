#!/usr/bin/env python3
"""Download and prepare datasets for Aurelius Tier 1 training.

Fetches real molecular datasets from Hugging Face Hub for training
the Tier 1 screening model with scientifically valid data.

Usage:
    python scripts/download_data.py --dataset esol
    python scripts/download_data.py --dataset qm9
    python scripts/download_data.py --dataset all --output ./data/

References:
    ESOL: Delaney, S. J. "ESOL: Estimating Aqueous Solubility
          Directly from Structure." J. Chem. Inf. Model. 2004.
    QM9: Ramakrishnan, R. et al. "QM9: 134 Kilo Molecules."
          Sci. Data 2014.
"""

from __future__ import annotations

from aurelius.cli_scripts.download_data import main as _main

if __name__ == "__main__":
    _main()
