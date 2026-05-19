#!/usr/bin/env python3
"""Train Tier 1 model on real datasets (ESOL, QM9).

This script trains the Tier 1 MLP on experimental solubility data
(ESOL) or quantum mechanical data (QM9) and saves the trained model
weights for use by the Aurelius screening pipeline.

Usage:
    python scripts/train_tier1.py --dataset esol --epochs 200
    python scripts/train_tier1.py --dataset qm9 --epochs 300
    python scripts/train_tier1.py --dataset esol --save-path ./models/tier1/esol_solubility

References:
    ESOL: Delaney, S. J. "ESOL: Estimating Aqueous Solubility
          Directly from Structure." J. Chem. Inf. Model. 2004, 44(6), 1947-1949.
    QM9: Ramakrishnan, R. et al. "Quantum Chemistry Structures and
          Properties of 134 Kilo Molecules." Sci. Data 2014, 1, 140035.
"""

from __future__ import annotations

from aurelius.cli_scripts.train_tier1 import main as _main

if __name__ == "__main__":
    _main()
