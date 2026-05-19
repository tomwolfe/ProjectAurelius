#!/usr/bin/env python3
"""Train Tier 0 MPNN model for activation energy prediction.

Usage:
    python scripts/train_tier0.py [--epochs N] [--batch-size N] [--learning-rate LR]
    python scripts/train_tier0.py --csv-path data/my_data.csv

This script generates a synthetic training dataset (500 molecules) using
RDKit + Arrhenius shifts + Gaussian noise, then trains a lightweight
MPNN model via MSE loss with early stopping.

Model weights are saved to models/tier0/mpnn_weights.pth.
Training data is saved to data/train_tier0_synthetic.csv.
"""

from __future__ import annotations

from aurelius.cli_scripts import train_tier0_main as _main

if __name__ == "__main__":
    _main()
