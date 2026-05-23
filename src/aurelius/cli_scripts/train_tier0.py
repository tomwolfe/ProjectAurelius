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

import argparse
import sys
from typing import Any

from aurelius.screening.tier0_gnn import train_tier0_model


def train_main(
    epochs: int = 200,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    csv_path: str | None = None,
) -> dict[str, Any]:
    """Train the Tier 0 MPNN model.

    Args:
        epochs: Maximum number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Learning rate for Adam optimizer.
        csv_path: Optional path to pre-generated CSV training data.

    Returns:
        Dictionary with training metrics.
    """
    results = train_tier0_model(
        n_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        train_csv_path=csv_path,
        output_path="models/tier0/mpnn_weights.pth",
    )
    return results


def main() -> None:
    """CLI entry point for tier0 model training."""
    parser = argparse.ArgumentParser(description="Train Tier 0 MPNN model for activation energy prediction.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs (default: 200)")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size (default: 16)")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--csv-path", type=str, default=None, help="Path to local CSV training data")
    parser.add_argument(
        "--output", type=str, default="models/tier0/mpnn_weights.pth", help="Output path for model weights"
    )

    args = parser.parse_args()

    try:
        results = train_tier0_model(
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            train_csv_path=args.csv_path,
            output_path=args.output,
        )
        print("\nTraining Summary:")
        print(f"  Final train loss: {results['final_train_loss']:.6f}")
        print(f"  Best val loss:    {results['best_val_loss']:.6f}")
        print(f"  Epochs run:       {results['epochs_run']}")
        print(f"  Training samples: {results['n_train']}")
        print(f"  Validation samples: {results['n_val']}")
        print(f"  Weights saved to: {results['weights_path']}")
    except Exception as e:
        print(f"[ERROR] Training failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
