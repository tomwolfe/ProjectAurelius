#!/usr/bin/env python3
"""Train Tier 1 model on QM9 dataset.

This script trains the Tier 1 MLP on quantum mechanical data (QM9)
and saves the trained model weights for use by the Aurelius screening
pipeline.

Usage:
    python scripts/train_tier1.py --epochs 300

References:
    QM9: Ramakrishnan, R. et al. "Quantum Chemistry Structures and
          Properties of 134 Kilo Molecules." Sci. Data 2014, 1, 140035.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

# Reuse dataset loading logic from download_data to avoid duplication
from aurelius.cli_scripts.download_data import load_qm9_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Tier 1 MLP on QM9 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_tier1.py --epochs 300 --batch-size 32
        """,
    )
    parser.add_argument(
        "--dataset",
        choices=["qm9"],
        default="qm9",
        help="Dataset to train on",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs (default: 300)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size (default: 32)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.005,
        help="Learning rate (default: 0.005)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Validation split fraction (default: 0.15)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save trained model (default: models/tier1/<dataset>)",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Path to local QM9 CSV file (bypasses HuggingFace download)",
    )
    return parser.parse_args()


def train_torch(
    X_train: np.ndarray[Any, Any],
    y_train: np.ndarray[Any, Any],
    X_val: np.ndarray[Any, Any],
    y_val: np.ndarray[Any, Any],
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    """Train MLP using PyTorch (CPU/MPS).

    Args:
        X_train: Training fingerprints (N, 2048).
        y_train: Training labels (N,).
        X_val: Validation fingerprints (M, 2048).
        y_val: Validation labels (M,).
        epochs: Number of epochs.
        lr: Learning rate.
        batch_size: Batch size.
        seed: Random seed.

    Returns:
        Dictionary with trained weights and training history.
    """
    import torch
    import torch.nn as nn

    model = nn.Sequential(  # type: ignore[attr-defined]
        nn.Linear(2048, 128),  # type: ignore[attr-defined]
        nn.ReLU(),  # type: ignore[attr-defined]
        nn.Linear(128, 1),  # type: ignore[attr-defined]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n_samples = X_train.shape[0]
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_weights = None
    patience = 30
    patience_counter = 0
    rng = np.random.RandomState(seed)

    for epoch in range(epochs):
        perm = rng.permutation(n_samples)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            x_batch = torch.from_numpy(X_shuffled[start:end]).float()
            y_batch = torch.from_numpy(y_shuffled[start:end]).float()

            optimizer.zero_grad()
            loss = criterion(x_batch, y_batch)
            loss.backward()
            optimizer.step()

        val_loss = criterion(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).float().reshape(-1, 1),
        ).item()
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 20 == 0:
            train_loss = criterion(
                torch.from_numpy(X_train).float(),
                torch.from_numpy(y_train).float().reshape(-1, 1),
            ).item()
            history["train_loss"].append(train_loss)
            print(f"[train_tier1] Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = {k: np.asarray(v) for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[train_tier1] Early stopping at epoch {epoch + 1}")
                break

    return {"weights": best_weights, "history": history}


def save_model(
    weights: dict[str, Any],
    save_path: str,
    dataset: str,
    hyperparams: dict[str, Any],
) -> None:
    """Save trained model weights and metadata.

    Args:
        weights: Dictionary of trained weight arrays.
        save_path: Directory to save model.
        dataset: Dataset name (qm9).
        hyperparams: Training hyperparameters.
    """
    os.makedirs(save_path, exist_ok=True)

    # Save weights as numpy files
    for name, array in weights.items():
        np.save(os.path.join(save_path, f"{name}.npy"), array)

    # Save metadata
    metadata = {
        "dataset": dataset,
        "architecture": "MLP-2048-128-1",
        "fp_type": "ECFP4_2048",
        "hyperparameters": hyperparams,
        "trained_at": Path(__file__).parent.parent.as_posix(),
    }
    with open(os.path.join(save_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[train_tier1] Model saved to: {save_path}")


def train_main(
    dataset: str = "qm9",
    epochs: int = 300,
    batch_size: int = 32,
    learning_rate: float = 0.005,
    csv_path: str | None = None,
    seed: int = 42,
    val_split: float = 0.15,
    save_path: str | None = None,
) -> dict[str, Any]:
    """Train Tier 1 model on QM9 dataset.

    This is the programmatic entry point for training, callable
    directly from other modules without CLI argument parsing.

    Args:
        dataset: Dataset to train on ("qm9").
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Learning rate for optimization.
        csv_path: Optional path to local CSV file (bypasses HuggingFace).
        seed: Random seed for reproducibility.
        val_split: Fraction of data held out for validation.
        save_path: Optional path to save the trained model.


    Returns:
        Dictionary with training results and metadata.
    """
    print(f"[train_tier1] Training on QM9 dataset ({epochs} epochs)")
    X, y, smiles_list = load_qm9_data()

    print(f"[train_tier1] Dataset: {len(X)} molecules, {X.shape[1]} features")

    # Train/validation split
    n_val = int(len(X) * val_split)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X))

    X_train, y_train = X[perm[: len(X) - n_val]], y[perm[: len(X) - n_val]]
    X_val, y_val = X[perm[len(X) - n_val :]], y[perm[len(X) - n_val :]]

    print(f"[train_tier1] Train: {len(X_train)}, Val: {len(X_val)}")

    # Train model
    print("[train_tier1] Training with PyTorch (CPU/MPS)")
    result = train_torch(X_train, y_train, X_val, y_val, epochs, learning_rate, batch_size, seed)

    # Print final metrics
    print("\n[train_tier1] Training complete!")
    print(f"  Final val_loss: {result['history']['val_loss'][-1]:.4f}")
    if result["history"]["train_loss"]:
        print(f"  Final train_loss: {result['history']['train_loss'][-1]:.4f}")

    # Save model
    save_path = save_path or os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models",
        "tier1",
        f"{dataset}_tier1",
    )
    hyperparams = {
        "dataset": dataset,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "val_split": val_split,
        "seed": seed,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_samples": len(X),
    }
    save_model(result["weights"], save_path, dataset, hyperparams)

    print("\n[train_tier1] Ready to use with Aurelius pipeline!")
    print(f"  Set AURELIUS_MODEL_DIR to: {os.path.dirname(save_path)}")

    return {"result": result, "save_path": save_path, "hyperparams": hyperparams}


def main() -> None:
    args = parse_args()
    train_main(
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        csv_path=args.csv_path,
        seed=args.seed,
        val_split=args.val_split,
        save_path=args.save_path,
    )
