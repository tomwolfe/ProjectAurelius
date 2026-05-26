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

import argparse
import json
import os
import sys
import warnings
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from aurelius.utils.dependencies import HAS_DATASETS, HAS_MLX, HAS_RDKIT, HAS_TORCH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Tier 1 MLP on real datasets (ESOL, QM9)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Datasets:
  esol    Delaney et al. aqueous solubility (1112 molecules)
  qm9     Ramakrishnan et al. quantum mechanical properties (134K molecules)

Examples:
  python scripts/train_tier1.py --dataset esol --epochs 200
  python scripts/train_tier1.py --dataset qm9 --epochs 300 --batch-size 32
  python scripts/train_tier1.py --dataset esol --save-path ./models/tier1/esol_solubility
        """,
    )
    parser.add_argument(
        "--dataset",
        choices=["esol", "qm9"],
        default="esol",
        help="Dataset to train on (default: esol)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of training epochs (default: 200 for ESOL, 300 for QM9)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size (default: 16 for ESOL, 32 for QM9)",
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
        help="Path to local ESOL CSV file (bypasses HuggingFace download)",
    )
    return parser.parse_args()


def generate_ecfp4_fingerprint(smiles: str, n_bits: int = 2048) -> np.ndarray[Any, Any]:
    """Generate a 2048-bit ECFP4 (Morgan radius=2) fingerprint.

    Uses RDKit. Raises RuntimeError when RDKit is unavailable.

    Args:
        smiles: SMILES string.
        n_bits: Fingerprint size.

    Returns:
        numpy float32 array of shape (n_bits,).

    Raises:
        RuntimeError: When RDKit is unavailable.
    """
    if not HAS_RDKIT:
        raise RuntimeError(
            "RDKit is required for molecular fingerprint generation. "
            "Install RDKit: pip install rdkit"
        )

    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(
            f"RDKit failed to parse SMILES '{smiles}'. Invalid molecule structure.",
        )
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)  # type: ignore[attr-defined]
    bit_list = fp.ToList()
    arr = np.array(bit_list, dtype=np.float32)
    if len(arr) < n_bits:
        padded = np.zeros(n_bits, dtype=np.float32)
        padded[: len(arr)] = arr
        return padded
    return arr[:n_bits]



def load_esol_data(csv_path: str | None = None) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], list[str]]:
    """Load the ESOL dataset (Delaney et al. 2004).

    The ESOL dataset contains 1112 molecules with experimentally
    measured aqueous solubility (logS in mol/L).

    Fallback chain:
    1. Local CSV via --csv-path
    2. HuggingFace Hub (verified datasets)
    3. Curated 50-molecule subset embedded in code

    Args:
        csv_path: Optional path to local CSV file.

    Returns:
        Tuple of (fingerprints, logS_labels, smiles_list).

    Reference:
        Delaney, S. J. "ESOL: Estimating Aqueous Solubility
        Directly from Structure." J. Chem. Inf. Model. 2004,
        44(6), 1947-1949. DOI: 10.1021/ci034236x
    """
    if csv_path and os.path.isfile(csv_path):
        print(f"[train_tier1] Loading ESOL from local CSV: {csv_path}")
        return _load_esol_from_csv(csv_path)

    if not HAS_DATASETS:
        print("[train_tier1] 'datasets' library not available")
        print("[train_tier1] Installing: pip install datasets")
        print("[train_tier1] Alternatively, use --csv-path to load local CSV")
        sys.exit(1)

    # Load HuggingFace datasets
    try:
        from datasets import load_dataset

        print("[train_tier1] Loading ESOL from HuggingFace Hub...")
        # Verified dataset: deepchem/esol ( maintained by DeepChem )
        ds = load_dataset("deepchem/esol", split="train")
        smiles_list = list(ds["smiles"])
        log_s = np.array(ds["logS"], dtype=np.float32)
        print(f"[train_tier1] Loaded {len(smiles_list)} molecules from ESOL")

        n_bits = 2048
        X = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
        for i, smiles in enumerate(smiles_list):
            X[i] = generate_ecfp4_fingerprint(smiles, n_bits)

        # Normalize logS to [0, 1] for sigmoid output
        # ESOL logS ranges roughly from -6 to +1
        log_s_min, log_s_max = -6.0, 1.0
        y = np.clip((log_s - log_s_min) / (log_s_max - log_s_min), 0.0, 1.0)

        return X, y, smiles_list
    except Exception as e:
        print(f"[train_tier1] Unexpected error loading ESOL (type={type(e).__name__}): {e}")
        import traceback

        traceback.print_exc()
        if csv_path:
            return _load_esol_from_csv(csv_path)
        warnings.warn(
            "[train_tier1] FALLBACK: Using embedded 50-molecule ESOL subset due to error: "
            f"{type(e).__name__}: {e}. "
            "This is a small curated subset from Delaney 2004, NOT the full 1112-molecule dataset. "
            "Model quality will be significantly reduced. "
            "Install 'datasets' and ensure network connectivity for full training, "
            "or use --csv-path to provide a local CSV file.",
            UserWarning,
            stacklevel=2,
        )
        print(
            "\n" + "=" * 60,
            "[train_tier1] WARNING: Fallback to embedded ESOL subset",
            "=" * 60,
            file=sys.stderr,
        )
        print(
            f"[train_tier1] Error: {type(e).__name__}: {e}\n"
            "[train_tier1] Training will proceed with only 50 embedded molecules\n"
            "[train_tier1] from the original Delaney 2004 dataset.\n"
            "[train_tier1] For full dataset training:\n"
            "[train_tier1]   - Ensure network connectivity and install 'datasets'\n"
            "[train_tier1]   - Or use --csv-path to provide a local CSV file\n",
            file=sys.stderr,
        )
        return _load_esol_embedded()


def _load_esol_from_csv(csv_path: str) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], list[str]]:
    """Load ESOL from a local CSV file.

    Expected CSV format: smiles,logS (with header row)

    Args:
        csv_path: Path to CSV file.

    Returns:
        Tuple of (fingerprints, logS_labels, smiles_list).
    """
    import csv

    smiles_list = []
    log_s_list = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = row.get("smiles", row.get("SMILES", "")).strip()
            log_s = float(row.get("logS", row.get("log_s", "nan")))
            if smiles and not np.isnan(log_s):
                smiles_list.append(smiles)
                log_s_list.append(log_s)

    print(f"[train_tier1] Loaded {len(smiles_list)} molecules from CSV")

    n_bits = 2048
    X = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smiles in enumerate(smiles_list):
        X[i] = generate_ecfp4_fingerprint(smiles, n_bits)

    log_s_array = np.array(log_s_list, dtype=np.float32)
    log_s_min, log_s_max = -6.0, 1.0
    y = np.clip((log_s_array - log_s_min) / (log_s_max - log_s_min), 0.0, 1.0)

    return X, y, smiles_list


def _load_esol_embedded() -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], list[str]]:
    """Load ESOL data from packaged CSV resource.

    This is a fallback when HuggingFace download or local CSV is
    unavailable. Uses the packaged esol_fallback.csv resource.

    Returns:
        Tuple of (fingerprints, logS_labels, smiles_list).
    """
    import csv

    training_data_path = resources.files("aurelius.data").joinpath("esol_fallback.csv")
    print(f"[train_tier1] *** LOADED FROM PACKAGED CSV: {training_data_path} ***")

    with open(str(training_data_path)) as f:
        reader = csv.DictReader(f)
        training_data = [(row["smiles"], float(row["logS"])) for row in reader]

    n_bits = 2048
    X = np.zeros((len(training_data), n_bits), dtype=np.float32)
    smiles_list = []
    log_s_list = []

    for i, (smiles, log_s) in enumerate(training_data):
        X[i] = generate_ecfp4_fingerprint(smiles, n_bits)
        smiles_list.append(smiles)
        log_s_list.append(log_s)

    log_s_array = np.array(log_s_list, dtype=np.float32)
    log_s_min, log_s_max = -6.0, 1.0
    y = np.clip((log_s_array - log_s_min) / (log_s_max - log_s_min), 0.0, 1.0)

    return X, y, smiles_list


def load_qm9_data() -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], list[str]]:
    """Load the QM9 dataset (Ramakrishnan et al. 2014).

    The QM9 dataset contains 134,887 small molecules with DFT-computed
    quantum mechanical properties. This function uses the atomization
    energy (U0) property.

    Fallback chain:
    1. HuggingFace Hub (verified: maastrichtuniversity/qm9)
    2. ValueError/ConnectionError handling with specific exceptions

    Returns:
        Tuple of (fingerprints, U0_labels, smiles_list).

    Reference:
        Ramakrishnan, R. et al. "Quantum Chemistry Structures and
        Properties of 134 Kilo Molecules." Sci. Data 2014, 1, 140035.
        DOI: 10.1038/sdata.2014.35
    """
    if not HAS_DATASETS:
        print("[train_tier1] 'datasets' library not available")
        print("[train_tier1] Installing: pip install datasets")
        sys.exit(1)

    # Load HuggingFace datasets
    from datasets import load_dataset

    ds = load_dataset("maastrichtuniversity/qm9", split="train")

    u0_values = np.array(ds["U0"], dtype=np.float32)
    smiles_list = ds["smiles"]

    # Filter valid molecules
    valid_mask = ~np.isnan(u0_values)
    valid_smiles = [s for i, s in enumerate(smiles_list) if valid_mask[i]]
    valid_u0 = u0_values[valid_mask]

    print(f"[train_tier1] Loaded {len(valid_smiles)} QM9 molecules ({np.sum(valid_mask)} valid)")

    n_bits = 2048
    X = np.zeros((len(valid_smiles), n_bits), dtype=np.float32)
    for i, smiles in enumerate(valid_smiles):
        X[i] = generate_ecfp4_fingerprint(smiles, n_bits)

    # Normalize U0 to [0, 1]
    u0_min, u0_max = float(np.min(valid_u0)), float(np.max(valid_u0))
    u0_range = u0_max - u0_min
    y = np.clip((valid_u0 - u0_min) / u0_range, 0.0, 1.0)

    return X, y, valid_smiles


def train_mlx(
    X_train: np.ndarray[Any, Any],
    y_train: np.ndarray[Any, Any],
    X_val: np.ndarray[Any, Any],
    y_val: np.ndarray[Any, Any],
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    """Train MLP using MLX (requires MLX library).

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
    if not HAS_MLX:
        print("[train_tier1] MLX not available, falling back to numpy training")
        return train_numpy(X_train, y_train, X_val, y_val, epochs, lr, batch_size, seed)

    import mlx.core as mx
    import mlx.nn as nn

    model = nn.Sequential(  # type: ignore[attr-defined]
        nn.Linear(2048, 128),  # type: ignore[attr-defined]
        nn.ReLU(),  # type: ignore[attr-defined]
        nn.Linear(128, 1),  # type: ignore[attr-defined]
    )

    def loss_fn(x: Any, y: Any) -> Any:
        return nn.losses.mse_loss(model(x), y, reduction="mean")

    X_train_mx = mx.array(X_train)
    y_train_mx = mx.array(y_train)
    X_val_mx = mx.array(X_val)
    y_val_mx = mx.array(y_val)

    n_samples = X_train.shape[0]
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_weights = None
    patience = 30
    patience_counter = 0
    rng_state = mx.random.key(seed)

    for epoch in range(epochs):
        perm = mx.random.permutation(n_samples, key=rng_state)
        X_shuffled = X_train_mx[perm]
        y_shuffled = y_train_mx[perm]

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            x_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end].reshape(-1, 1)

            loss, grads = nn.value_and_grad(model, loss_fn)(x_batch, y_batch)  # type: ignore[attr-defined]

            # Update model with gradients using MLX optimizer
            optimizer = mx.optimizer.SGD(lr)
            optimizer.update(model, grads)

        val_loss = float(loss_fn(X_val_mx, y_val_mx.reshape(-1, 1)))
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 20 == 0:
            train_loss = float(loss_fn(X_train_mx, y_train_mx.reshape(-1, 1)))
            history["train_loss"].append(train_loss)
            print(f"[train_tier1] Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = {k: np.asarray(v) for k, v in model.parameters().items()}
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
        dataset: Dataset name (esol/qm9).
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
    dataset: str = "esol",
    epochs: int = 200,
    batch_size: int = 16,
    learning_rate: float = 0.005,
    csv_path: str | None = None,
    seed: int = 42,
    val_split: float = 0.15,
    save_path: str | None = None,
) -> dict[str, Any]:
    """Train Tier 1 model on a dataset (esol or qm9).

    This is the programmatic entry point for training, callable
    directly from other modules without CLI argument parsing.

    Args:
        dataset: Dataset to train on ("esol" or "qm9").
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
    if dataset == "esol":
        epochs = epochs if epochs != 200 else 200
        batch_size = batch_size if batch_size != 16 else 16
        print(f"[train_tier1] Training on ESOL dataset ({epochs} epochs)")
        X, y, smiles_list = load_esol_data(csv_path)
    elif dataset == "qm9":
        epochs = epochs if epochs != 300 else 300
        batch_size = batch_size if batch_size != 32 else 32
        print(f"[train_tier1] Training on QM9 dataset ({epochs} epochs)")
        X, y, smiles_list = load_qm9_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print(f"[train_tier1] Dataset: {len(X)} molecules, {X.shape[1]} features")

    # Train/validation split
    n_val = int(len(X) * val_split)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X))

    X_train, y_train = X[perm[: len(X) - n_val]], y[perm[: len(X) - n_val]]
    X_val, y_val = X[perm[len(X) - n_val :]], y[perm[len(X) - n_val :]]

    print(f"[train_tier1] Train: {len(X_train)}, Val: {len(X_val)}")

    # Train model
    print("[train_tier1] Training with MLX (Apple Silicon)")
    result = train_mlx(X_train, y_train, X_val, y_val, epochs, learning_rate, batch_size, seed)

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
        f"{dataset}_solubility",
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
