"""Tier 1: Training functions for molecular viability models.

Contains training loops for ESOL and QM9 datasets, plus synthetic
data training helpers used in demo/fallback modes.

References:
    Delaney, S. J. "ESOL: Estimating Aqueous Solubility
    Directly from Structure." J. Chem. Inf. Model. 2004.
    Ramakrishnan, R. et al. "QM9: 134 Kilo Molecules."
    Sci. Data 2014.
"""

from __future__ import annotations

from importlib import resources
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from aurelius.screening.tier1.models import PyTorchBackend
from aurelius.utils.dependencies import HAS_TORCH

if not HAS_TORCH:
    raise ImportError(
        "PyTorch is required for training. Install torch to use model backends."
    )


def _get_fingerprint_fn() -> Any:
    """Lazily import fingerprint generation to avoid circular imports."""
    from aurelius.utils.chem_utils import generate_ecfp4_fingerprint

    return generate_ecfp4_fingerprint


def _generate_synthetic_training_data() -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], list[str]]:
    """Generate synthetic training data for fallback modes.

    Simple molecules are labeled as soluble (1.0),
    complex molecules as insoluble (0.0).

    Returns:
        Tuple of (X_train, y_train, smiles_list).
    """
    import csv

    synthetic_data_path = resources.files("aurelius.data").joinpath("synthetic_training_data.csv")

    training_data: list[tuple[str, float]] = []
    with open(str(synthetic_data_path)) as f:
        reader = csv.DictReader(f)
        for row in reader:
            training_data.append((row["smiles"], float(row["label"])))

    generate_fp = _get_fingerprint_fn()
    X_train = np.zeros((len(training_data), 2048), dtype=np.float32)
    y_train = np.zeros(len(training_data), dtype=np.float32)

    smiles_list = []
    for i, (smiles, label) in enumerate(training_data):
        fp = generate_fp(smiles)
        X_train[i] = fp
        y_train[i] = label
        smiles_list.append(smiles)

    return X_train, y_train, smiles_list


def train_on_esol(
    model: PyTorchBackend | None = None,
    epochs: int = 200,
    lr: float = 0.005,
    batch_size: int = 16,
    seed: int = 42,
    val_split: float = 0.15,
) -> PyTorchBackend:
    """Train the PyTorch model on the ESOL dataset (Delaney et al. 2004).

    The ESOL (Estimated SOLubility) dataset contains 1112 molecules
    with experimentally measured aqueous solubility (logS in mol/L).
    This is a standard benchmark for molecular property prediction.

    Args:
        epochs: Maximum number of training epochs.
        lr: Learning rate for gradient descent.
        batch_size: Mini-batch size.
        seed: Random seed for reproducibility.
        val_split: Fraction of data held out for validation.

    Returns:
        The trained PyTorchBackend instance.

    Raises:
        RuntimeError: If RDKit is unavailable.
    """
    generate_fp = _get_fingerprint_fn()

    # Load ESOL dataset via huggingface datasets library
    try:
        from datasets import load_dataset

        _ds = load_dataset("deepchem/esol", split="train")
        training_data = [(sm, float(v)) for sm, v in zip(_ds["smiles"], _ds["logS"], strict=True)]
    except (ImportError, ValueError, ConnectionError, OSError):
        # Fallback to packaged CSV resource
        training_data_path = resources.files("aurelius.data").joinpath("esol_fallback.csv")
        print(
            f"[tier1] 'datasets' library not available or dataset unavailable, loading from packaged CSV: {training_data_path}"
        )
        print("[Aurelius v9.0 Tier1] Using packaged ESOL fallback data (50 molecules from Delaney 2004)")

        import csv

        with open(str(training_data_path)) as f:
            reader = csv.DictReader(f)
            training_data = [(row["smiles"], float(row["logS"])) for row in reader]

    X_train = np.zeros((len(training_data), 2048), dtype=np.float32)
    y_train = np.zeros(len(training_data), dtype=np.float32)

    for i, (smiles, log_s) in enumerate(training_data):
        fp = generate_fp(smiles)
        X_train[i] = fp
        normalized = (log_s + 6.0) / 7.0
        y_train[i] = np.clip(normalized, 0.0, 1.0)

    n_samples = X_train.shape[0]

    # Split into train/validation
    n_val = int(n_samples * val_split)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_samples)
    X_train_split = X_train[perm[: n_samples - n_val]]
    y_train_split = y_train[perm[: n_samples - n_val]]
    X_val_split = X_train[perm[n_samples - n_val :]]
    y_val_split = y_train[perm[n_samples - n_val :]]

    model = PyTorchBackend()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience = 30
    patience_counter = 0
    best_state: dict[str, Any] | None = None

    for epoch in range(epochs):
        epoch_perm = rng.permutation(n_samples - n_val)
        X_shuffled = X_train_split[epoch_perm]

        for start in range(0, n_samples - n_val, batch_size):
            end = min(start + batch_size, n_samples - n_val)
            x_batch = torch.from_numpy(X_shuffled[start:end]).float()
            y_batch = torch.from_numpy(y_train_split[start:end]).float()

            optimizer.zero_grad()
            pred = model(x_batch).squeeze(-1)  # type: ignore[attr-defined]
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            val_pred = model(torch.from_numpy(X_val_split).float()).squeeze(-1)  # type: ignore[attr-defined]
            val_loss = criterion(val_pred, torch.from_numpy(y_val_split).float()).item()

        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                train_pred = model(torch.from_numpy(X_train_split).float()).squeeze(-1)  # type: ignore[attr-defined]
                train_loss = criterion(train_pred, torch.from_numpy(y_train_split).float()).item()
            print(
                f"[Aurelius v9.0 Tier1] Epoch {epoch + 1}/{epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}  # type: ignore[attr-defined]
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Aurelius v9.0 Tier1] Early stopping at epoch {epoch + 1} (best val_loss={best_val_loss:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)  # type: ignore[attr-defined]

    return model


def train_on_qm9(
    epochs: int = 300,
    lr: float = 0.005,
    batch_size: int = 32,
    seed: int = 42,
) -> PyTorchBackend:
    """Train the PyTorch model on the QM9 dataset (Ramakrishnan et al. 2014).

    The QM9 dataset contains 130,837 small molecules with DFT-computed
    quantum mechanical properties. This function trains on the
    atomization energy (U0) property.

    Args:
        epochs: Number of training epochs.
        lr: Learning rate for gradient descent.
        batch_size: Mini-batch size.
        seed: Random seed for reproducibility.

    Returns:
        The trained PyTorchBackend instance.

    Raises:
        ValueError: If QM9 U0 has zero range after filtering.
        ConnectionError: If network error loading QM9 dataset.
    """
    generate_fp = _get_fingerprint_fn()

    from datasets import load_dataset

    ds = load_dataset("maastrichtuniversity/qm9", split="train")

    u0_values = np.array(ds["U0"], dtype=np.float32)
    smiles_list = ds["smiles"]

    valid_mask = ~np.isnan(u0_values)
    valid_smiles = [s for i, s in enumerate(smiles_list) if valid_mask[i]]
    valid_u0 = u0_values[valid_mask]

    u0_min, u0_max = float(np.min(valid_u0)), float(np.max(valid_u0))
    u0_range = u0_max - u0_min
    if u0_range == 0:
        raise ValueError("QM9 U0 has zero range after filtering")

    n_samples = len(valid_smiles)
    X_train = np.zeros((n_samples, 2048), dtype=np.float32)
    y_train = np.zeros(n_samples, dtype=np.float32)

    for i, smiles in enumerate(valid_smiles):
        fp = generate_fp(smiles)
        X_train[i] = fp
        y_train[i] = np.clip((valid_u0[i] - u0_min) / u0_range, 0.0, 1.0)

    model = PyTorchBackend()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience = 30
    patience_counter = 0
    best_state: dict[str, Any] | None = None

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
            pred = model(x_batch).squeeze(-1)  # type: ignore[attr-defined]
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            val_pred = model(torch.from_numpy(X_train).float()).squeeze(-1)  # type: ignore[attr-defined]
            val_loss = criterion(val_pred, torch.from_numpy(y_train).float()).item()

        if (epoch + 1) % 50 == 0:
            print(f"[Aurelius v9.0 Tier1] QM9 training epoch {epoch + 1}/{epochs}: loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}  # type: ignore[attr-defined]
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Aurelius v9.0 Tier1] Early stopping at epoch {epoch + 1} (best val_loss={best_val_loss:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)  # type: ignore[attr-defined]

    return model


def _train_synthetic_pytorch() -> PyTorchBackend:
    """Train PyTorch fallback on synthetic solubility dataset.

    This provides a trained PyTorch model when MLX is unavailable,
    ensuring the pipeline can run on Linux/Windows/CPU-only systems.

    Returns:
        The trained PyTorchBackend instance.
    """
    X_train, y_train, _ = _generate_synthetic_training_data()

    n_samples = X_train.shape[0]
    n_val = int(n_samples * 0.15)

    rng = np.random.RandomState(42)
    perm = rng.permutation(n_samples)
    X_train_split = X_train[perm[: n_samples - n_val]]
    y_train_split = y_train[perm[: n_samples - n_val]]
    X_val_split = X_train[perm[n_samples - n_val :]]
    y_val_split = y_train[perm[n_samples - n_val :]]

    model = PyTorchBackend()
    criterion = _torch_nn.MSELoss()
    optimizer = _torch.optim.Adam(model.parameters(), lr=0.01)

    best_val_loss = float("inf")
    best_state: dict[str, Any] = {}
    patience = 20
    patience_counter = 0

    for epoch in range(100):
        epoch_perm = rng.permutation(n_samples - n_val)
        X_shuffled = X_train_split[epoch_perm]
        y_shuffled = y_train_split[epoch_perm]

        for start in range(0, n_samples - n_val, 16):
            end = min(start + 16, n_samples - n_val)
            x_batch = _torch.from_numpy(X_shuffled[start:end]).float()
            y_batch = _torch.from_numpy(y_shuffled[start:end]).float()

            optimizer.zero_grad()
            pred = model(x_batch).squeeze(-1)  # type: ignore[attr-defined]
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

        with _torch.no_grad():
            val_pred = model(_torch.from_numpy(X_val_split).float()).squeeze(-1)  # type: ignore[attr-defined]
            val_loss = criterion(val_pred, _torch.from_numpy(y_val_split).float()).item()

        if (epoch + 1) % 20 == 0:
            with _torch.no_grad():
                train_pred = model(_torch.from_numpy(X_train_split).float()).squeeze(-1)  # type: ignore[attr-defined]
                train_loss = criterion(train_pred, _torch.from_numpy(y_train_split).float()).item()
            print(
                f"[Aurelius v9.0 Tier1] PyTorch synthetic epoch {epoch + 1}/100: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"[Aurelius v9.0 Tier1] PyTorch early stopping at epoch {epoch + 1} "
                    f"(best val_loss={best_val_loss:.4f})"
                )
                break

    if best_state:
        model.load_state_dict(best_state)  # type: ignore[attr-defined]

    return model


def _train_synthetic_pytorch() -> PyTorchBackend:
    """Train PyTorch fallback on synthetic solubility dataset.

    This provides a trained PyTorch model when MLX is unavailable,
    ensuring the pipeline can run on Linux/Windows/CPU-only systems.

    Returns:
        The trained PyTorchBackend instance.
    """
    X_train, y_train, _ = _generate_synthetic_training_data()

    n_samples = X_train.shape[0]
    n_val = int(n_samples * 0.15)

    rng = np.random.RandomState(42)
    perm = rng.permutation(n_samples)
    X_train_split = X_train[perm[: n_samples - n_val]]
    y_train_split = y_train[perm[: n_samples - n_val]]
    X_val_split = X_train[perm[n_samples - n_val :]]
    y_val_split = y_train[perm[n_samples - n_val :]]

    model = PyTorchBackend()
    criterion = _torch_nn.MSELoss()
    optimizer = _torch.optim.Adam(model.parameters(), lr=0.01)

    best_val_loss = float("inf")
    best_state: dict[str, Any] = {}
    patience = 20
    patience_counter = 0

    for epoch in range(100):
        epoch_perm = rng.permutation(n_samples - n_val)
        X_shuffled = X_train_split[epoch_perm]
        y_shuffled = y_train_split[epoch_perm]

        for start in range(0, n_samples - n_val, 16):
            end = min(start + 16, n_samples - n_val)
            x_batch = _torch.from_numpy(X_shuffled[start:end]).float()
            y_batch = _torch.from_numpy(y_shuffled[start:end]).float()

            optimizer.zero_grad()
            pred = model(x_batch).squeeze(-1)  # type: ignore[attr-defined]
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

        with _torch.no_grad():
            val_pred = model(_torch.from_numpy(X_val_split).float()).squeeze(-1)  # type: ignore[attr-defined]
            val_loss = criterion(val_pred, _torch.from_numpy(y_val_split).float()).item()

        if (epoch + 1) % 20 == 0:
            with _torch.no_grad():
                train_pred = model(_torch.from_numpy(X_train_split).float()).squeeze(-1)  # type: ignore[attr-defined]
                train_loss = criterion(train_pred, _torch.from_numpy(y_train_split).float()).item()
            print(
                f"[Aurelius v9.0 Tier1] PyTorch synthetic epoch {epoch + 1}/100: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"[Aurelius v9.0 Tier1] PyTorch early stopping at epoch {epoch + 1} "
                    f"(best val_loss={best_val_loss:.4f})"
                )
                break

    if best_state:
        model.load_state_dict(best_state)  # type: ignore[attr-defined]

    return model


__all__ = [
    "train_on_esol",
    "train_on_qm9",
]
