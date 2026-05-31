"""Data loaders for molecular datasets.

Provides a single function to load QM9 HOMO/LUMO data from the bundled
CSV file using pandas, with no external HuggingFace dependency required.
"""

from __future__ import annotations

import logging
import os

import pandas as pd


def load_qm9_homo_lumo_data(csv_path: str | None = None) -> list[tuple[str, float, float]]:
    """Load QM9 HOMO/LUMO data from the bundled CSV file.

    Args:
        csv_path: Optional path to a local CSV file. If None, loads the
            bundled ``qm9_subset.csv``.

    Returns:
        List of (smiles, homo_eV, lumo_eV) tuples.

    Raises:
        RuntimeError: If the CSV file cannot be found or contains no data.
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "qm9_subset.csv")
    csv_path = os.path.abspath(csv_path)
    if not os.path.exists(csv_path):
        raise RuntimeError(f"QM9 data CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"smiles", "homo", "lumo"}
    if not required_cols.issubset(df.columns):
        raise RuntimeError(f"CSV missing required columns: {required_cols - set(df.columns)}")

    df = df.dropna(subset=["smiles", "homo", "lumo"])
    data: list[tuple[str, float, float]] = []
    for _, row in df.iterrows():
        try:
            data.append((str(row["smiles"]).strip(), float(row["homo"]), float(row["lumo"])))
        except (ValueError, TypeError):
            continue

    if not data:
        raise RuntimeError(f"No valid HOMO/LUMO data found in {csv_path}")

    logger = logging.getLogger(__name__)
    logger.info(
        "load_qm9_homo_lumo_data: loaded %d entries from %s",
        len(data), csv_path,
    )
    return data
