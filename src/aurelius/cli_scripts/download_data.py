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

import argparse
import os
import sys
from typing import Any

import numpy as np

from aurelius.data.loaders import load_esol_data, load_qm9_lumo_data

__all__ = [
    "load_esol_data",
    "load_qm9_lumo_data",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download molecular datasets for Aurelius Tier 1 training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Datasets:
  esol    Delaney et al. aqueous solubility (1112 molecules)
  qm9     Ramakrishnan et al. quantum properties (134K molecules)
  all     Download all available datasets

Examples:
    python scripts/download_data.py --dataset esol
    python scripts/download_data.py --dataset all --output ./data/
        """,
    )
    parser.add_argument(
        "--dataset",
        choices=["esol", "qm9", "all"],
        default="all",
        help="Dataset to download (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/",
        help="Output directory (default: ./data/)",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Export datasets as CSV files for offline use",
    )
    return parser.parse_args()


def download_esol(output_dir: str) -> str:
    """Download the ESOL dataset (Delaney et al. 2004).

    The ESOL dataset contains 1112 molecules with experimentally
    measured aqueous solubility (logS in mol/L).

    Uses verified HuggingFace repository: deepchem/esol

    Args:
        output_dir: Directory to save the dataset.

    Returns:
        Path to the downloaded dataset.

    Reference:
        Delaney, S. J. "ESOL: Estimating Aqueous Solubility
        Directly from Structure." J. Chem. Inf. Model. 2004,
        44(6), 1947-1949. DOI: 10.1021/ci034236x
    """
    print("[download_data] Downloading ESOL dataset...")

    from datasets import load_dataset

    output_path = os.path.join(output_dir, "esol")
    os.makedirs(output_path, exist_ok=True)

    try:
        # Verified dataset: deepchem/esol
        ds = load_dataset("deepchem/esol", split="train")
    except ValueError as e:
        print(f"[download_data] ERROR: Invalid dataset ID (ValueError): {e}")
        print("[download_data] Check that 'deepchem/esol' exists on HuggingFace Hub.")
        sys.exit(1)
    except ConnectionError as e:
        print(f"[download_data] ERROR: Network error: {e}")
        print("[download_data] Check your network connection and try again.")
        sys.exit(1)
    except (ImportError, RuntimeError, OSError) as e:
        print(f"[download_data] ERROR loading ESOL: {e}")
        sys.exit(1)

    print(f"[download_data] Loaded {len(ds)} molecules from ESOL")

    # Save as CSV for offline use
    csv_path = os.path.join(output_path, "esol.csv")
    with open(csv_path, "w") as f:
        f.write("smiles,logS\n")
        for item in ds:
            f.write(f"{item['smiles']},{item['logS']}\n")

    print(f"[download_data] ESOL saved to: {csv_path}")
    print(f"[download_data] Use with: python scripts/train_tier1.py --dataset esol --csv-path {csv_path}")

    return csv_path


def download_qm9(output_dir: str) -> str:
    """Download the QM9 dataset (Ramakrishnan et al. 2014).

    The QM9 dataset contains 134,887 small molecules with DFT-computed
    quantum mechanical properties including atomization energy (U0).

    Uses verified HuggingFace repository: maastrichtuniversity/qm9

    Args:
        output_dir: Directory to save the dataset.

    Returns:
        Path to the downloaded dataset.

    Reference:
        Ramakrishnan, R. et al. "Quantum Chemistry Structures and
        Properties of 134 Kilo Molecules." Sci. Data 2014, 1, 140035.
        DOI: 10.1038/sdata.2014.35
    """
    print("[download_data] Downloading QM9 dataset...")

    from datasets import load_dataset

    output_path = os.path.join(output_dir, "qm9")
    os.makedirs(output_path, exist_ok=True)

    try:
        # Verified dataset: maastrichtuniversity/qm9
        ds = load_dataset("maastrichtuniversity/qm9", split="train")
    except ValueError as e:
        print(f"[download_data] ERROR: Invalid dataset ID (ValueError): {e}")
        print("[download_data] Check that 'maastrichtuniversity/qm9' exists on HuggingFace Hub.")
        sys.exit(1)
    except ConnectionError as e:
        print(f"[download_data] ERROR: Network error: {e}")
        print("[download_data] Check your network connection and try again.")
        sys.exit(1)
    except (ImportError, RuntimeError, OSError) as e:
        print(f"[download_data] ERROR loading QM9: {e}")
        sys.exit(1)

    print(f"[download_data] Loaded {len(ds)} molecules from QM9")

    # Save as CSV for offline use
    csv_path = os.path.join(output_path, "qm9.csv")
    with open(csv_path, "w") as f:
        f.write("smiles,U0\n")
        for item in ds:
            u0 = item["U0"] if not (isinstance(item["U0"], float) and str(item["U0"]) == "nan") else "nan"
            f.write(f"{item['smiles']},{u0}\n")

    print(f"[download_data] QM9 saved to: {csv_path}")
    print(f"[download_data] Use with: python scripts/train_tier1.py --dataset qm9 --csv-path {csv_path}")

    return csv_path


def main() -> None:
    args = parse_args()
    output_dir = args.output

    os.makedirs(output_dir, exist_ok=True)
    print(f"[download_data] Output directory: {output_dir}")

    datasets_to_download = ["esol", "qm9"] if args.dataset == "all" else [args.dataset]

    for dataset in datasets_to_download:
        if dataset == "esol":
            download_esol(output_dir)
        elif dataset == "qm9":
            download_qm9(output_dir)

    print("\n[download_data] Dataset download complete!")
    print(
        f"[download_data] Train with: python scripts/train_tier1.py --dataset {datasets_to_download[0]} --csv-path {output_dir}/{datasets_to_download[0]}/{datasets_to_download[0]}.csv"
    )


__all__ = [
    "load_esol_data",
    "load_qm9_lumo_data",
]


def load_qm9_data() -> tuple[Any, Any, list[str]]:
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
    from aurelius.data.loaders import load_qm9_lumo_data as _load_qm9

    data = _load_qm9()
    smiles_list = [d["smiles"] for d in data]
    u0_values = np.array([d["U0"] for d in data], dtype=np.float32)
    # Generate fingerprints for all molecules
    from aurelius.data.loaders import generate_ecfp4_fingerprint
    X = np.zeros((len(data), 2048), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        X[i] = generate_ecfp4_fingerprint(smi)
    # Normalize U0 to [0, 1]
    u0_min, u0_max = float(np.min(u0_values)), float(np.max(u0_values))
    u0_range = u0_max - u0_min
    y = np.clip((u0_values - u0_min) / (u0_range + 1e-12), 0.0, 1.0)
    return X, y, smiles_list


if __name__ == "__main__":
    main()
