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
from pathlib import Path


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

    try:
        from datasets import load_dataset
    except ImportError:
        print("[download_data] ERROR: 'datasets' library not installed.")
        print("[download_data] Install with: pip install datasets")
        sys.exit(1)

    output_path = os.path.join(output_dir, "esol")
    os.makedirs(output_path, exist_ok=True)

    ds = load_dataset("matin/dehesa", split="train")
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

    try:
        from datasets import load_dataset
    except ImportError:
        print("[download_data] ERROR: 'datasets' library not installed.")
        print("[download_data] Install with: pip install datasets")
        sys.exit(1)

    output_path = os.path.join(output_dir, "qm9")
    os.makedirs(output_path, exist_ok=True)

    ds = load_dataset("matin/qm9", split="train")
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
    print(f"[download_data] Train with: python scripts/train_tier1.py --dataset {datasets_to_download[0]} --csv-path {output_dir}/{datasets_to_download[0]}/{datasets_to_download[0]}.csv")


if __name__ == "__main__":
    main()
