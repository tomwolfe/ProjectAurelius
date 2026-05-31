#!/usr/bin/env python3
"""Download and prepare QM9 dataset for Aurelius training.

Fetches QM9 quantum mechanical dataset from Hugging Face Hub.

Usage:
    python scripts/download_data.py

References:
    QM9: Ramakrishnan, R. et al. "QM9: 134 Kilo Molecules."
          Sci. Data 2014.
"""

from __future__ import annotations

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download QM9 dataset for Aurelius training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/download_data.py
        """,
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
        ds = load_dataset("maastrichtuniversity/qm9", split="train")
    except ValueError as e:
        print(f"[download_data] ERROR: Invalid dataset ID (ValueError): {e}")
        sys.exit(1)
    except ConnectionError as e:
        print(f"[download_data] ERROR: Network error: {e}")
        sys.exit(1)
    except (ImportError, RuntimeError, OSError) as e:
        print(f"[download_data] ERROR loading QM9: {e}")
        sys.exit(1)

    print(f"[download_data] Loaded {len(ds)} molecules from QM9")

    csv_path = os.path.join(output_path, "qm9.csv")
    with open(csv_path, "w") as f:
        f.write("smiles,U0\n")
        for item in ds:
            u0 = item["U0"] if not (isinstance(item["U0"], float) and str(item["U0"]) == "nan") else "nan"
            f.write(f"{item['smiles']},{u0}\n")

    print(f"[download_data] QM9 saved to: {csv_path}")
    return csv_path


def main() -> None:
    args = parse_args()
    output_dir = args.output

    os.makedirs(output_dir, exist_ok=True)
    print(f"[download_data] Output directory: {output_dir}")

    download_qm9(output_dir)

    print("\n[download_data] Dataset download complete!")


if __name__ == "__main__":
    main()
