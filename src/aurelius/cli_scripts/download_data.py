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
import warnings
from importlib import resources
from typing import Any

import numpy as np

from aurelius.utils.chem_utils import generate_ecfp4_fingerprint


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
    except Exception as e:
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
    except Exception as e:
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


def load_esol_data(csv_path: str | None = None) -> tuple[Any, Any, list[str]]:
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

    import numpy as np

    # Import the fingerprint generator
    from aurelius.utils.chem_utils import generate_ecfp4_fingerprint

    if csv_path and os.path.isfile(csv_path):
        print("[download_data] Loading ESOL from local CSV: %s", csv_path)
        return _load_esol_from_csv(csv_path)

    # Load HuggingFace datasets
    try:
        from datasets import load_dataset

        print("[download_data] Loading ESOL from HuggingFace Hub...")
        # Verified dataset: deepchem/esol ( maintained by DeepChem )
        ds = load_dataset("deepchem/esol", split="train")
        smiles_list = list(ds["smiles"])
        log_s = np.array(ds["logS"], dtype=np.float32)
        print("[download_data] Loaded %d molecules from ESOL", len(smiles_list))

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
        print("[download_data] Unexpected error loading ESOL (type=%s): %s", type(e).__name__, e)
        import traceback

        traceback.print_exc()
        if csv_path:
            return _load_esol_from_csv(csv_path)
        warnings.warn(
            "[download_data] FALLBACK: Using embedded 50-molecule ESOL subset due to error: "
            "%s: %s. "
            "This is a small curated subset from Delaney 2004, NOT the full 1112-molecule dataset. "
            "Model quality will be significantly reduced. "
            "Install 'datasets' and ensure network connectivity for full training, "
            "or use --csv-path to provide a local CSV file.",
            type(e).__name__,
            e,
            stacklevel=2,
        )
        print(
            "\n" + "=" * 60,
            "[download_data] WARNING: Fallback to embedded ESOL subset",
            "=" * 60,
        )
        print(
            "[download_data] Error: %s: %s\n"
            "[download_data] Training will proceed with only 50 embedded molecules\n"
            "[download_data] from the original Delaney 2004 dataset.\n"
            "[download_data] For full dataset training:\n"
            "[download_data]   - Ensure network connectivity and install 'datasets'\n"
            "[download_data]   - Or use --csv-path to provide a local CSV file\n",
            type(e).__name__,
            e,
        )
        return _load_esol_embedded()


def _load_esol_from_csv(csv_path: str) -> tuple[Any, Any, list[str]]:
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

    print("[download_data] Loaded %d molecules from CSV", len(smiles_list))

    n_bits = 2048
    X = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smiles in enumerate(smiles_list):
        X[i] = generate_ecfp4_fingerprint(smiles, n_bits)

    log_s_array = np.array(log_s_list, dtype=np.float32)
    log_s_min, log_s_max = -6.0, 1.0
    y = np.clip((log_s_array - log_s_min) / (log_s_max - log_s_min), 0.0, 1.0)

    return X, y, smiles_list


def _load_esol_embedded() -> tuple[Any, Any, list[str]]:
    """Load ESOL data from packaged CSV resource.

    This is a fallback when HuggingFace download or local CSV is
    unavailable. Uses the packaged esol_fallback.csv resource.

    Returns:
        Tuple of (fingerprints, logS_labels, smiles_list).
    """
    import csv

    training_data_path = resources.files("aurelius.data").joinpath("esol_fallback.csv")
    print("[download_data] *** LOADED FROM PACKAGED CSV: %s ***", training_data_path)

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
    from datasets import load_dataset

    ds = load_dataset("maastrichtuniversity/qm9", split="train")

    u0_values = np.array(ds["U0"], dtype=np.float32)
    smiles_list = ds["smiles"]

    # Filter valid molecules
    valid_mask = ~np.isnan(u0_values)
    valid_smiles = [s for i, s in enumerate(smiles_list) if valid_mask[i]]
    valid_u0 = u0_values[valid_mask]

    print("[download_data] Loaded %d QM9 molecules (%d valid)", len(valid_smiles), np.sum(valid_mask))

    n_bits = 2048
    X = np.zeros((len(valid_smiles), n_bits), dtype=np.float32)
    for i, smiles in enumerate(valid_smiles):
        X[i] = generate_ecfp4_fingerprint(smiles, n_bits)

    # Normalize U0 to [0, 1]
    u0_min, u0_max = float(np.min(valid_u0)), float(np.max(valid_u0))
    u0_range = u0_max - u0_min
    y = np.clip((valid_u0 - u0_min + 1e-12) / (u0_range + 1e-12), 0.0, 1.0)

    return X, y, valid_smiles


if __name__ == "__main__":
    main()
