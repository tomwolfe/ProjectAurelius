"""Data loaders for molecular datasets.

Provides functions to load ESOL and QM9 datasets from HuggingFace
or local fallback CSV files.
"""

from __future__ import annotations

import csv
import logging
import os

from typing import Any


def load_esol_data(csv_path: str | None = None) -> list[tuple[str, float]]:
    """Load the ESOL dataset.

    Args:
        csv_path: Optional path to a local CSV file. If None, loads from
            HuggingFace Hub.

    Returns:
        List of (smiles, logS) tuples.
    """
    from datasets import load_dataset

    ds = load_dataset("deepchem/esol", split="train")
    return [(sm, float(v)) for sm, v in zip(ds["smiles"], ds["logS"], strict=True)]


def load_qm9_lumo_data(csv_path: str | None = None) -> list[tuple[str, float]]:
    """Load the QM9 LUMO gap dataset.

    Args:
        csv_path: Optional path to a local CSV file. If None, loads from
            HuggingFace Hub.

    Returns:
        List of (smiles, lumo_gap) tuples.
    """
    from datasets import load_dataset

    ds = load_dataset("maastrichtuniversity/qm9", split="train")
    lumo_values = ds["LUMO"]
    smiles_list = list(ds["smiles"])
    return [(sm, float(lumo)) for sm, lumo in zip(smiles_list, lumo_values, strict=True)]


def load_qm9_homo_lumo_data(csv_path: str | None = None) -> list[tuple[str, float, float]]:
    """Load QM9 HOMO/LUMO data, trying HuggingFace first, then local CSV fallback.

    Returns:
        List of (smiles, homo_eV, lumo_eV) tuples.

    Raises:
        RuntimeError: If neither data source is available.
    """
    # Try HuggingFace first
    try:
        from datasets import load_dataset

        ds = load_dataset("maastrichtuniversity/qm9", split="train")
        homo_values = ds["HOMO"]
        lumo_values = ds["LUMO"]
        smiles_list = list(ds["smiles"])
        logger = logging.getLogger(__name__)
        logger.info(
            "load_qm9_homo_lumo_data: loaded %d entries from HuggingFace maastrichtuniversity/qm9",
            len(smiles_list),
        )
        return [(sm, float(h), float(l)) for sm, h, l in zip(smiles_list, homo_values, lumo_values, strict=True)]
    except Exception as hf_exc:
        logger = logging.getLogger(__name__)
        logger.warning("HuggingFace QM9 load failed: %s", hf_exc)

    # Fallback: bundled CSV
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "qm9_subset.csv")
    csv_path = os.path.abspath(csv_path)
    if not os.path.exists(csv_path):
        raise RuntimeError(
            f"QM9 data not available. Tried HuggingFace and local path {csv_path}. "
            "Install the 'datasets' package or provide a valid CSV."
        )

    data: list[tuple[str, float, float]] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = row.get("smiles", "").strip()
            homo_s = row.get("homo", "").strip()
            lumo_s = row.get("lumo", "").strip()
            if smi and homo_s and lumo_s:
                try:
                    data.append((smi, float(homo_s), float(lumo_s)))
                except ValueError:
                    continue

    logger = logging.getLogger(__name__)
    logger.info(
        "load_qm9_homo_lumo_data: loaded %d entries from fallback CSV %s",
        len(data),
        csv_path,
    )
    return data


def generate_ecfp4_fingerprint(smiles: str) -> list[int]:
    """Generate an ECFP4 (Morgan radius=2) fingerprint for a SMILES string.

    Args:
        smiles: A valid SMILES string.

    Returns:
        A list of 2048 binary bits representing the fingerprint.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    bits = fp.GetOnBits()
    result = [0] * 2048
    for idx in bits:
        result[idx] = 1
    return result
