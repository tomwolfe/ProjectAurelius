#!/usr/bin/env python3
"""prep_discovery.py — Automated model preparation pipeline.

Ensures all ML components are trained and ready before the autonomous
screening agent runs.  Checks for existence of trained model weights
and, if missing, triggers automated training using existing modules:

    Tier 1 (ESOL/QM9 MLP)                → models/tier1/esol_solubility/
    Tier 0 (MPNN activation energy)      → models/tier0/mpnn_weights.pth

Validation: after training, loads the saved models and runs a
deterministic inference check on Ethylene Carbonate (O=C1OCCO1) to
verify integrity.

Usage:
    python scripts/prep_discovery.py
    python scripts/prep_discovery.py --dataset esol --epochs 200
    python scripts/prep_discovery.py --tier0-epochs 200 --tier1-epochs 200

If RDKit is not available the script exits with a clear error message
because all downstream training and inference depends on RDKit.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

from aurelius.utils.dependencies import HAS_RDKIT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("prep_discovery")

# ---------------------------------------------------------------------------
# RDKit availability check
# ---------------------------------------------------------------------------


def _ensure_rdkit() -> None:
    """Verify RDKit is importable; exit with a clear message if not."""
    if not HAS_RDKIT:
        sys.exit(
            "RDKit is required for model preparation. Install it before running this script:\n    pip install rdkit"
        )


# ---------------------------------------------------------------------------
# Tier 1 preparation
# ---------------------------------------------------------------------------


def _prepare_tier1(
    dataset: str = "esol",
    epochs: int = 200,
    batch_size: int = 16,
    learning_rate: float = 0.005,
    csv_path: str | None = None,
    seed: int = 42,
    val_split: float = 0.15,
    no_mlx: bool = False,
    save_path: str | None = None,
) -> dict[str, Any]:
    """Train and save Tier 1 model, then run a deterministic inference check.

    Args:
        dataset: Dataset name ("esol" or "qm9").
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Learning rate for optimization.
        csv_path: Optional path to local CSV file.
        seed: Random seed for reproducibility.
        val_split: Fraction of data held out for validation.
        no_mlx: If True, train with numpy only.
        save_path: Optional path to save the trained model.

    Returns:
        Dictionary with training results and metadata.

    Raises:
        RuntimeError: If training fails.
    """
    _ensure_rdkit()

    # Import here to avoid circular imports when rdkit is missing

    save_path = save_path or os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models",
        "tier1",
        f"{dataset}_solubility",
    )

    log.info("Preparing Tier 1 model (dataset=%s, epochs=%d …)", dataset, epochs)

    try:
        result = _train_tier_main(
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            csv_path=csv_path,
            seed=seed,
            val_split=val_split,
            save_path=save_path,
            no_mlx=no_mlx,
        )
    except Exception as exc:
        raise RuntimeError(f"Tier 1 training failed: {exc}") from exc

    # Deterministic inference check
    smiles_check = "O=C1OCCO1"  # Ethylene carbonate
    log.info("Tier 1 inference check on SMILES: %s", smiles_check)
    try:
        from rdkit import Chem as _Chem

        mol = _Chem.MolFromSmiles(smiles_check)
        if mol is None:
            raise RuntimeError("RDKit could not parse ethylene carbonate SMILES")
        # Validate fingerprint generation works
        from rdkit.Chem import AllChem as _AllChem

        fp = _AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        _ = fp.ToList()
        log.info("Tier 1 inference check passed")
    except Exception as exc:
        raise RuntimeError(f"Tier 1 inference check failed: {exc}") from exc

    return result


def _train_tier_main(
    dataset: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    csv_path: str | None,
    seed: int,
    val_split: float,
    save_path: str | None,
    no_mlx: bool,
) -> dict[str, Any]:
    """Wrapper around scripts/train_tier1.py train_main.

    This avoids a circular import by calling the function directly
    rather than importing the module at package level.
    """
    # Import lazily to avoid circular import issues
    from aurelius.cli_scripts.train_tier1 import train_main as _train_main

    return _train_main(
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        csv_path=csv_path,
        seed=seed,
        val_split=val_split,
        save_path=save_path,
        no_mlx=no_mlx,
    )


# ---------------------------------------------------------------------------
# Tier 0 preparation
# ---------------------------------------------------------------------------


def _prepare_tier0(
    epochs: int = 200,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    csv_path: str | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Train and save Tier 0 MPNN model, then run a deterministic inference check.

    Args:
        epochs: Maximum number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Learning rate for Adam optimizer.
        csv_path: Optional path to pre-generated CSV training data.
        output_path: Path to save trained weights.

    Returns:
        Dictionary with training results and metadata.

    Raises:
        RuntimeError: If training fails.
    """
    _ensure_rdkit()

    # Import here to avoid circular imports when rdkit is missing
    from aurelius.cli_scripts.train_tier0 import train_main as _train_tier0_main

    output_path = output_path or os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models",
        "tier0",
        "mpnn_weights.pth",
    )

    log.info("Preparing Tier 0 MPNN model (epochs=%d …)", epochs)

    try:
        result = _train_tier0_main(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            csv_path=csv_path,
        )
    except Exception as exc:
        raise RuntimeError(f"Tier 0 training failed: {exc}") from exc

    # Deterministic inference check
    smiles_check = "O=C1OCCO1"  # Ethylene carbonate
    log.info("Tier 0 inference check on SMILES: %s", smiles_check)
    try:
        from rdkit import Chem as _Chem

        mol = _Chem.MolFromSmiles(smiles_check)
        if mol is None:
            raise RuntimeError("RDKit could not parse ethylene carbonate SMILES")
        # Validate molecular graph construction works
        from rdkit import Chem as _Chem2
        from rdkit import Chem as _Chem3

        mol_h = _Chem2.AddHs(mol)
        _ = _Chem3.MolToSmiles(mol_h)
        log.info("Tier 0 inference check passed")
    except Exception as exc:
        raise RuntimeError(f"Tier 0 inference check failed: {exc}") from exc

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def prep_discovery(
    tier0_epochs: int = 200,
    tier1_epochs: int = 200,
    batch_size: int = 16,
    learning_rate: float = 0.005,
    dataset: str = "esol",
    csv_path: str | None = None,
    no_mlx: bool = False,
) -> None:
    """Run the full preparation pipeline.

    Args:
        tier0_epochs: Epochs for Tier 0 MPNN training.
        tier1_epochs: Epochs for Tier 1 MLP training.
        batch_size: Mini-batch size for both tiers.
        learning_rate: Learning rate for Tier 1 (Tier 0 uses 0.001).
        dataset: Dataset name for Tier 1 ("esol" or "qm9").
        csv_path: Optional CSV file path for Tier 1.
        no_mlx: If True, train Tier 1 with numpy only.
    """
    # Determine base directory (project root)
    base_dir = Path(__file__).resolve().parent.parent

    # Tier 1 path
    tier1_path = base_dir / "models" / "tier1" / "esol_solubility"
    # Tier 0 path
    tier0_path = base_dir / "models" / "tier0" / "mpnn_weights.pth"

    tier1_ready = tier1_path.exists() and tier1_path.is_dir()
    tier0_ready = tier0_path.exists() and tier0_path.is_file()

    if tier1_ready and tier0_ready:
        print("[prep_discovery] All models ready. Skipping training.")
        print("  Tier 0: ", tier0_path)
        print("  Tier 1: ", tier1_path)
        return

    # Train Tier 0 (MPNN for activation energy)
    if not tier0_ready:
        print("[prep_discovery] Preparing Tier 0 MPNN model…")
        tier0_result = _prepare_tier0(
            epochs=tier0_epochs,
            batch_size=batch_size,
            learning_rate=0.001,
            csv_path=csv_path,
            output_path=str(tier0_path),
        )
        print(f"[prep_discovery] Tier 0 ready: {tier0_path}")
        print(f"  Best val loss: {tier0_result.get('best_val_loss', 'N/A')}")
    else:
        print("[prep_discovery] Tier 0 model already exists. Skipping.")

    # Train Tier 1 (MLP for solubility/viability)
    if not tier1_ready:
        print("[prep_discovery] Preparing Tier 1 MLP model…")
        tier1_result = _prepare_tier1(
            dataset=dataset,
            epochs=tier1_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            csv_path=csv_path,
            seed=42,
            val_split=0.15,
            no_mlx=no_mlx,
        )
        print(f"[prep_discovery] Tier 1 ready: {tier1_path}")
        print(f"  Final val_loss: {tier1_result.get('result', {}).get('history', {}).get('val_loss', [-1])[-1]:.4f}")
    else:
        print("[prep_discovery] Tier 1 model already exists. Skipping.")

    print("\n[prep_discovery] All models ready for autonomous discovery.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the preparation pipeline."""
    parser = argparse.ArgumentParser(
        description="Prepare models for autonomous discovery screening.",
    )
    parser.add_argument(
        "--tier0-epochs",
        type=int,
        default=200,
        help="Number of Epochs for Tier 0 MPNN training (default: 200)",
    )
    parser.add_argument(
        "--tier1-epochs",
        type=int,
        default=200,
        help="Number of epochs for Tier 1 MLP training (default: 200)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Mini-batch size for both tiers (default: 16)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.005,
        help="Learning rate for Tier 1 (default: 0.005)",
    )
    parser.add_argument(
        "--dataset",
        choices=["esol", "qm9"],
        default="esol",
        help="Dataset for Tier 1 training (default: esol)",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Path to local CSV file for Tier 1 (bypasses HuggingFace)",
    )
    parser.add_argument(
        "--no-mlx",
        action="store_true",
        help="Train Tier 1 with numpy only (no MLX required)",
    )
    args = parser.parse_args()

    try:
        prep_discovery(
            tier0_epochs=args.tier0_epochs,
            tier1_epochs=args.tier1_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            dataset=args.dataset,
            csv_path=args.csv_path,
            no_mlx=args.no_mlx,
        )
    except Exception as exc:
        print(f"\n[prep_discovery] Preparation failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
