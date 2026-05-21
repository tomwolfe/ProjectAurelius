"""Tier 0: Data generation and training functions.

Handles synthetic training data generation and MPNN model training
for activation energy prediction.

Training Data:
    - Deterministic synthetic dataset generated via RDKit + Arrhenius shifts + Gaussian noise
    - 500 rows, sigma=0.05 eV noise on literature-calibrated values

Apple Silicon Optimization:
    - Pure PyTorch (no torch_scatter) for MPS compatibility
    - Index-based aggregation via torch.scatter_add + torch.bincount
    - Model footprint < 50MB on MPS
"""

from __future__ import annotations

import csv
import os
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore[assignment, unused-ignore]
    nn = None  # type: ignore[assignment, unused-ignore]

if HAS_TORCH:
    from aurelius.screening.tier0.models import Tier0MPNN


def _build_molecular_graph(
    smiles: str, device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a molecular graph from a SMILES string.

    Uses RDKit to extract atomic features and adjacency information,
    yielding node feature and edge index tensors suitable for
    message passing.

    Args:
        smiles: SMILES string of the molecule.
        device: PyTorch device for tensor placement (default: "cpu").

    Returns:
        Tuple of (node_features, edge_index).

    Raises:
        RuntimeError: If PyTorch or RDKit is not available.
        ValueError: If SMILES is invalid.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for molecular graph construction.")

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        raise RuntimeError(
            "RDKit is required for molecular graph construction. "
            "Install with: pip install rdkit"
        ) from None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol_with_h = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_with_h, randomSeed=42)  # type: ignore[attr-defined, unused-ignore]

    atoms = mol_with_h.GetAtoms()  # type: ignore[no-untyped-call, unused-ignore]
    n_atoms = mol_with_h.GetNumAtoms()

    node_features = torch.zeros(n_atoms, 4, dtype=torch.float32, device=device)
    for i, atom in enumerate(atoms):
        node_features[i, 0] = atom.GetAtomicNum() / 100.0
        node_features[i, 1] = atom.GetDegree() / 10.0
        node_features[i, 2] = float(atom.GetFormalCharge()) / 10.0
        node_features[i, 3] = 1.0 if atom.GetIsAromatic() else 0.0

    adj = Chem.rdmolops.GetAdjacencyMatrix(mol_with_h)
    edge_list: list[tuple[int, int]] = []
    for i in range(n_atoms):
        for j in range(n_atoms):
            if adj[i, j] == 1 and i < j:
                edge_list.append((i, j))
                edge_list.append((j, i))

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    return node_features, edge_index


def _load_tier0_seed_smiles() -> list[str]:
    """Load seed SMILES for Tier 0 synthetic data generation.

    Returns:
        List of SMILES strings for synthetic training data.
    """
    import json
    from importlib import resources

    data_path = resources.files("aurelius.data")
    smiles_path = data_path.joinpath("tier0_seed_smiles.json")

    try:
        with open(smiles_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        # Fallback to empty list if file not found
        return []


def generate_synthetic_training_data(
    n_samples: int = 500,
    noise_sigma: float = 0.05,
    output_path: str | None = None,
) -> list[dict[str, Any]]:
    """Generate a deterministic synthetic training dataset.

    Uses RDKit to compute molecular descriptors, then applies
    Arrhenius-shifted activation energy models with Gaussian noise
    (sigma=0.05 eV) to create training targets.

    Args:
        n_samples: Number of samples to generate (default: 500).
        noise_sigma: Standard deviation of Gaussian noise in eV (default: 0.05).
        output_path: Optional path to save CSV. If None, returns data only.

    Returns:
        List of dictionaries with SMILES and activation energy targets.

    Raises:
        RuntimeError: If RDKit is not available.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError:
        raise RuntimeError(
            "RDKit is required for synthetic data generation. "
            "Install with: pip install rdkit"
        ) from None

    # Load seed SMILES from external file
    base_smiles = _load_tier0_seed_smiles()

    valid_smiles: list[str] = []
    for smi in base_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_smiles.append(smi)

    while len(valid_smiles) < n_samples:
        valid_smiles.extend(valid_smiles[: n_samples - len(valid_smiles)])
    valid_smiles = valid_smiles[:n_samples]

    rng = np.random.RandomState(42)

    from rdkit.Chem import AllChem

    training_data: list[dict[str, Any]] = []
    for smi in valid_smiles:
        mol = Chem.MolFromSmiles(smi)
        mol_with_h = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_with_h, randomSeed=42)  # type: ignore[attr-defined, unused-ignore]

        logp = float(Descriptors.MolLogP(mol_with_h))  # type: ignore[attr-defined, unused-ignore]
        hba = int(Descriptors.NumHAcceptors(mol_with_h))  # type: ignore[attr-defined, unused-ignore]
        hbd = int(Descriptors.NumHDonors(mol_with_h))  # type: ignore[attr-defined, unused-ignore]
        tpsa = float(Descriptors.TPSA(mol_with_h))  # type: ignore[attr-defined, unused-ignore]
        aromatic_count = sum(1 for a in mol_with_h.GetAtoms() if a.GetIsAromatic())  # type: ignore[no-untyped-call, misc, unused-ignore]
        aromatic_ratio = aromatic_count / max(mol_with_h.GetNumAtoms(), 1)

        ec_base = 0.65 + 0.08 * logp - 0.02 * hba - 0.03 * hbd - 0.003 * tpsa + 0.15 * aromatic_ratio
        dm_base = ec_base * 1.15
        pf6_base = 1.15 + 0.05 * logp + 0.01 * hba + 0.02 * hbd + 0.002 * tpsa + 0.10 * aromatic_ratio
        poly_base = 0.45 + 0.06 * logp - 0.01 * hba - 0.02 * hbd - 0.002 * tpsa + 0.20 * aromatic_ratio

        ec_base = float(np.clip(ec_base, 0.45, 0.95))
        dm_base = float(np.clip(dm_base, 0.45, 1.10))
        pf6_base = float(np.clip(pf6_base, 0.90, 1.50))
        poly_base = float(np.clip(poly_base, 0.30, 0.70))

        ec = float(ec_base + rng.normal(0, noise_sigma))
        dm = float(dm_base + rng.normal(0, noise_sigma))
        pf6 = float(pf6_base + rng.normal(0, noise_sigma))
        poly = float(poly_base + rng.normal(0, noise_sigma))

        entry = {
            "smiles": smi,
            "ec_reduction": round(ec, 6),
            "dm_reduction": round(dm, 6),
            "pf6_decomposition": round(pf6, 6),
            "polymerization": round(poly, 6),
        }
        training_data.append(entry)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["smiles", "ec_reduction", "dm_reduction", "pf6_decomposition", "polymerization"])
            writer.writeheader()
            writer.writerows(training_data)

    return training_data


def train_tier0_model(
    n_epochs: int = 200,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    early_stop_patience: int = 30,
    train_csv_path: str | None = None,
    output_path: str = "models/tier0/mpnn_weights.pth",
    data_dir: str = "data",
) -> dict[str, Any]:
    """Train the Tier 0 MPNN model on synthetic data.

    Uses MSE loss with early stopping. Generates synthetic training
    data if no CSV is provided.

    Args:
        n_epochs: Maximum number of training epochs (default: 200).
        batch_size: Mini-batch size (default: 16).
        learning_rate: Learning rate (default: 0.001).
        early_stop_patience: Early stopping patience in epochs (default: 30).
        train_csv_path: Optional path to pre-generated CSV.
        output_path: Path to save trained weights.
        data_dir: Directory for synthetic data CSV.

    Returns:
        Dictionary with training metrics.

    Raises:
        RuntimeError: If PyTorch is unavailable.
        ValueError: If no training data available.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for model training.")

    try:
        from rdkit import Chem  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "RDKit is required for molecular graph construction. "
            "Install with: pip install rdkit"
        ) from None

    if train_csv_path:
        required_columns = {"smiles", "ec_reduction", "dm_reduction", "pf6_decomposition", "polymerization"}
        training_data: list[dict[str, Any]] = []

        with open(train_csv_path) as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file '{train_csv_path}' appears to be empty or has no headers.")

            actual_columns = set(reader.fieldnames)
            missing_columns = required_columns - actual_columns
            if missing_columns:
                raise ValueError(
                    f"CSV file '{train_csv_path}' is missing required columns: {sorted(missing_columns)}. "
                    f"Required columns: {sorted(required_columns)}."
                )

            for row in reader:
                training_data.append({
                    "smiles": row["smiles"],
                    "ec_reduction": float(row["ec_reduction"]),
                    "dm_reduction": float(row["dm_reduction"]),
                    "pf6_decomposition": float(row["pf6_decomposition"]),
                    "polymerization": float(row["polymerization"]),
                })
    else:
        csv_path = os.path.join(data_dir, "train_tier0_synthetic.csv")
        training_data = generate_synthetic_training_data(n_samples=500, output_path=csv_path)

    if not training_data:
        raise ValueError("No training data available. Provide --csv-path or generate synthetic data.")

    node_features_list: list[torch.Tensor] = []
    edge_index_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []

    device = "cpu"

    for entry in training_data:
        nf, ei = _build_molecular_graph(entry["smiles"], device=device)
        target = torch.tensor([
            entry["ec_reduction"],
            entry["dm_reduction"],
            entry["pf6_decomposition"],
            entry["polymerization"],
        ], dtype=torch.float32, device=device)
        node_features_list.append(nf)
        edge_index_list.append(ei)
        targets_list.append(target)

    n_train = len(training_data)

    def _collate_fn(batch_indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        """Collate variable-length graphs into padded batch."""
        batch_nf: list[torch.Tensor] = []
        batch_ei: list[torch.Tensor] = []
        batch_targets: list[torch.Tensor] = []
        offsets: list[int] = [0]

        for idx in batch_indices:
            nf = node_features_list[idx]
            ei = edge_index_list[idx]
            tgt = targets_list[idx]

            n_prev = offsets[-1]
            ei_padded = ei.clone()
            if ei_padded.numel() > 0:
                ei_padded[0] = ei_padded[0] + n_prev
                ei_padded[1] = ei_padded[1] + n_prev

            batch_nf.append(nf)
            batch_ei.append(ei_padded)
            batch_targets.append(tgt)
            offsets.append(offsets[-1] + nf.shape[0])

        padded_nf = torch.cat(batch_nf, dim=0)
        padded_ei = torch.cat(batch_ei, dim=1) if batch_ei else torch.empty((2, 0), dtype=torch.long)
        batch_tgt = torch.stack(batch_targets)

        return padded_nf, padded_ei, batch_tgt, offsets

    model = Tier0MPNN(node_dim=4, edge_dim=8, hidden_dim=64, output_dim=4)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float("inf")
    best_state: dict[str, Any] = {}
    patience_counter = 0
    epoch_losses: list[float] = []
    val_losses: list[float] = []

    rng = np.random.RandomState(42)
    indices = list(range(n_train))
    rng.shuffle(indices)
    split = int(0.8 * n_train)
    train_idx = indices[:split]
    val_idx = indices[split:]

    print(f"[Tier0MPNN] Training on {len(train_idx)} samples, validating on {len(val_idx)} samples")

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        rng.shuffle(train_idx)
        for i in range(0, len(train_idx), batch_size):
            batch_indices = train_idx[i : i + batch_size]
            if not batch_indices:
                continue

            nf, ei, tgt, _ = _collate_fn(batch_indices=batch_indices)
            nf = nf.to(device)
            tgt = tgt.to(device)

            optimizer.zero_grad()
            preds = model(nf, ei)
            loss = criterion(preds, tgt)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        epoch_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for i in range(0, len(val_idx), batch_size):
                batch_indices = val_idx[i : i + batch_size]
                if not batch_indices:
                    continue
                nf, ei, tgt, _ = _collate_fn(batch_indices=batch_indices)
                nf = nf.to(device)
                tgt = tgt.to(device)
                preds = model(nf, ei)
                val_loss += criterion(preds, tgt).item()
                n_val_batches += 1

        avg_val_loss = val_loss / max(n_val_batches, 1)
        val_losses.append(avg_val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[Tier0MPNN] Epoch {epoch+1}/{n_epochs} - "
                  f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"[Tier0MPNN] Early stopping at epoch {epoch+1}. Best val loss: {best_loss:.6f}")
                break

    if best_state:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    model.save_weights(output_path)

    csv_path = os.path.join(data_dir, "train_tier0_synthetic.csv")
    if not os.path.exists(csv_path):
        generate_synthetic_training_data(n_samples=500, output_path=csv_path)

    print(f"[Tier0MPNN] Training complete. Best val loss: {best_loss:.6f}. "
          f"Weights saved to {output_path}")

    return {
        "final_train_loss": float(epoch_losses[-1]) if epoch_losses else 0.0,
        "best_val_loss": float(best_loss),
        "epochs_run": len(epoch_losses),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "weights_path": output_path,
    }
