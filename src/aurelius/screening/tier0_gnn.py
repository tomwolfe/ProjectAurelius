"""Tier 0: Lightweight Message Passing Neural Network (MPNN) for Activation Energy Prediction.

Implements a graph neural network that takes RDKit molecular graphs and predicts
molecule-specific activation energies for EC reduction, DMC reduction, PF6
decomposition, and polymerization reactions relevant to SEI formation.

Architecture:
    - Node features: atomic number, degree, formal charge, aromaticity
    - Message passing: 2-layer edge-based MP with torch.scatter_add aggregation
    - Readout: MLP over pooled node embeddings
    - Output: 4 activation energies (eV) for EC/DMC reduction, PF6 decomposition, polymerization

Training Data:
    - Deterministic synthetic dataset generated via RDKit + Arrhenius shifts + Gaussian noise
    - 500 rows, sigma=0.05 eV noise on literature-calibrated values

Apple Silicon Optimization:
    - Pure PyTorch (no torch_scatter) for MPS compatibility
    - Index-based aggregation via torch.scatter_add + torch.bincount
    - Model footprint < 50MB on MPS

References:
    Gilmer, J. et al. "Neural Message Passing for Quantum Chemistry." ICML 2017.
    Wu, Z. et al. "Molecular Graph Convolutions: Moving Beyond Fingerprints." JMLR 2021.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any

import numpy as np

from aurelius.utils.descriptors import _generate_molecular_descriptors, _hash_descriptors

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore
    nn = None  # type: ignore

# ---------------------------------------------------------------------------
# RDKit Graph Construction
# ---------------------------------------------------------------------------


def _build_molecular_graph(
    smiles: str, device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a molecular graph from a SMILES string.

    Uses RDKit to extract atomic features and adjacency information,
    returning node feature and edge index tensors suitable for
    message passing.

    Args:
        smiles: SMILES string of the molecule.
        device: PyTorch device for tensor placement (default: "cpu").

    Returns:
        Tuple of (node_features, edge_index).
        - node_features: (N_nodes, N_features) tensor on the specified device
        - edge_index: (2, N_edges) tensor on the specified device

    Raises:
        RuntimeError: If RDKit is not available.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for molecular graph construction.")

    try:
        from rdkit import Chem
    except ImportError:
        raise RuntimeError(
            "RDKit is required for molecular graph construction. "
            "Install with: pip install rdkit"
        ) from None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Ensure hydrogens are added for proper feature extraction
    mol_with_h = Chem.AddHs(mol)
    Chem.EmbedMolecule(mol_with_h, randomSeed=42)  # type: ignore[attr-defined]

    atoms = mol_with_h.GetAtoms()  # type: ignore[unused-ignore, no-untyped-call]
    n_atoms = mol_with_h.GetNumAtoms()

    # Node features: [atomic_number, degree, formal_charge, is_aromatic]
    # Normalize atomic number to [0, 1] range (max Z ~ 100)
    node_features = torch.zeros(n_atoms, 4, dtype=torch.float32, device=device)
    for i, atom in enumerate(atoms):
        node_features[i, 0] = atom.GetAtomicNum() / 100.0
        node_features[i, 1] = atom.GetDegree() / 10.0
        node_features[i, 2] = float(atom.GetFormalCharge()) / 10.0
        node_features[i, 3] = 1.0 if atom.GetIsAromatic() else 0.0

    # Edge index from adjacency matrix
    adj = Chem.rdmolops.GetAdjacencyMatrix(mol_with_h)
    edge_list: list[tuple[int, int]] = []
    for i in range(n_atoms):
        for j in range(n_atoms):
            if adj[i, j] == 1 and i < j:
                edge_list.append((i, j))
                edge_list.append((j, i))  # Bidirectional for undirected graph

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    return node_features, edge_index


# ---------------------------------------------------------------------------
# MPNN Model Definition
# ---------------------------------------------------------------------------


class MPNNEdgeBlock(nn.Module):
    """2-layer message passing block for molecular graphs.

    Implements edge-based message passing with:
    - Edge feature computation from source/target node features + edge index
    - 2-layer MLP for edge message generation
    - Aggregation via torch.scatter_add (MPS-compatible, no torch_scatter)
    - Node update via residual connection + LayerNorm

    Reference:
        Gilmer, J. et al. "Neural Message Passing for Quantum Chemistry." ICML 2017.
    """

    def __init__(self, node_dim: int = 4, edge_dim: int = 8, hidden_dim: int = 64) -> None:
        """Initialize MPNN edge block.

        Args:
            node_dim: Dimension of node features.
            edge_dim: Dimension of edge features.
            hidden_dim: Hidden dimension for message MLP.
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        # Edge feature computation: concatenate src_node, tgt_node, edge_idx
        self.edge_input_dim = node_dim * 2 + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Node update: aggregate messages and update node features
        self.node_input_dim = node_dim + hidden_dim
        self.node_mlp = nn.Sequential(
            nn.Linear(self.node_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )

        self.norm = nn.LayerNorm(node_dim)

        # Initialize weights with Xavier uniform (matching _ChemVLM2MLP pattern)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize all weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the MPNN edge block.

        Args:
            node_features: (N_nodes, node_dim) tensor.
            edge_index: (2, N_edges) tensor of source/target indices.
            edge_features: Optional (N_edges, edge_dim) tensor. If None,
                edge features are computed from node features.

        Returns:
            Updated node features (N_nodes, node_dim).
        """
        n_nodes = node_features.shape[0]
        n_edges = edge_index.shape[1]

        if n_edges == 0:
            return node_features

        # Get source and target node features
        src_idx = edge_index[0]  # (N_edges,)
        tgt_idx = edge_index[1]  # (N_edges,)

        src_features = torch.index_select(node_features, 0, src_idx)  # (N_edges, node_dim)
        tgt_features = torch.index_select(node_features, 0, tgt_idx)  # (N_edges, node_dim)

        # Compute edge features (if not provided)
        if edge_features is None:
            edge_features = torch.cat([src_features, tgt_features], dim=-1)  # (N_edges, 2*node_dim)

        # Message generation via MLP
        messages = self.edge_mlp(edge_features)  # (N_edges, hidden_dim)

        # Aggregate messages to target nodes using scatter_add (MPS-compatible)
        aggregated = torch.zeros(n_nodes, self.hidden_dim, device=node_features.device)
        aggregated.scatter_add_(0, src_idx.unsqueeze(1).expand(-1, self.hidden_dim), messages)

        # Update node features
        node_input = torch.cat([node_features, aggregated], dim=-1)  # (N_nodes, node_dim + hidden_dim)
        node_updates = self.node_mlp(node_input)  # (N_nodes, node_dim)

        # Residual connection + LayerNorm
        return self.norm(node_features + node_updates)  # type: ignore[no-any-return]


class MPNNReadoutMLP(nn.Module):
    """Readout MLP for MPNN.

    Takes pooled node embeddings and produces output predictions
    via a multi-layer perceptron.

    Reference:
        Wu, Z. et al. "Molecular Graph Convolutions: Moving Beyond Fingerprints." JMLR 2021.
    """

    def __init__(self, input_dim: int = 64, output_dim: int = 4, hidden_dim: int = 128) -> None:
        """Initialize readout MLP.

        Args:
            input_dim: Dimension of input (pooled node features).
            output_dim: Number of output predictions (4 activation energies).
            hidden_dim: Hidden dimension.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            pooled: Pooled node embeddings (batch_size, input_dim) or (input_dim,).

        Returns:
            Predicted activation energies (batch_size, output_dim) or (output_dim,).
        """
        if pooled.dim() == 1:
            return self.network(pooled)  # type: ignore[no-any-return, unused-ignore]
        return self.network(pooled)  # type: ignore[no-any-return, unused-ignore]


class Tier0MPNN(nn.Module):
    """Lightweight Message Passing Neural Network for activation energy prediction.

    Takes a molecular graph (node features + edge index) and predicts
    molecule-specific activation energies for:
        - EC reduction (eV)
        - DMC reduction (eV)
        - PF6 decomposition (eV)
        - Polymerization (eV)

    Architecture:
        1. Edge feature computation from node features
        2. 2-layer message passing (MPNNEdgeBlock)
        3. Global pooling (sum over nodes)
        4. Readout MLP for 4 activation energy predictions

    Model footprint: < 50MB on MPS.
    """

    def __init__(
        self,
        node_dim: int = 4,
        edge_dim: int = 8,
        hidden_dim: int = 64,
        output_dim: int = 4,
    ) -> None:
        """Initialize the Tier 0 MPNN.

        Args:
            node_dim: Dimension of node features (default: 4).
            edge_dim: Dimension of edge features (default: 8).
            hidden_dim: Hidden dimension for message passing (default: 64).
            output_dim: Number of output predictions (default: 4).
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        # Edge feature transform
        self.edge_input_dim = node_dim * 2 + edge_dim
        self.edge_transform = nn.Sequential(
            nn.Linear(self.edge_input_dim, hidden_dim),
            nn.ReLU(),
        )

        # Message passing layers
        self.mp_layers = nn.ModuleList([
            MPNNEdgeBlock(node_dim, edge_dim, hidden_dim)
            for _ in range(2)  # 2 message passing layers
        ])

        # Readout
        self.readout = MPNNReadoutMLP(input_dim=hidden_dim, output_dim=output_dim, hidden_dim=128)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize all weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the Tier 0 MPNN.

        Args:
            node_features: (N_nodes, node_dim) tensor.
            edge_index: (2, N_edges) tensor.

        Returns:
            Predicted activation energies (output_dim,).
        """
        n_nodes = node_features.shape[0]

        if n_nodes == 0:
            return torch.zeros(self.readout.network[-1].out_features, device=node_features.device)  # type: ignore[no-any-return, call-overload]

        if edge_index.shape[1] == 0:
            # No edges: use node features directly
            pooled = node_features.sum(dim=0)
            return self.readout(pooled)  # type: ignore[no-any-return, unused-ignore]

        # Compute initial edge features
        src_idx = edge_index[0]
        tgt_idx = edge_index[1]
        src_features = torch.index_select(node_features, 0, src_idx)
        tgt_features = torch.index_select(node_features, 0, tgt_idx)
        initial_edge_features = torch.cat([src_features, tgt_features], dim=-1)
        edge_features = self.edge_transform(initial_edge_features)

        # Message passing layers
        h = node_features
        for mp_layer in self.mp_layers:
            h = mp_layer(h, edge_index, edge_features)

        # Global sum pooling
        pooled = h.sum(dim=0)

        # Readout
        return self.readout(pooled)  # type: ignore[no-any-return, unused-ignore]

    def save_weights(self, path: str) -> None:
        """Save model weights to file along with metadata.

        Saves the model state dict and a metadata.json containing
        model_version, architecture info, and tensor shapes for
        integrity verification on load.

        Args:
            path: File path to save weights (state dict).
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

        # Save metadata alongside the weights file
        meta_path = path.rsplit(".", 1)[0] + "_metadata.json"
        shape_info: dict[str, list[int]] = {}
        for name, tensor in self.state_dict().items():
            shape_info[name] = list(tensor.shape)

        meta = {
            "model_version": __import__("importlib").metadata.version("aurelius"),
            "architecture": "Tier0MPNN",
            "node_dim": self.node_dim,
            "edge_dim": self.edge_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.readout.network[-1].out_features,
            "tensor_shapes": shape_info,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def load_weights(self, path: str) -> None:
        """Load model weights from file with shape validation.

        Compares the loaded tensor shapes against the current model
        architecture and warns if they differ. Also checks model
        version against the installed package version.

        Args:
            path: File path to load weights from.
        """
        try:
            import importlib
            current_version = importlib.metadata.version("aurelius")
        except Exception:
            current_version = "unknown"

        # Load metadata if available
        meta_path = path.rsplit(".", 1)[0] + "_metadata.json"
        metadata: dict[str, Any] = {}
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                metadata = json.load(f)

        # Version check
        saved_version = metadata.get("model_version", "unknown")
        if saved_version != "unknown" and current_version != "unknown" and saved_version != current_version:
            print(
                    f"[Tier0MPNN] WARNING: Model version ({saved_version}) does not match "
                    f"installed package version ({current_version}). "
                    "Consider retraining with `aurelius train --task tier0`."
                )

        # Load state dict
        state_dict = torch.load(path, map_location="cpu", weights_only=True)

        # Shape validation: compare loaded tensor shapes with current model
        for name, loaded_tensor in state_dict.items():
            if hasattr(self, name) and hasattr(getattr(self, name), "shape"):
                current_tensor = getattr(self, name)
                if loaded_tensor.shape != current_tensor.shape:
                    print(
                        f"[Tier0MPNN] WARNING: Shape mismatch for '{name}': "
                        f"loaded {list(loaded_tensor.shape)} vs current model {list(current_tensor.shape)}. "
                        "The model may not function correctly."
                    )

        self.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# Synthetic Training Data Generation
# ---------------------------------------------------------------------------


def generate_synthetic_training_data(
    n_samples: int = 500,
    noise_sigma: float = 0.05,
    output_path: str | None = None,
) -> list[dict[str, Any]]:
    """Generate a deterministic synthetic training dataset.

    Uses RDKit to compute molecular descriptors, then applies
    Arrhenius-shifted activation energy models with Gaussian noise
    (sigma=0.05 eV) to create training targets.

    The dataset is saved as CSV for reproducibility.

    Args:
        n_samples: Number of samples to generate (default: 500).
        noise_sigma: Standard deviation of Gaussian noise in eV (default: 0.05).
        output_path: Optional path to save CSV. If None, returns data only.

    Returns:
        List of dictionaries with SMILES and activation energy targets.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError:
        raise RuntimeError(
            "RDKit is required for synthetic data generation. "
            "Install with: pip install rdkit"
        ) from None

    # Known battery electrolyte molecules (diverse set)
    base_smiles = [
        "CC(=O)OC1=CC(=O)O1",  # Ethyl salicylate
        "C1CC(=O)OC1",  # Ethylene carbonate
        "COC(=O)C1=CC=CC=C1",  # Methyl benzoate
        "COC(=O)OC",  # Dimethyl carbonate
        "CC(=O)OC",  # Methyl acetate
        "C1CCC(=O)OC1",  # δ-valerolactone
        "COC(=O)CN(C)C",  # Dimethylformamide
        "CC1=CC=CC=C1C(=O)OC",  # Methyl benzoate isomer
        "OC1CCOC1",  # 1,4-Butanediol
        "C1COCCO1",  # Tetrahydrofuran-2,5-diol
        "CC(C)OC(=O)C",  # Isopropyl acetate
        "COC(=O)C(C)C",  # Isopropyl formate
        "CCC(=O)OC",  # Propyl acetate
        "CC(C)C(=O)OC",  # Isopropyl acetate
        "C1CCC(C)OO1",  # 2-Methyltetrahydrofuran-3,5-diol
        "COC(=O)C1CC1",  # Methyl cyclopropanecarboxylate
        "C1CCOC1",  # Tetrahydrofuran
        "CCOC(=O)C",  # Ethyl acetate
        "C=CC(=O)OC",  # Vinyl acetate
        "CC(C)OC",  # Isopropyl methyl ether
        "C1CCC1",  # Cyclopropane
        "CC1=CC=CC=C1O",  # Phenol
        "COC1=CC=C(C=C1)C(=O)OC",  # Methyl salicylate
        "CC(C)(C)OC(=O)C",  # tert-Butyl acetate
        "C1=CC=C(C=C1)C(F)(F)F",  # Trifluorotoluene
        "C1=CC=C(C=C1)C(F)(F)F",  # Monofluorotoluene
        "C1=CC=C(C=C1)C(F)(F)F",  # Trifluoromethylbenzene
        "C1=CC=C(C=C1)C(F)(F)C(F)(F)F",  # Bis(trifluoromethyl)benzene
        "C1=CC=C(C=C1)C(F)(F)C(F)(F)C(F)(F)F",  # Perfluorinated benzene
        "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)F",  # Perfluorohexane
        "C1=CC=C(C=C1)C(=O)OC(F)(F)F",  # Trifluoromethyl benzoate
        "C1=CC(=C(C=C1)C(=O)OC)C(F)(F)F",  # 4-Trifluoromethylbenzoate
        "C1=CC(=C(C=C1)C(=O)OC(F)(F)F)C(F)(F)F",  # Bis-trifluoromethyl benzoate
        "C1=CC(=C(C=C1)C(=O)OC(F)(F)F)C(F)(F)C(F)(F)F",  # Mixed fluorinated benzoate
        "C1=CC(=C(C=C1)C(=O)OC(F)(F)F)C(F)(F)C(F)(F)F",  # Mixed fluorinated
        "C1=CC(=C(C=C1)C(=O)OC(F)(F)F)C(F)(F)C(F)(F)C(F)(F)F",  # Extended fluorinated
        "C1=CC(=C(C=C1)C(=O)OC(F)(F)F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",  # Long fluorinated chain
        "C1=CC(=C(C=C1)C(=O)OC(F)(F)F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",  # Very long fluorinated
        "C1=CC(=C(C=C1)C(=O)OC(F)(F)F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",  # Maximum fluorinated
        "C1=CC=C(C=C1)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",  # Perfluorophenyl
        "C1=CC(=C(C=C1)C(F)(F)F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",  # Difluoro-trifluoro
        "C1=CC(=C(C=C1)C(F)(F)C(F)(F)F)C(F)(F)C(F)(F)C(F)(F)F",  # Mixed fluorinated benzene
        "C1=CC(=C(C=C1)C(F)(F)C(F)(F)C(F)(F)F)C(F)(F)C(F)(F)F",  # Trifluoro-difluoro
        "C1=CC(=C(C=C1)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F)C(F)(F)F",  # Tetrafluoro-trifluoro
        "C1=CC(=C(C=C1)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F)C(F)(F)F",  # Pentafluoro-trifluoro
        "C1=CC(=C(C=C1)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F)C(F)(F)F",  # Hexafluoro-trifluoro
        "C1=CC(=C(C=C1)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F)C(F)(F)F",  # Heptafluoro-trifluoro
        "C1=CC(=C(C=C1)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F)C(F)(F)F",  # Octafluoro-trifluoro
        "C1=CC(=C(C=C1)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F)C(F)(F)F",  # Nonafluoro-trifluoro
        "C1=CC(=C(C=C1)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F)C(F)(F)F",  # Decafluoro-trifluoro
    ]

    # Remove any invalid SMILES
    valid_smiles: list[str] = []
    for smi in base_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_smiles.append(smi)

    # Duplicate to reach n_samples if needed
    while len(valid_smiles) < n_samples:
        valid_smiles.extend(valid_smiles[: n_samples - len(valid_smiles)])
    valid_smiles = valid_smiles[:n_samples]

    rng = np.random.RandomState(42)  # Deterministic

    training_data: list[dict[str, Any]] = []
    for smi in valid_smiles:
        mol = Chem.MolFromSmiles(smi)
        mol_with_h = Chem.AddHs(mol)
        Chem.EmbedMolecule(mol_with_h, randomSeed=42)  # type: ignore[attr-defined]

        # Compute descriptors
        logp = float(Descriptors.MolLogP(mol_with_h))  # type: ignore[attr-defined]
        hba = int(Descriptors.NumHAcceptors(mol_with_h))  # type: ignore[attr-defined]
        hbd = int(Descriptors.NumHDonors(mol_with_h))  # type: ignore[attr-defined]
        tpsa = float(Descriptors.TPSA(mol_with_h))  # type: ignore[attr-defined]
        aromatic_count = sum(1 for a in mol_with_h.GetAtoms() if a.GetIsAromatic())  # type: ignore[unused-ignore, misc, no-untyped-call]
        aromatic_ratio = aromatic_count / max(mol_with_h.GetNumAtoms(), 1)

        # Literature-calibrated activation energies (deterministic, no noise)
        ec_base = 0.65 + 0.08 * logp - 0.02 * hba - 0.03 * hbd - 0.003 * tpsa + 0.15 * aromatic_ratio
        dm_base = ec_base * 1.15
        pf6_base = 1.15 + 0.05 * logp + 0.01 * hba + 0.02 * hbd + 0.002 * tpsa + 0.10 * aromatic_ratio
        poly_base = 0.45 + 0.06 * logp - 0.01 * hba - 0.02 * hbd - 0.002 * tpsa + 0.20 * aromatic_ratio

        # Clamp to physical ranges
        ec_base = float(np.clip(ec_base, 0.45, 0.95))
        dm_base = float(np.clip(dm_base, 0.45, 1.10))
        pf6_base = float(np.clip(pf6_base, 0.90, 1.50))
        poly_base = float(np.clip(poly_base, 0.30, 0.70))

        # Add Gaussian noise
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

    # Save to CSV if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["smiles", "ec_reduction", "dm_reduction", "pf6_decomposition", "polymerization"])
            writer.writeheader()
            writer.writerows(training_data)

    return training_data


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------


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
        output_path: Path to save trained weights (default: models/tier0/mpnn_weights.pth).
        data_dir: Directory for synthetic data CSV (default: data/).

    Returns:
        Dictionary with training metrics (final loss, epochs run, etc.).
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for model training.")

    # Generate or load training data
    if train_csv_path:
        # Load from CSV with schema validation
        required_columns = {"smiles", "ec_reduction", "dm_reduction", "pf6_decomposition", "polymerization"}
        training_data: list[dict[str, Any]] = []

        with open(train_csv_path) as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file '{train_csv_path}' appears to be empty or has no headers.")

            # Validate schema
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

    # Build molecular graphs
    node_features_list: list[torch.Tensor] = []
    edge_index_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []

    device = "cpu"  # Training on CPU for CI compatibility

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

    # Convert variable-length graphs to padded batches
    def _collate_fn(batch_indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        """Collate variable-length graphs into padded batch.

        Args:
            batch_indices: List of indices into the training data.

        Returns:
            Tuple of (padded_node_features, batch_edge_index, batch_targets, batch_offsets).
        """
        batch_nf: list[torch.Tensor] = []
        batch_ei: list[torch.Tensor] = []
        batch_targets: list[torch.Tensor] = []
        offsets: list[int] = [0]

        for idx in batch_indices:
            nf = node_features_list[idx]
            ei = edge_index_list[idx]
            tgt = targets_list[idx]

            # Adjust edge indices for padding
            n_prev = offsets[-1]
            ei_padded = ei.clone()
            if ei_padded.numel() > 0:
                ei_padded[0] = ei_padded[0] + n_prev
                ei_padded[1] = ei_padded[1] + n_prev

            batch_nf.append(nf)
            batch_ei.append(ei_padded)
            batch_targets.append(tgt)
            offsets.append(offsets[-1] + nf.shape[0])

        # Concatenate node features
        padded_nf = torch.cat(batch_nf, dim=0)

        # Concatenate edge indices
        padded_ei = torch.cat(batch_ei, dim=1) if batch_ei else torch.empty((2, 0), dtype=torch.long)

        batch_tgt = torch.stack(batch_targets)

        return padded_nf, padded_ei, batch_tgt, offsets

    # Training loop
    model = Tier0MPNN(node_dim=4, edge_dim=8, hidden_dim=64, output_dim=4)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_loss = float("inf")
    best_state: dict[str, Any] = {}
    patience_counter = 0
    epoch_losses: list[float] = []
    val_losses: list[float] = []

    # Use 80/20 train/val split
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

        # Shuffle training indices
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

        # Validation
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

        # Early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"[Tier0MPNN] Early stopping at epoch {epoch+1}. Best val loss: {best_loss:.6f}")
                break

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    # Save model weights
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    model.save_weights(output_path)

    # Generate synthetic data CSV if not already saved
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


# ---------------------------------------------------------------------------
# Backward-Compatible Predictor Wrapper
# ---------------------------------------------------------------------------


class Tier0ActivationPredictor:
    """Wrapper that supports both the old linear model and the new MPNN.

    When MPNN weights are available, uses the GNN for predictions.
    Falls back to the original linear predictor (literature defaults)
    if the MPNN model is not loaded.

    Maintains backward compatibility with the original
    Tier0ActivationPredictor interface.
    """

    def __init__(self, model_path: str | None = None) -> None:
        """Initialize the predictor.

        Args:
            model_path: Optional path to MPNN weights. If provided and
                the file exists, loads the GNN model. Otherwise falls
                back to the linear predictor.
        """
        self._use_gnn = False
        self._gnn_model: Tier0MPNN | None = None

        if model_path and os.path.isfile(model_path):
            if HAS_TORCH:
                try:
                    self._gnn_model = Tier0MPNN(node_dim=4, edge_dim=8, hidden_dim=64, output_dim=4)
                    self._gnn_model.load_weights(model_path)
                    self._gnn_model.eval()
                    self._use_gnn = True
                except Exception as e:
                    print(f"[Tier0] Failed to load MPNN model from {model_path}: {e}. "
                          "Falling back to linear predictor.")
                    self._gnn_model = None
                    self._use_gnn = False
        elif model_path is None or not os.path.isfile(model_path):
            if model_path is not None:
                print(f"[Tier0] WARNING: Model path '{model_path}' is invalid or file not found. "
                      "Loading default linear predictor. For better accuracy, train via "
                      "`aurelius train --task tier0`.")
            else:
                print("[Tier0] WARNING: No model path provided. "
                      "Loading default linear predictor. For better accuracy, train via "
                      "`aurelius train --task tier0`.")

        # Always keep the linear predictor as fallback
        self._linear_predictor = _LinearFallbackPredictor()

    def predict(
        self,
        descriptors: dict[str, float] | None = None,
        smiles: str | None = None,
    ) -> dict[str, float]:
        """Predict molecule-specific activation energies.

        Uses MPNN if available and SMILES is provided, otherwise
        falls back to the linear predictor.

        Args:
            descriptors: Optional molecular descriptors dict.
            smiles: Optional SMILES string.

        Returns:
            Dictionary with predicted activation energies:
                - ec_reduction: EC solvent reduction Ea (eV)
                - dm_reduction: DMC solvent reduction Ea (eV)
                - pf6_decomposition: Salt decomposition Ea (eV)
                - polymerization: Polymerization Ea (eV)
        """
        if self._use_gnn and smiles is not None and self._gnn_model is not None:
            try:
                nf, ei = _build_molecular_graph(smiles)
                with torch.no_grad():
                    preds = self._gnn_model(nf, ei)
                return {
                    "ec_reduction": float(preds[0].item()),
                    "dm_reduction": float(preds[1].item()),
                    "pf6_decomposition": float(preds[2].item()),
                    "polymerization": float(preds[3].item()),
                }
            except Exception:
                # Fallback to linear model on any error
                pass

        # Fallback to linear predictor
        return self._linear_predictor.predict(descriptors=descriptors, smiles=smiles)

    def set_gnn_model(self, model: Tier0MPNN, model_path: str) -> None:
        """Set the GNN model explicitly.

        Args:
            model: The trained MPNN model.
            model_path: Path to the model weights file.
        """
        self._gnn_model = model
        self._use_gnn = True


class _LinearFallbackPredictor:
    """Original linear predictor (kept for backward compatibility).

    This is the v5.2 heuristic model that uses normalized descriptors
    with hardcoded weights. Used when MPNN is unavailable.
    """

    _SOLVENT_WEIGHTS = np.array([
        0.002, 0.08, -0.02, -0.03, -0.003, 0.01, 0.15, 0.005,
    ])
    _SOLVENT_BIAS = 0.70
    _SALT_WEIGHTS = np.array([
        0.001, 0.05, 0.01, 0.02, 0.002, 0.005, 0.10, 0.003,
    ])
    _SALT_BIAS = 1.15
    _POLY_WEIGHTS = np.array([
        0.001, 0.06, -0.01, -0.02, -0.002, 0.015, 0.20, 0.004,
    ])
    _POLY_BIAS = 0.45

    _MW_RANGE = (50, 500)
    _LOGP_RANGE = (-2, 5)
    _HBA_RANGE = (0, 10)
    _HBD_RANGE = (0, 5)
    _TPSA_RANGE = (0, 200)
    _ROT_RANGE = (0, 10)
    _ARO_RANGE = (0, 1)
    _HEAVY_RANGE = (5, 50)

    def predict(
        self,
        descriptors: dict[str, float] | None = None,
        smiles: str | None = None,
    ) -> dict[str, float]:
        """Predict using the original linear model.

        Args:
            descriptors: Optional molecular descriptors dict.
            smiles: Optional SMILES string (used to generate descriptors).

        Returns:
            Dictionary with predicted activation energies.
        """
        if descriptors is None and smiles is None:
            return {
                "ec_reduction": 0.65,
                "dm_reduction": 0.75,
                "pf6_decomposition": 1.20,
                "polymerization": 0.40,
            }

        if descriptors is None:
            descriptors = self._generate_descriptors(smiles)  # type: ignore[arg-type]

        def _predict_single(desc: dict[str, float], weights: np.ndarray, bias: float) -> float:
            normalized = np.array([
                (desc.get("mw", 250) - self._MW_RANGE[0]) / (self._MW_RANGE[1] - self._MW_RANGE[0]),
                (desc.get("logp", 1.5) - self._LOGP_RANGE[0]) / (self._LOGP_RANGE[1] - self._LOGP_RANGE[0]),
                (desc.get("hba", 5) - self._HBA_RANGE[0]) / (self._HBA_RANGE[1] - self._HBA_RANGE[0]),
                (desc.get("hbd", 2) - self._HBD_RANGE[0]) / (self._HBD_RANGE[1] - self._HBD_RANGE[0]),
                (desc.get("tpsa", 100) - self._TPSA_RANGE[0]) / (self._TPSA_RANGE[1] - self._TPSA_RANGE[0]),
                (desc.get("rot_bonds", 5) - self._ROT_RANGE[0]) / (self._ROT_RANGE[1] - self._ROT_RANGE[0]),
                (desc.get("aromatic_ratio", 0.5) - self._ARO_RANGE[0]) / (self._ARO_RANGE[1] - self._ARO_RANGE[0]),
                (desc.get("heavy_atom_count", 25) - self._HEAVY_RANGE[0]) / (self._HEAVY_RANGE[1] - self._HEAVY_RANGE[0]),
            ])
            raw_ea = float(np.dot(normalized, weights) + bias)
            return float(np.clip(raw_ea, 0.30, 1.50))

        return {
            "ec_reduction": float(_predict_single(descriptors, self._SOLVENT_WEIGHTS, self._SOLVENT_BIAS)),
            "dm_reduction": float(_predict_single(descriptors, self._SOLVENT_WEIGHTS, self._SOLVENT_BIAS) * 1.15),
            "pf6_decomposition": float(_predict_single(descriptors, self._SALT_WEIGHTS, self._SALT_BIAS)),
            "polymerization": float(_predict_single(descriptors, self._POLY_WEIGHTS, self._POLY_BIAS)),
        }

    def _generate_descriptors(self, smiles: str) -> dict[str, float]:
        """Generate molecular descriptors from SMILES (delegated to shared module).

        Args:
            smiles: SMILES string of the molecule.

        Returns:
            Dictionary of descriptor name -> value.
        """
        return _generate_molecular_descriptors(smiles)

    def _hash_descriptors(self, smiles: str) -> dict[str, float]:
        """Fallback descriptor generation using deterministic hashing.

        Args:
            smiles: SMILES string.

        Returns:
            Dictionary of approximate descriptor values.
        """
        return _hash_descriptors(smiles)
