"""Tier 0: MPNN (Message Passing Neural Network) model definitions.

Implements a lightweight graph neural network for activation energy
prediction in battery electrolyte screening.

Architecture:
    - Node features: atomic number, degree, formal charge, aromaticity
    - Message passing: 2-layer edge-based MP with torch.scatter_add aggregation
    - Readout: MLP over pooled node embeddings
    - Output: 4 activation energies (eV) for EC/DMC reduction, PF6 decomposition, polymerization

References:
    Gilmer, J. et al. "Neural Message Passing for Quantum Chemistry." ICML 2017.
    Wu, Z. et al. "Molecular Graph Convolutions: Moving Beyond Fingerprints." JMLR 2021.
"""

from __future__ import annotations

import json
import os
from typing import Any

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


if HAS_TORCH:

    class MPNNEdgeBlock(nn.Module):
        """2-layer message passing block for molecular graphs.

        Implements edge-based message passing with:
        - Edge feature computation from source/target node features + edge index
        - 2-layer MLP for edge message generation
        - Aggregation via torch.scatter_add (MPS-compatible, no torch_scatter)
        - Node update via residual connection + LayerNorm
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

            self.edge_input_dim = node_dim * 2 + edge_dim
            self.edge_mlp = nn.Sequential(
                nn.Linear(self.edge_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.node_input_dim = node_dim + hidden_dim
            self.node_mlp = nn.Sequential(
                nn.Linear(self.node_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, node_dim),
            )

            self.norm = nn.LayerNorm(node_dim)

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

            src_idx = edge_index[0]
            tgt_idx = edge_index[1]

            src_features = torch.index_select(node_features, 0, src_idx)
            tgt_features = torch.index_select(node_features, 0, tgt_idx)

            if edge_features is None:
                edge_features = torch.cat([src_features, tgt_features], dim=-1)

            messages = self.edge_mlp(edge_features)

            aggregated = torch.zeros(n_nodes, self.hidden_dim, device=node_features.device)
            aggregated.scatter_add_(0, src_idx.unsqueeze(1).expand(-1, self.hidden_dim), messages)

            node_input = torch.cat([node_features, aggregated], dim=-1)
            node_updates = self.node_mlp(node_input)

            return self.norm(node_features + node_updates)  # type: ignore[no-any-return]

    class MPNNReadoutMLP(nn.Module):
        """Readout MLP for MPNN.

        Takes pooled node embeddings and produces output predictions
        via a multi-layer perceptron.
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
                return self.network(pooled)  # type: ignore[no-any-return]
            return self.network(pooled)  # type: ignore[no-any-return]

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

            self.edge_input_dim = node_dim * 2 + edge_dim
            self.edge_transform = nn.Sequential(
                nn.Linear(self.edge_input_dim, hidden_dim),
                nn.ReLU(),
            )

            self.mp_layers = nn.ModuleList([
                MPNNEdgeBlock(node_dim, edge_dim, hidden_dim)
                for _ in range(2)
            ])

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
                return torch.zeros(self.readout.network[-1].out_features, device=node_features.device)  # type: ignore[call-overload, no-any-return]

            if edge_index.shape[1] == 0:
                pooled = node_features.sum(dim=0)
                return self.readout(pooled)  # type: ignore[no-any-return]

            src_idx = edge_index[0]
            tgt_idx = edge_index[1]
            src_features = torch.index_select(node_features, 0, src_idx)
            tgt_features = torch.index_select(node_features, 0, tgt_idx)
            initial_edge_features = torch.cat([src_features, tgt_features], dim=-1)
            edge_features = self.edge_transform(initial_edge_features)

            h = node_features
            for mp_layer in self.mp_layers:
                h = mp_layer(h, edge_index, edge_features)

            pooled = h.sum(dim=0)

            return self.readout(pooled)  # type: ignore[no-any-return]

        def save_weights(self, path: str) -> None:
            """Save model weights to file along with metadata.

            Args:
                path: File path to save weights (state dict).
            """
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            torch.save(self.state_dict(), path)

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

            Args:
                path: File path to load weights from.
            """
            try:
                import importlib
                current_version = importlib.metadata.version("aurelius")
            except Exception:
                current_version = "unknown"

            meta_path = path.rsplit(".", 1)[0] + "_metadata.json"
            metadata: dict[str, Any] = {}
            if os.path.isfile(meta_path):
                with open(meta_path) as f:
                    metadata = json.load(f)

            saved_version = metadata.get("model_version", "unknown")
            if saved_version != "unknown" and current_version != "unknown" and saved_version != current_version:
                print(
                        f"[Tier0MPNN] WARNING: Model version ({saved_version}) does not match "
                        f"installed package version ({current_version}). "
                        "Consider retraining with `aurelius train --task tier0`."
                    )

            state_dict = torch.load(path, map_location="cpu", weights_only=True)

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
