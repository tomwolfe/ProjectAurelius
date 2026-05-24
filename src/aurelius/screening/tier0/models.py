"""Tier 0: MPNN (Message Passing Neural Network) model definitions.

Implements a lightweight graph neural network for activation energy
prediction in battery electrolyte screening.

Architecture:
    - Node features: atomic number, degree, formal charge, aromaticity
    - Message passing: 2-layer edge-based MP with scatter_add aggregation
    - Readout: MLP over pooled node embeddings
    - Output: 4 activation energies (eV) for EC/DMC reduction, PF6 decomposition, polymerization

References:
    Gilmer, J. et al. "Neural Message Passing for Quantum Chemistry." ICML 2017.
    Wu, Z. et al. "Molecular Graph Convolutions: Moving Beyond Fingerprints." JMLR 2021.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol, runtime_checkable

from aurelius.utils.dependencies import HAS_TORCH

__all__ = [
    "HAS_TORCH",
    "ModelBackend",
    "ModelFactory",
    "PyTorchBackend",
]

# ---------------------------------------------------------------------------
# Protocol / Strategy Pattern
# ---------------------------------------------------------------------------


@runtime_checkable
class ModelBackend(Protocol):
    """Protocol defining the interface for all model backends.

    All backends must implement:
    - __call__ or forward: Forward pass
    - parameters: Return model parameters
    - save_weights: Save model weights to disk
    - load_weights: Load model weights from disk
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    def parameters(self) -> list[Any]: ...

    def save_weights(self, path: str) -> None: ...

    def load_weights(self, path: str) -> None: ...


# ---------------------------------------------------------------------------
# PyTorch Backend
# ---------------------------------------------------------------------------

if HAS_TORCH:
    _torch: Any = None
    _torch_nn: Any = None
    try:
        import torch  # type: ignore[import-not-found, unused-ignore]
        import torch.nn as torch_nn  # type: ignore[import-not-found, unused-ignore]

        _torch = torch
        _torch_nn = torch_nn
    except Exception:
        pass

    class PyTorchBackend:
        """PyTorch-based MPNN model for activation energy prediction.

        Architecture:
            1. Node feature projection
            2. 2-layer message passing (edge-based)
            3. Global pooling (sum over nodes)
            4. Readout MLP for 4 activation energy predictions

        This class provides the same functionality as the original
        Tier0MPNN class, but as a clean backend implementation
        following the Strategy Pattern.
        """

        def __init__(
            self,
            node_dim: int = 4,
            edge_dim: int = 0,
            hidden_dim: int = 64,
            output_dim: int = 4,
        ) -> None:
            super().__init__()
            self.node_dim = node_dim
            self.edge_dim = edge_dim
            self.hidden_dim = hidden_dim

            self.node_proj = _torch_nn.Linear(node_dim, hidden_dim)

            self.edge_input_dim = 2 * hidden_dim
            self.edge_transform = _torch_nn.Sequential(
                _torch_nn.Linear(self.edge_input_dim, hidden_dim),
                _torch_nn.ReLU(),
            )

            self.mp_layers = _torch_nn.ModuleList([
                _MPNNEdgeBlockBackend(hidden_dim, edge_dim, hidden_dim)
                for _ in range(2)
            ])

            self.readout = _MPNNReadoutMLPBackend(
                input_dim=hidden_dim, output_dim=output_dim, hidden_dim=128
            )

            self._init_weights()

        def _init_weights(self) -> None:
            for module in self.modules():
                if isinstance(module, _torch_nn.Linear):
                    _torch_nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        _torch_nn.init.zeros_(module.bias)

        def __call__(self, node_features: Any, edge_index: Any) -> Any:
            """Forward pass of the MPNN.

            Args:
                node_features: (N_nodes, node_dim) tensor.
                edge_index: (2, N_edges) tensor.

            Returns:
                Predicted activation energies (output_dim,).
            """
            n_nodes = node_features.shape[0]

            if n_nodes == 0:
                return _torch.zeros(
                    self.readout.network[-1].out_features,
                    device=node_features.device,
                )

            h = self.node_proj(node_features)

            if edge_index.shape[1] == 0:
                pooled = h.sum(dim=0)
                return self.readout(pooled)

            src_idx = edge_index[0]
            tgt_idx = edge_index[1]
            src_features = _torch.index_select(h, 0, src_idx)
            tgt_features = _torch.index_select(h, 0, tgt_idx)
            initial_edge_features = _torch.cat(
                [src_features, tgt_features], dim=-1
            )
            edge_features = self.edge_transform(initial_edge_features)

            for mp_layer in self.mp_layers:
                h = mp_layer(h, edge_index, edge_features)

            pooled = h.sum(dim=0)
            return self.readout(pooled)

        def parameters(self) -> list[Any]:
            return list(self.parameters())

        def save_weights(self, path: str) -> None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            _torch.save(self.state_dict(), path)

            meta_path = path.rsplit(".", 1)[0] + "_metadata.json"
            shape_info: dict[str, list[int]] = {}
            for name, tensor in self.state_dict().items():
                shape_info[name] = list(tensor.shape)

            meta = {
                "model_version": _importlib_metadata_version("aurelius"),
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
            try:
                current_version = _importlib_metadata_version("aurelius")
            except Exception:
                current_version = "unknown"

            meta_path_path = path.rsplit(".", 1)[0] + "_metadata.json"
            metadata: dict[str, Any] = {}
            if os.path.isfile(meta_path_path):
                with open(meta_path) as f:
                    metadata = json.load(f)

            saved_version = metadata.get("model_version", "unknown")
            if saved_version != "unknown" and current_version != "unknown" and saved_version != current_version:
                print(
                    f"[Tier0MPNN] WARNING: Model version ({saved_version}) does not match "
                    f"installed package version ({current_version}). "
                    "Consider retraining with `aurelius train --task tier0`."
                )

            state_dict = _torch.load(path, map_location="cpu", weights_only=True)

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

    # Import helper for metadata
    import importlib.metadata as _importlib_metadata

    def _importlib_metadata_version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except Exception:
            return "unknown"

    # ------------------------------------------------------------------
    # Internal backend classes (not exported)
    # ------------------------------------------------------------------

    class _MPNNEdgeBlockBackend:
        """2-layer message passing block for molecular graphs (PyTorch backend)."""

        def __init__(self, node_dim: int = 4, edge_dim: int = 0, hidden_dim: int = 64) -> None:
            super().__init__()
            self.node_dim = node_dim
            self.edge_dim = edge_dim
            self.hidden_dim = hidden_dim

            self.edge_proj = _torch_nn.Linear(2 * node_dim, hidden_dim)
            self.edge_input_dim = hidden_dim
            self.edge_mlp = _torch_nn.Sequential(
                _torch_nn.Linear(self.edge_input_dim, hidden_dim),
                _torch_nn.ReLU(),
                _torch_nn.Linear(hidden_dim, hidden_dim),
            )

            self.node_input_dim = node_dim + hidden_dim
            self.node_mlp = _torch_nn.Sequential(
                _torch_nn.Linear(self.node_input_dim, hidden_dim),
                _torch_nn.ReLU(),
                _torch_nn.Linear(hidden_dim, node_dim),
            )

            self.norm = _torch_nn.LayerNorm(node_dim)

            self._init_weights()

        def _init_weights(self) -> None:
            for module in self.modules():
                if isinstance(module, _torch_nn.Linear):
                    _torch_nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        _torch_nn.init.zeros_(module.bias)

        def __call__(
            self,
            node_features: Any,
            edge_index: Any,
            edge_features: Any | None = None,
        ) -> Any:
            n_nodes = node_features.shape[0]
            n_edges = edge_index.shape[1]

            if n_edges == 0:
                return node_features

            src_idx = edge_index[0]
            tgt_idx = edge_index[1]

            src_features = _torch.index_select(node_features, 0, src_idx)
            tgt_features = _torch.index_select(node_features, 0, tgt_idx)

            if edge_features is None:
                raw_edge = _torch.cat([src_features, tgt_features], dim=-1)
                edge_features = _torch.relu(self.edge_proj(raw_edge))

            messages = self.edge_mlp(edge_features)

            aggregated = _torch.zeros(n_nodes, self.hidden_dim, device=node_features.device)
            aggregated.scatter_add_(0, src_idx.unsqueeze(1).expand(-1, self.hidden_dim), messages)

            node_input = _torch.cat([node_features, aggregated], dim=-1)
            node_updates = self.node_mlp(node_input)

            return self.norm(node_features + node_updates)

        def parameters(self) -> list[Any]:
            return list(self.parameters())

    class _MPNNReadoutMLPBackend:
        """Readout MLP for MPNN (PyTorch backend)."""

        def __init__(self, input_dim: int = 64, output_dim: int = 4, hidden_dim: int = 128) -> None:
            super().__init__()
            self.network = _torch_nn.Sequential(
                _torch_nn.Linear(input_dim, hidden_dim),
                _torch_nn.ReLU(),
                _torch_nn.Dropout(0.1),
                _torch_nn.Linear(hidden_dim, hidden_dim // 2),
                _torch_nn.ReLU(),
                _torch_nn.Dropout(0.1),
                _torch_nn.Linear(hidden_dim // 2, output_dim),
            )
            self._init_weights()

        def _init_weights(self) -> None:
            for module in self.modules():
                if isinstance(module, _torch_nn.Linear):
                    _torch_nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        _torch_nn.init.zeros_(module.bias)

        def __call__(self, pooled: Any) -> Any:
            if pooled.dim() == 1:
                return self.network(pooled)
            return self.network(pooled)

        def parameters(self) -> list[Any]:
            return list(self.parameters())

else:
    class PyTorchBackend:
        """Stub when PyTorch is unavailable."""

        def __init__(
            self,
            node_dim: int = 4,
            edge_dim: int = 0,
            hidden_dim: int = 64,
            output_dim: int = 4,
        ) -> None:
            raise RuntimeError(
                "PyTorch is required for PyTorchBackend. Install with: pip install torch"
            )

        __call__: Any
        parameters: Any
        save_weights: Any
        load_weights: Any


def ModelFactory() -> PyTorchBackend:
    """Return a PyTorch backend instance for Tier 0 MPNN.

    Returns:
        A PyTorchBackend instance.
    """
    return PyTorchBackend()
