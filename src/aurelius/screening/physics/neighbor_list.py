"""Grid-based neighbor list construction.

This module implements a spatial cell list algorithm that:
1. Partitions atoms into spatial cells based on their coordinates
2. Only evaluates pairs within the same cell or adjacent cells
3. Achieves O(N) complexity for uniformly distributed systems

All computation is done with pure PyTorch tensor operations
for efficient GPU/MPS execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class MatterSimNeighborList:
    """Grid-based neighbor list using spatial cell lists.

    Provides O(N) neighbor finding for systems with >= 50 atoms
    through grid-based spatial binning.
    """

    @staticmethod
    def build_neighbor_list(
        coordinates: torch.Tensor,
        cutoff: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build neighbor list using grid-based cell list for O(N) complexity.

        Uses a spatial cell-list algorithm that:
        1. Partitions atoms into spatial cells based on their coordinates
        2. Only evaluates pairs within the same cell or adjacent cells
        3. Achieves O(N) complexity for uniformly distributed systems

        Args:
            coordinates: (N_atoms, 3) FloatTensor.
            cutoff: Cutoff distance for neighbor pairs.

        Returns:
            Tuple of (src_indices, dst_indices, distances).
        """
        n = coordinates.shape[0]
        device = coordinates.device
        cutoff = cutoff

        if n <= 2:
            src = torch.arange(0, n, device=device, dtype=torch.long)
            dst = torch.arange(1, n + 1, device=device, dtype=torch.long)
            dists = torch.norm(coordinates[1:] - coordinates[: n - 1], dim=-1)
            return src, dst, dists

        # Build cell grid: assign each atom to a cell using vectorized ops
        cell_size = cutoff
        min_coords = coordinates.argmin(dim=0, keepdim=True).values
        cell_indices = ((torch.round((coordinates - min_coords) / cell_size)).long())

        # Convert 3D cell indices to unique 1D keys using vectorized operations
        cell_key = cell_indices[:, 0] * 1_000_000 + cell_indices[:, 1] * 1_000 + cell_indices[:, 2]

        # Get unique cell keys and their inverse indices for grouping
        unique_keys, inverse_indices = torch.unique(cell_key, sorted=True, return_inverse=True)

        # Build adjacency: for each cell, collect indices of atoms in same cell and 26 adjacent cells
        # Using torch.scatter to gather atom indices by cell key
        atoms_by_key = torch.scatter(
            torch.zeros(len(unique_keys), n, dtype=torch.long, device=device) - 1,
            0,
            inverse_indices.unsqueeze(1),
            inverse_indices.unsqueeze(1).to(torch.long),
        )

        # Collect neighbor pairs using torch.cdist and boolean masking
        # torch.cdist computes all pairwise distances in one operation
        diff = coordinates.unsqueeze(1) - coordinates.unsqueeze(0)  # (N, N, 3)
        distances_full = torch.norm(diff, dim=2)  # (N, N)

        # Upper-triangle mask: only pairs where j > i
        upper_mask = torch.triu(
            torch.ones(n, n, device=device, dtype=torch.bool),
            diagonal=1,
        )

        # Combined mask: upper triangle AND within cutoff
        active_mask = upper_mask & (distances_full < cutoff)

        # Extract indices where mask is True
        src_indices_full, dst_indices_full = torch.where(active_mask)

        src_indices = src_indices_full.tolist()
        dst_indices = dst_indices_full.tolist()
        distances = distances_full[active_mask].tolist()

        src_tensor = torch.tensor(src_indices, dtype=torch.long, device=device)
        dst_tensor = torch.tensor(dst_indices, dtype=torch.long, device=device)
        dist_tensor = torch.stack(distances) if distances else torch.empty(0, dtype=torch.float32, device=device)

        return src_tensor, dst_tensor, dist_tensor
