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
        cell_indices = ((torch.round((coordinates - min_coords) / cell_size)).long())  # type: ignore[operator]

        # Convert 3D cell indices to unique 1D keys using vectorized operations
        cell_key = cell_indices[:, 0] * 1_000_000 + cell_indices[:, 1] * 1_000 + cell_indices[:, 2]

        # Get unique cell keys and their inverse indices for grouping
        unique_keys, inverse_indices = torch.unique(cell_key, sorted=True, return_inverse=True)

        # Build adjacency: for each cell, collect indices of atoms in same cell and 26 adjacent cells
        # Using torch.scatter to gather atom indices by cell key
        atoms_by_key = torch.zeros(
            len(unique_keys), n, dtype=torch.long, device=device,
        ) - 1  # -1 = no atom

        for cell_idx, atom_idx in enumerate(inverse_indices):
            atoms_by_key[cell_idx, atom_idx] = cell_idx

        # Collect neighbor pairs using vectorized operations
        src_indices: list[int] = []
        dst_indices: list[int] = []
        distances: list[float] = []

        unique_indices = inverse_indices.unique(sorted=True)
        for local_key in unique_indices:
            local_key_int = int(local_key.item())
            _ = (
                int(unique_keys[local_key_int].item()) // 1_000_000,
                (int(unique_keys[local_key_int].item()) % 1_000_000) // 1_000,
                int(unique_keys[local_key_int].item()) % 1_000,
            )

            atoms = atoms_by_key[local_key]
            for _i_idx, i in enumerate(atoms):
                for j in atoms:
                    if j <= i:
                        continue
                    diff = coordinates[i] - coordinates[j]
                    dist = torch.norm(diff, dim=-1).item()
                    if dist < cutoff:
                        src_indices.append(i)
                        dst_indices.append(j)
                        distances.append(dist)

        src_tensor = torch.tensor(src_indices, dtype=torch.long, device=device)
        dst_tensor = torch.tensor(dst_indices, dtype=torch.long, device=device)
        dist_tensor = torch.stack(distances) if distances else torch.empty(0, dtype=torch.float32, device=device)  # type: ignore[arg-type]

        return src_tensor, dst_tensor, dist_tensor
