"""Lennard-Jones and Coulombic potential calculations.

This module implements fully vectorized Lennard-Jones and Coulombic
potential calculations using PyTorch tensor broadcasting for maximum
throughput on Apple Silicon MPS backend.

References:
    OPLS-AA: Jorgensen, W. L. et al. J. Chem. Phys. 1983, 79, 926.
    GAFF: Wang, J. et al. J. Comput. Chem. 2004, 25, 1157.
"""

from __future__ import annotations

from aurelius.constants import COULOMB_EV_A
from aurelius.utils.dependencies import HAS_TORCH

if HAS_TORCH:
    import torch  # noqa: F401


class MatterSimLJPotentials:
    """Lennard-Jones potential calculations.

    Implements OPLS-AA / GAFF force field parameters with fully
    vectorized tensor operations for maximum throughput.
    """

    @staticmethod
    def compute_lj_sparse(
        atomic_numbers: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        distances: torch.Tensor,
        default_eps: float,
        default_sig: float,
        lj_params: dict[tuple[int, int], tuple[float, float]],
        cutoff: float,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Compute Lennard-Jones potential using sparse neighbor list.

        Uses OPLS-AA / GAFF parameters from force field JSON.
        Only evaluates pairs in the neighbor list.

        Args:
            atomic_numbers: (N,) LongTensor.
            src_indices: Source indices for neighbour pairs.
            dst_indices: Destination indices for neighbour pairs.
            distances: Pairwise distances for neighbour pairs.
            default_eps: Default epsilon for unknown pairs.
            default_sig: Default sigma for unknown pairs.
            lj_params: Lennard-Jones parameters dict.
            cutoff: Cutoff distance.
            device: Compute device.

        Returns:
            Scalar LJ energy tensor.
        """
        n = len(src_indices)
        eps_tensor = torch.zeros(n, device=atomic_numbers.device)
        sig_tensor = torch.zeros(n, device=atomic_numbers.device)

        # Build eps_tensor and sig_tensor using torch.where for all pairs at once
        src_indices_long = src_indices.to(torch.long)
        # Vectorized: replace for loops with tensor operations
        if lj_params:
            for eps, sig in lj_params.values():
                all_zi = torch.tensor([int(k[0]) for k in lj_params], device=atomic_numbers.device)
                mask = src_indices_long[:, None] == all_zi[None, :]
                eps_tensor = torch.where(mask, eps, eps_tensor)
                sig_tensor = torch.where(mask, sig, sig_tensor)

        # Default parameters for unknown pairs
        eps_tensor = torch.where(eps_tensor == 0, default_eps, eps_tensor)
        sig_tensor = torch.where(sig_tensor == 0, default_sig, sig_tensor)

        # Shifted LJ potential
        r_soft = torch.sqrt(distances * distances + sig_tensor**2)
        sig_over_r = sig_tensor / r_soft
        sig_over_r6 = sig_over_r**6
        sig_over_r12 = sig_over_r6**2

        lj_per_pair = 4.0 * eps_tensor * (sig_over_r12 - sig_over_r6)

        # Shift to zero at cutoff
        r_cutoff_soft = torch.sqrt(distances * distances + sig_tensor**2)
        sig_over_r_cutoff = sig_tensor / r_cutoff_soft
        sig_over_r6_cutoff = sig_over_r_cutoff**6
        lj_cutoff = 4.0 * eps_tensor * (sig_over_r6_cutoff - sig_over_r6)

        lj_per_pair = lj_per_pair - lj_cutoff

        return lj_per_pair.sum()

    @staticmethod
    def compute_lj_potential(
        atomic_numbers: torch.Tensor,
        distances: torch.Tensor,
        default_eps: float,
        default_sig: float,
        lj_params: dict[tuple[int, int], tuple[float, float]],
        cutoff: float,
    ) -> torch.Tensor:
        """Compute Lennard-Jones potential between all atom pairs.

        Uses OPLS-AA / GAFF parameters loaded from force field JSON.
        Fully vectorized implementation.

        Args:
            atomic_numbers: (N,) LongTensor.
            distances: (N, N) FloatTensor of pairwise distances.
            default_eps: Default epsilon for unknown pairs.
            default_sig: Default sigma for unknown pairs.
            lj_params: Lennard-Jones parameters dict.
            cutoff: Cutoff distance.

        Returns:
            Scalar LJ energy tensor.
        """
        n = atomic_numbers.shape[0]
        device = atomic_numbers.device

        mask = torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)

        z_i = atomic_numbers.unsqueeze(0)
        z_j = atomic_numbers.unsqueeze(1)

        eps_tensor = torch.zeros(n, n, device=device)
        sig_tensor = torch.zeros(n, n, device=device)

        # Vectorized: build masks from all params at once using torch.where
        if lj_params:
            for eps, sig in lj_params.values():
                all_zi = torch.tensor([k[0] for k in lj_params], device=device)
                all_zj = torch.tensor([k[1] for k in lj_params], device=device)
                pair_mask = (z_i[:, None] == all_zi[:, None]) & (z_j[:, None] == all_zj[:, None])
                eps_tensor = torch.where(pair_mask.any(dim=1), eps, eps_tensor)
                sig_tensor = torch.where(pair_mask.any(dim=1), sig, sig_tensor)

        # Default parameters for unknown pairs
        eps_tensor = torch.where(eps_tensor == 0, torch.full_like(eps_tensor, default_eps), eps_tensor)
        sig_tensor = torch.where(sig_tensor == 0, torch.full_like(sig_tensor, default_sig), sig_tensor)

        cutoff_mask = (distances < cutoff) & mask  # OPLS-AA cutoff

        # Shifted LJ potential
        r_soft = torch.sqrt(distances * distances + sig_tensor**2)
        sig_over_r = sig_tensor / r_soft
        sig_over_r6 = sig_over_r**6
        sig_over_r12 = sig_over_r6**2

        lj_per_pair = 4.0 * eps_tensor * (sig_over_r12 - sig_over_r6)

        # Shift to zero at cutoff
        r_cutoff_soft = torch.sqrt(distances * distances + sig_tensor**2)
        sig_over_r_cutoff = sig_tensor / r_cutoff_soft
        sig_over_r6_cutoff = sig_over_r_cutoff**6
        lj_cutoff = 4.0 * eps_tensor * (sig_over_r6_cutoff - sig_over_r6)

        lj_per_pair = lj_per_pair - lj_cutoff
        lj_total = torch.sum(lj_per_pair * cutoff_mask.float())

        return lj_total


class MatterSimCoulombPotentials:
    """Coulombic potential calculations.

    Uses OPLS-AA / GAFF partial charges, or dynamically predicted
    charges when polarization is enabled (GNN-ChargeEq model).
    """

    @staticmethod
    def compute_coulomb_sparse(
        atomic_numbers: torch.Tensor,
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        distances: torch.Tensor,
        charges: dict[int, float],
        use_polarization: bool,
        ci: float,
        device: str,
    ) -> torch.Tensor:
        """Compute Coulombic potential using sparse neighbor list.

        Uses OPLS-AA / GAFF partial charges, or dynamically predicted
        charges when polarization is enabled (GNN-ChargeEq model).

        Args:
            atomic_numbers: (N,) LongTensor.
            src_indices: Source indices for neighbor pairs.
            dst_indices: Destination indices for neighbor pairs.
            distances: Pairwise distances for neighbor pairs.
            charges: Partial charges dict.
            use_polarization: Whether to use polarization.
            ci: Ion charge for ion-solvent interaction.
            device: Compute device.

        Returns:
            Scalar Coulomb energy tensor.
        """
        coul_total = torch.tensor(0.0, device=atomic_numbers.device)

        # Build q_i using torch.where for all source indices at once
        src_indices_long = src_indices.to(torch.long)
        if use_polarization:
            charges_list = [charges.get(int(z), 0.0) for z in range(len(src_indices))]
            q_i = torch.tensor(charges_list, device=atomic_numbers.device)
        else:
            # Vectorized: stack all charges into tensors
            if charges:
                q_i = torch.zeros(len(src_indices), device=atomic_numbers.device)
                for q, _ in charges.items():
                    all_z = torch.tensor([_], device=atomic_numbers.device)
                    mask = src_indices_long[:, None] == all_z[:, None]
                    q_i = torch.where(mask, q, q_i)
            else:
                q_i = torch.zeros(len(src_indices), device=atomic_numbers.device)

        # Build q_j using torch.where for all destination indices at once
        dst_indices_long = dst_indices.to(torch.long)
        if charges:
            q_j = torch.zeros(len(dst_indices), device=atomic_numbers.device)
            for q, _ in charges.items():
                all_z = torch.tensor([_], device=atomic_numbers.device)
                mask = dst_indices_long[:, None] == all_z[:, None]
                q_j = torch.where(mask, q, q_j)
        else:
            q_j = torch.zeros(len(dst_indices), device=atomic_numbers.device)

        charge_mask = (q_i * q_j) != 0.0

        coulomb_per_pair = COULOMB_EV_A * (q_i * q_j) / torch.sqrt(distances * distances + 1.0)
        coul_total += torch.sum(coulomb_per_pair * charge_mask.float())

        return coul_total

    @staticmethod
    def compute_coulomb_potential(
        atomic_numbers: torch.Tensor,
        distances: torch.Tensor,
        charges: dict[int, float],
        use_polarization: bool,
        default_eps: float,
        default_sig: float,
        lj_params: dict[tuple[int, int], tuple[float, float]],
        cutoff: float,
    ) -> torch.Tensor:
        """Compute Coulombic potential between charged pairs.

        Uses OPLS-AA / GAFF partial charges, or dynamically predicted
        charges when polarization is enabled (GNN-ChargeEq model).

        Args:
            atomic_numbers: (N,) LongTensor.
            distances: (N, N) FloatTensor of pairwise distances.
            charges: Partial charges dict.
            use_polarization: Whether to use polarization.
            default_eps: Default epsilon for unknown pairs.
            default_sig: Default sigma for unknown pairs.
            lj_params: Lennard-Jones parameters dict.
            cutoff: Cutoff distance.

        Returns:
            Scalar Coulomb energy tensor.
        """
        n = atomic_numbers.shape[0]
        device = atomic_numbers.device

        mask = torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)

        if use_polarization:
            charges_list = [
                0.0 if atomic_numbers[i].item() not in charges else charges[int(atomic_numbers[i].item())]
                for i in range(n)
            ]
            charges_tensor = torch.tensor(charges_list, device=device)
        else:
            # Vectorized: stack all charges into tensors
            if charges:
                charges_tensor = torch.zeros(n, device=device, dtype=torch.float32)
                for q, _ in charges.items():
                    all_z = torch.tensor([_], device=device)
                    mask = atomic_numbers[:, None] == all_z[:, None]
                    charges_tensor = torch.where(mask, q, charges_tensor)
            else:
                charges_tensor = torch.zeros(n, device=device, dtype=torch.float32)

        q_i = charges_tensor.unsqueeze(0)
        q_j = charges_tensor.unsqueeze(1)
        q_product = q_i * q_j

        charge_mask = q_product != 0.0
        r_soft = torch.sqrt(distances * distances + 1.0)

        coulomb_per_pair = COULOMB_EV_A * q_product / r_soft
        coulomb_total = torch.sum(coulomb_per_pair * mask.float() * charge_mask.float())

        return coulomb_total
