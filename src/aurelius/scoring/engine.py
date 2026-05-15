"""Phase 4: Revised Aurelius Score v5.1 (S_A_v5.1).

Formula:
    S_A = 0.3(σ) + 0.2(E_des_barrier) + 0.2(SEI Homogeneity)
          + 0.2(MX_Synthesis_Score) - 0.1(GWP)

Where:
    σ          = MLX-NA filter confidence score (Tier 1)
    E_des_barrier = Normalized desolvation path integral (Tier 2)
    SEI Homogeneity = Interface homogeneity score (Tier 3)
    MX_Synthesis_Score = Automated lab compatibility metric (2026 AI for Materials)
    GWP        = Global Warming Potential penalty
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from aurelius.types import (
    AureliusScoreResult,
    DesolvationPathResult,
    GCMDTwinResult,
    MLXFilterResult,
    MoleculeInput,
    SEIEvolution,
    Tier2Result,
)


class AureliusScoringEngine:
    """Revised Aurelius Score v5.1 calculation engine.

    Computes S_A_v5.1 from the three-tier screening pipeline results.
    All result types are imported from the centralized types module
    to eliminate circular imports.
    """

    def __init__(
        self,
        weight_sigma: float = 0.3,
        weight_desolvation: float = 0.2,
        weight_sei_homogeneity: float = 0.2,
        weight_mx_synthesis: float = 0.2,
        weight_gwp: float = 0.1,
        viability_threshold: float = 65.0,
    ) -> None:
        self.weights = {
            "sigma": weight_sigma,
            "desolvation": weight_desolvation,
            "sei_homogeneity": weight_sei_homogeneity,
            "mx_synthesis": weight_mx_synthesis,
            "gwp": weight_gwp,
        }
        self.viability_threshold = viability_threshold

    def compute_score(
        self,
        molecule_input: MoleculeInput,
        tier1_result: Optional[MLXFilterResult] = None,
        tier2_result: Optional[Tier2Result] = None,
        tier3_result: Optional[GCMDTwinResult] = None,
        gwp_value: float = 1.0,
    ) -> AureliusScoreResult:
        """Compute the complete Aurelius v5.1 score.

        S_A = 0.3(σ) + 0.2(E_des_barrier) + 0.2(SEI Homogeneity)
              + 0.2(MX_Synthesis_Score) - 0.1(GWP)
        """
        result = AureliusScoreResult(
            molecule_smiles=molecule_input.smiles,
            viability_threshold=self.viability_threshold,
            weight_sigma=self.weights["sigma"],
            weight_desolvation=self.weights["desolvation"],
            weight_sei=self.weights["sei_homogeneity"],
            weight_mx=self.weights["mx_synthesis"],
            weight_gwp=self.weights["gwp"],
        )

        # Component 1: σ (MLX-NA filter confidence)
        if tier1_result is not None:
            result.sigma_score = self._normalize_sigma(tier1_result.confidence_score)
            result.tier1_viable = tier1_result.is_viable
        else:
            result.sigma_score = 50.0
            result.tier1_viable = True

        # Component 2: E_des_barrier (Normalized desolvation path integral)
        if tier2_result is not None:
            result.desolvation_score = self._normalize_desolvation_barrier(
                tier2_result.desolvation_path
            )
            result.tier2_viable = tier2_result.is_viable
        else:
            result.desolvation_score = 50.0
            result.tier2_viable = True

        # Component 3: SEI Homogeneity
        if tier3_result is not None:
            result.sei_homogeneity_score = self._normalize_sei_homogeneity(
                tier3_result.sei_evolution
            )
            result.tier3_viable = tier3_result.sei_evolution.electronic_insulation
        else:
            result.sei_homogeneity_score = 50.0
            result.tier3_viable = True

        # Component 4: MX_Synthesis_Score (Automated lab compatibility)
        result.mx_synthesis_score = self._compute_mx_synthesis_score(
            molecule_input, tier3_result
        )

        # Component 5: GWP penalty
        result.gwp_penalty = self._compute_gwp_penalty(gwp_value)

        # Total Aurelius Score v5.1
        raw_score = (
            self.weights["sigma"] * result.sigma_score
            + self.weights["desolvation"] * result.desolvation_score
            + self.weights["sei_homogeneity"] * result.sei_homogeneity_score
            + self.weights["mx_synthesis"] * result.mx_synthesis_score
            - self.weights["gwp"] * result.gwp_penalty
        )

        result.total_score = float(np.clip(raw_score, 0, 100))
        result.is_viable = result.total_score >= self.viability_threshold

        # Collect rejection reasons
        if not result.tier1_viable:
            result.rejection_reasons.append("Tier 1 (MLX-NA filter): molecule not viable")
        if not result.tier2_viable:
            result.rejection_reasons.append(
                "Tier 2 (MatterSim-MT): desolvation barrier exceeded"
            )
        if not result.tier3_viable:
            result.rejection_reasons.append(
                "Tier 3 (GCMD Digital Twin): SEI not electronically insulating"
            )
        if result.total_score < self.viability_threshold:
            result.rejection_reasons.append(
                f"Aurelius Score {result.total_score:.1f} < threshold {self.viability_threshold}"
            )

        return result

    def print_scorecard(self, score: AureliusScoreResult) -> str:
        """Generate a formatted scorecard for the Aurelius v5.1 result."""
        lines = [
            f"{'='*60}",
            f"  AURELIUS SCORE v5.1 - Scorecard",
            f"{'='*60}",
            f"  Molecule:     {score.molecule_smiles}",
            f"  Total S_A:    {score.total_score:.1f}/100 {'VIABLE' if score.is_viable else 'REJECTED'}",
            f"{'─'*60}",
            f"  Component Scores:",
            f"    σ (MLX-NA filter):       {score.sigma_score:>6.1f}  × {score.weight_sigma}",
            f"    E_des_barrier:           {score.desolvation_score:>6.1f}  × {score.weight_desolvation}",
            f"    SEI Homogeneity:         {score.sei_homogeneity_score:>6.1f}  × {score.weight_sei}",
            f"    MX_Synthesis_Score:      {score.mx_synthesis_score:>6.1f}  × {score.weight_mx}",
            f"    GWP Penalty:            -{score.gwp_penalty:>5.1f}  × {score.weight_gwp}",
            f"{'─'*60}",
            f"  Tier Status:",
            f"    Tier 1 (MLX-NA):  {'PASS' if score.tier1_viable else 'FAIL'}",
            f"    Tier 2 (MatterSim): {'PASS' if score.tier2_viable else 'FAIL'}",
            f"    Tier 3 (GCMD DT): {'PASS' if score.tier3_viable else 'FAIL'}",
        ]

        if score.rejection_reasons:
            lines.append(f"{'─'*60}")
            lines.append("  Rejection Reasons:")
            for reason in score.rejection_reasons:
                lines.append(f"    - {reason}")

        lines.append(f"{'='*60}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_sigma(confidence: float) -> float:
        """Normalize MLX-NA filter confidence to 0-100 scale."""
        return float(confidence * 100.0)

    @staticmethod
    def _normalize_desolvation_barrier(path_result: DesolvationPathResult) -> float:
        """Normalize desolvation path integral score to 0-100 scale.

        Lower barrier = better score. Local maxima > 0.5 eV = rejection.
        """
        if path_result.rejected:
            return 0.0

        barrier = path_result.barrier_height_eV
        score = 100.0 * np.exp(-barrier / 0.3)
        return float(np.clip(score, 0, 100))

    @staticmethod
    def _normalize_sei_homogeneity(sei: SEIEvolution) -> float:
        """Normalize SEI homogeneity to 0-100 scale."""
        base = sei.homogeneity_score * 100.0
        if sei.electronic_insulation:
            base = min(base * 1.1, 100.0)
        return float(np.clip(base, 0, 100))

    def _compute_mx_synthesis_score(
        self,
        molecule_input: MoleculeInput,
        tier3_result: Optional[GCMDTwinResult],
    ) -> float:
        """Compute MX_Synthesis_Score (automated lab compatibility).

        New metric from the 2026 AI for Materials Conference.
        Assesses how well a molecule's properties align with
        automated synthesis laboratory capabilities.
        """
        score = 70.0

        common_solvents = ["ec:dmc", "ec:emc", "pc:dmc", "water"]
        if molecule_input.solvent_type in common_solvents:
            score += 10.0

        common_salts = ["NaPF6", "NaTFSI", "NaClO4"]
        if molecule_input.salt_type in common_salts:
            score += 5.0

        common_ions = ["Na+", "Li+", "K+"]
        if molecule_input.ion_type in common_ions:
            score += 5.0

        if tier3_result is not None:
            if tier3_result.sei_evolution.electronic_insulation:
                score += 5.0
            if tier3_result.sei_evolution.homogeneity_score > 0.7:
                score += 5.0

        return float(np.clip(score, 0, 100))

    @staticmethod
    def _compute_gwp_penalty(gwp_value: float) -> float:
        """Compute GWP penalty (0-100 scale).

        Higher GWP → higher penalty.
        """
        return float(np.clip(gwp_value, 0, 100))
