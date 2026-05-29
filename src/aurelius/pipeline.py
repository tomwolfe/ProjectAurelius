"""Aurelius v8.0 Pipeline Orchestrator.

Coordinates a streamlined two-step discovery pipeline:
  1. **Filter** — Quick structural validity and synthetic accessibility (SA) check.
  2. **Oracle** — Evaluate target property (e.g. HOMO/LUMO gap) using the real
     pre-trained ML model.

The results are then fed back to the GP surrogate for Bayesian optimisation.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from aurelius.config import AureliusConfig, apply_global_config
from aurelius.scoring.oracle import Oracle, PretrainedGNNOracle
from aurelius.screening.tier1 import MLXNAFilter
from aurelius.types import (
    DesolvationPathResult,
    MLXFilterResult,
    MoleculeInput,
    Tier2Result,
)
from aurelius.utils.dependencies import HAS_MLX, HAS_RDKIT

logger = logging.getLogger(__name__)


class AureliusPipeline:
    """Full Aurelius v8.0 screening pipeline orchestrator.

    Coordinates the streamlined Filter → Oracle pipeline and computes
    the final Aurelius Score.
    """

    def __init__(
        self,
        config: AureliusConfig | None = None,
        use_real_models: bool = True,
    ) -> None:
        """Initialise the Aurelius pipeline.

        Args:
            config: Pipeline configuration. If None, loads default.
        """
        self.config = config or apply_global_config()
        self._mlx_filter: MLXNAFilter | None = None
        self._use_real_models = use_real_models
        self._oracle: Oracle | None = None
        self.has_mlx = HAS_MLX
        self.has_torch = True  # Always available for the new pipeline

    def initialize(self) -> None:
        """Initialise all pipeline components."""
        try:
            import rdkit  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "RDKit is required for pipeline initialisation. "
                "Install with: pip install rdkit"
            ) from None

        # Phase 1: MLX-NA structural filter
        if self._use_real_models and self.config.tier1_mlxfilter_enabled:
            try:
                self._mlx_filter = MLXNAFilter(
                    quantization_format=self.config.chemvlm_quantization,
                )
                logger.info("Tier 1 (MLX-NA): ENABLED [REAL]")
            except Exception as exc:
                logger.warning("Tier 1 (MLX-NA): DISABLED – %s", exc)
                self._mlx_filter = None

        # Phase 2: Oracle for real ML-based property evaluation
        self._oracle = PretrainedGNNOracle()
        logger.info("Oracle (Pretrained GNN): ENABLED")

    def _generate_failed_run(self, smiles: str, reason: str, **kwargs: Any) -> dict[str, Any]:
        """Generate a failed run result dict for early-exit scenarios."""
        from aurelius.types import (
            MLXFilterResult,
            MoleculeInput,
        )

        molecule_input = MoleculeInput(
            smiles=smiles,
            solvent_type=kwargs.get("solvent_type", "ec:dmc"),
            salt_type=kwargs.get("salt_type", "NaPF6"),
            ion_type=kwargs.get("ion_type", "Na+"),
            temperature_k=kwargs.get("temperature_k", 298.15),
            voltage_cutoff=kwargs.get("voltage_cutoff", 0.05),
            max_sei_time_ps=kwargs.get("max_sei_time_ps", 1000.0),
            n_scan_cycles=kwargs.get("n_scan_cycles", 500),
        )

        failed_tier1 = MLXFilterResult(
            molecule_smiles=smiles,
            is_viable=False,
            confidence_score=0.0,
            inference_time_ms=0.0,
            na_utilization_pct=0.0,
        )

        # Compute Aurelius score from oracle results directly
        oracle_result = self._oracle.evaluate(smiles) if self._oracle else {}
        lumo_gap = oracle_result.get("lumo_gap_eV", 999.0)

        # Simplified scoring: viability based on LUMO gap
        total_score = max(0.0, 100.0 - lumo_gap * 5.0)  # Linear decay
        is_viable = lumo_gap < 6.0  # Reasonable LUMO gap threshold

        return {
            "tier1": failed_tier1,
            "tier2": None,
            "score": {
                "total_score": total_score,
                "is_viable": is_viable,
                "rejection_reasons": [reason],
            },
        }

    def screen_molecule(self, smiles: str, **kwargs: Any) -> dict[str, Any]:
        """Run a single molecule through the Filter → Oracle pipeline.

        Returns a dict with tier results and the final Aurelius score.
        Includes per-tier timing metrics for performance monitoring.
        """
        if not self._oracle:
            raise RuntimeError("Pipeline not initialised. Call initialise() first.")

        logger.info("Processing: %s", smiles)
        pipeline_start = time.perf_counter()

        # ── Step 1: Filter (structural validity + SA score) ──
        t1_result = None
        if self._mlx_filter:
            t1_start = time.perf_counter()
            t1_result = self._mlx_filter.screen_molecule(smiles)
            tier_timings: dict[str, float] = {}
            tier_timings["tier1_ms"] = (time.perf_counter() - t1_start) * 1000
            results: dict[str, Any] = {"tier1": t1_result}
            logger.info(
                "Tier 1 Result: %s -> %s (confidence=%.3f, time=%.1fms)",
                t1_result.molecule_smiles,
                "VIABLE" if t1_result.is_viable else "REJECTED",
                t1_result.confidence_score,
                t1_result.inference_time_ms,
            )
            if not t1_result.is_viable:
                logger.warning("Short-circuiting: %s failed Tier 1.", smiles)
                return self._generate_failed_run(smiles, "Failed Tier 1 Structural Filter", **kwargs)
        else:
            results = {}
            tier_timings = {}

        # ── Step 2: Oracle (real ML-based property evaluation) ──
        t2_result = None
        if self._oracle:
            t2_start = time.perf_counter()
            oracle_result = self._oracle.evaluate(smiles)
            tier_timings["tier2_ms"] = (time.perf_counter() - t2_start) * 1000

            # Convert oracle output to a Tier2Result for compatibility
            from aurelius.types import (
                DesolvationPathResult,
                Tier2Result,
            )

            t2_result = Tier2Result(
                molecule_smiles=smiles,
                is_viable=oracle_result.get("lumo_gap_eV", 999.0) < 10.0,  # reasonable threshold
                desolvation_path=DesolvationPathResult(
                    molecule_smiles=smiles,
                    barrier_height_eV=oracle_result.get("lumo_gap_eV", 999.0),
                    local_maxima_eV=oracle_result.get("lumo_gap_eV", 999.0) * 0.8,
                    path_integral_eV_A=oracle_result.get("lumo_gap_eV", 999.0) * 1.2,
                    rejected=False,
                ),
                simulation_time_ms=tier_timings["tier2_ms"],
                memory_used_gb=0.1,
            )
            results["tier2"] = t2_result
            logger.info(
                "Tier 2 (Oracle) Result: %s -> LUMO gap=%.3f eV",
                t2_result.molecule_smiles,
                t2_result.desolvation_path.barrier_height_eV,
            )

            # HARD SHORT-CIRCUIT: Reject if LUMO gap too large
            if t2_result.desolvation_path.barrier_height_eV > 10.0:
                logger.warning(
                    "Short-circuiting: %s failed Oracle (LUMO gap=%.3f eV)",
                    smiles,
                    t2_result.desolvation_path.barrier_height_eV,
                )
                return self._generate_failed_run(
                    smiles,
                    f"Failed Tier 2 Oracle (LUMO gap: {t2_result.desolvation_path.barrier_height_eV} eV)",
                    **kwargs,
                )

        # Final consolidated score compilation
        score = self._compute_score(oracle_result)
        results["score"] = score

        # Print scorecard
        logger.debug("Scorecard:\n%s", self._format_score(score))

        # Performance report
        total_ms = (time.perf_counter() - pipeline_start) * 1000
        timing_lines = []
        for tier, t_ms in tier_timings.items():
            timing_lines.append(f"    {tier}: {t_ms:.1f}ms")
        if timing_lines:
            logger.info("Performance: total=%.1fms | %s", total_ms, " | ".join(timing_lines))

        return results

    def _compute_score(self, oracle_result: dict[str, float]) -> dict[str, Any]:
        """Compute the final Aurelius Score from oracle results.

        Args:
            oracle_result: Dictionary with keys like ``lumo_gap_eV``, ``homo_eV``, etc.

        Returns:
            Dict with ``total_score``, ``is_viable``, and ``rejection_reasons``.
        """
        lumo_gap = oracle_result.get("lumo_gap_eV", 999.0)

        # Simple scoring based on LUMO gap
        total_score = max(0.0, 100.0 - lumo_gap * 5.0)
        is_viable = lumo_gap < 6.0

        rejection_reasons: list[str] = []
        if not is_viable:
            rejection_reasons.append(
                f"Aurelius Score {total_score:.1f} below viability threshold (LUMO gap: {lumo_gap:.3f} eV)"
            )

        return {
            "total_score": total_score,
            "is_viable": is_viable,
            "rejection_reasons": rejection_reasons,
        }

    @staticmethod
    def _format_score(score: dict[str, Any]) -> str:
        """Format a score dict for logging."""
        total = score.get("total_score", 0.0)
        viable = score.get("is_viable", False)
        return f"Score: {total:.1f}/100 {'VIABLE' if viable else 'REJECTED'}"

    def screen_batch(self, smiles_list: list[str], n_workers: int = 1, **kwargs: Any) -> list[dict[str, Any]]:
        """Screen a batch of molecules through the full pipeline.

        When ``n_workers`` is greater than 1, molecules are screened
        in parallel using ``ThreadPoolExecutor``.
        """
        if n_workers < 1 or n_workers == 1:
            return [self.screen_molecule(smiles, **kwargs) for smiles in smiles_list]

        results: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(self.screen_molecule, smiles, **kwargs): i
                for i, smiles in enumerate(smiles_list)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        return [results[i] for i in range(len(smiles_list))]
