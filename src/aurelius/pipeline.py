"""Aurelius v5.2 Pipeline Orchestrator.

Coordinates the full three-tier screening pipeline:
  Tier 1: MLX-NA Filter (ChemVLM-2 MX4)
  Tier 2: MatterSim-MT (torch.compile Graph Mode)
  Tier 3: GCMD Digital Twin (Arrhenius kMC)

Then computes the Aurelius Score v5.2 (S_A_v5.2).
"""

from __future__ import annotations

import logging
import multiprocessing
import time
from typing import Any

from aurelius.config import AureliusConfig, apply_global_config
from aurelius.scoring.engine import AureliusScoringEngine
from aurelius.screening.tier1 import MLXNAFilter
from aurelius.screening.tier2_mattersim import MatterSimMTSimulator
from aurelius.screening.tier3_gcmtwin import GCMDigitalTwin, GCMDTConfig  # type: ignore[attr-defined]
from aurelius.solvation.engine import MWSESolvationEngine
from aurelius.types import (
    DesolvationPathResult,
    MLXFilterResult,
    MoleculeInput,
    Tier2Result,
)
from aurelius.utils.dependencies import HAS_MLX, HAS_TORCH

logger = logging.getLogger(__name__)


class AureliusPipeline:
    """Full Aurelius v5.2 screening pipeline orchestrator.

    Coordinates all three tiers and computes the final Aurelius Score.
    """

    def __init__(
        self,
        config: AureliusConfig | None = None,
        use_real_models: bool = True,

    ) -> None:
        """Initialize the Aurelius pipeline.

        Args:
            config: Pipeline configuration. If None, loads default.

        """
        self.config = config or apply_global_config()
        self._mlx_filter: MLXNAFilter | None = None
        self._mattersim_sim: MatterSimMTSimulator | None = None
        self._gcmtwin: GCMDigitalTwin | None = None
        self._scoring_engine: AureliusScoringEngine | None = None
        self._solvation_engine: MWSESolvationEngine | None = None
        self._use_real_models = use_real_models
        self.has_mlx = HAS_MLX
        self.has_torch = HAS_TORCH


    def initialize(self) -> None:
        """Initialize all pipeline components."""
        # Enforce RDKit availability for real model paths
        try:
            import rdkit  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "RDKit is required for pipeline initialization. "
                "Install with: pip install rdkit"
            ) from None

        print("\n" + "=" * 60)
        print("  PROJECT AURELIUS v7.0 - Pipeline Initialization")
        print("  The 2nm Fusion Edition | M5 Pro Neural Accelerators")
        print("=" * 60 + "\n")

        # Phase 1: MWSE Solvation engine
        self._solvation_engine = MWSESolvationEngine(
            kex_window_ps=self.config.kex_screening_window_ps,
        )

        # Phase 3: Screening tiers
        if self._use_real_models and self.config.tier1_mlxfilter_enabled:
            self._mlx_filter = MLXNAFilter(
                quantization_format=self.config.chemvlm_quantization,
            )
            print("[Aurelius v5.2] Tier 1 (MLX-NA): ENABLED [REAL]")

        if self._use_real_models and self.config.tier2_mattersim_enabled:
            self._mattersim_sim = MatterSimMTSimulator(
                barrier_threshold_eV=self.config.desolvation_barrier_threshold_eV,
            )
            device = self._mattersim_sim._select_device() if self._mattersim_sim else "cpu"
            print(f"[Aurelius v5.2] Tier 2 (MatterSim-MT): ENABLED (device={device})")

        if self._use_real_models and self.config.tier3_gcmtwin_enabled:
            self._gcmtwin = GCMDigitalTwin(
                gcmtwin_config=GCMDTConfig(
                    max_simulation_steps=self.config.turquant_max_context,
                ),
                use_tier0_prediction=True,
                tier0_model_path="models/tier0/mpnn_weights.pth",
            )
            print("[Aurelius v5.2] Tier 3 (GCMD Digital Twin): ENABLED")

        # Phase 4: Scoring engine
        self._scoring_engine = AureliusScoringEngine(
            weight_sigma=self.config.weight_sigma,
            weight_desolvation=self.config.weight_desolvation_barrier,
            weight_sei_homogeneity=self.config.weight_sei_homogeneity,
            weight_mx_synthesis=self.config.weight_mx_synthesis_score,
            weight_gwp=self.config.weight_gwp,
        )

        print(f"\n[Aurelius v5.2] Pipeline ready. Viability threshold: {self._scoring_engine.viability_threshold}\n")

    def _generate_failed_run(self, smiles: str, reason: str, **kwargs: Any) -> dict[str, Any]:
        """Generate a failed run result dict for early-exit scenarios.

        Returns an automatic-failure profile so downstream scoring
        components still receive a well-structured response.
        """
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
        failed_tier2 = Tier2Result(
            molecule_smiles=smiles,
            is_viable=False,
            desolvation_path=DesolvationPathResult(
                molecule_smiles=smiles,
                barrier_height_eV=0.0,
                local_maxima_eV=0.0,
                path_integral_eV_A=0.0,
                rejected=True,
                rejection_reason=reason,
            ),
            simulation_time_ms=0.0,
            memory_used_gb=0.0,
        )

        score = self._scoring_engine.compute_score(  # type: ignore[union-attr]
            molecule_input, failed_tier1, failed_tier2, None, 1.0
        )

        return {
            "tier1": failed_tier1,
            "tier2": failed_tier2,
            "tier3": None,
            "score": score,
        }

    def screen_molecule(self, smiles: str, **kwargs: Any) -> dict[str, Any]:
        """Run the complete three-tier screening pipeline on a molecule.

        Returns a dict with all tier results and the final Aurelius score.
        Includes per-tier timing metrics for performance monitoring.
        """
        if not self._scoring_engine:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        print(f"[Aurelius Pipeline] Processing: {smiles}")
        pipeline_start = time.perf_counter()

        # Build molecule input
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

        results = {}
        tier_timings: dict[str, float] = {}

        # Tier 1: MLX-NA Filter
        t1_result = None
        if self._mlx_filter:
            t1_start = time.perf_counter()
            t1_result = self._mlx_filter.screen_molecule(smiles)
            tier_timings["tier1_ms"] = (time.perf_counter() - t1_start) * 1000
            results["tier1"] = t1_result
            print(
                f"\n  Tier 1 Result: {t1_result.molecule_smiles} "
                f"-> {'VIABLE' if t1_result.is_viable else 'REJECTED'} "
                f"(confidence={t1_result.confidence_score:.3f}, "
                f"time={t1_result.inference_time_ms:.1f}ms)"
            )
            if not t1_result.is_viable:
                print(f"[Aurelius Pipeline] Short-circuiting: {smiles} failed Tier 1.")
                results["tier_timings"] = tier_timings  # type: ignore[assignment]
                return self._generate_failed_run(smiles, "Failed Tier 1 Structural Filter", **kwargs)  # type: ignore[return-value]

        # MWSE Solvation analysis
        mwse_state = None
        if self._solvation_engine:
            mwse_state = self._solvation_engine.evaluate_mwse_state(
                molecule_input.ion_type, molecule_input.solvent_type
            )
            results["mwse"] = mwse_state  # type: ignore[assignment]

        # Tier 2: MatterSim-MT
        t2_result = None
        if self._mattersim_sim:
            t2_start = time.perf_counter()
            t2_result = self._mattersim_sim.simulate_desolvation(
                smiles,
                molecule_input.ion_type,
                molecule_input.solvent_type,
                molecule_input.n_scan_cycles,
            )
            tier_timings["tier2_ms"] = (time.perf_counter() - t2_start) * 1000
            results["tier2"] = t2_result  # type: ignore[assignment]
            print(
                f"  Tier 2 Result: {t2_result.molecule_smiles} "
                f"-> {'VIABLE' if t2_result.is_viable else 'REJECTED'} "
                f"(barrier={t2_result.desolvation_path.barrier_height_eV:.3f} eV, "
                f"time={t2_result.simulation_time_ms:.1f}ms)"
            )

            # HARD SHORT-CIRCUIT: Explicit early exit
            if not t2_result.is_viable:
                print(f"[Aurelius Pipeline] Short-circuiting: {smiles} failed Tier 2 viability.")
                results["tier_timings"] = tier_timings  # type: ignore[assignment]
                return self._generate_failed_run(
                    smiles,
                    f"Failed Tier 2 Solvation (Barrier: {t2_result.desolvation_path.barrier_height_eV} eV)",
                    **kwargs,
                )

        # Tier 3: GCMD Digital Twin
        t3_result = None
        if self._gcmtwin:
            t3_start = time.perf_counter()
            t3_result = self._gcmtwin.simulate_sei_evolution(
                smiles,
                molecule_input.solvent_type,
                molecule_input.salt_type,
                molecule_input.voltage_cutoff,
                molecule_input.max_sei_time_ps,
            )
            tier_timings["tier3_ms"] = (time.perf_counter() - t3_start) * 1000
            results["tier3"] = t3_result  # type: ignore[assignment]
            print(
                f"  Tier 3 Result: {t3_result.molecule_smiles} "
                f"-> SEI: {t3_result.sei_evolution.thickness_angstrom:.1f}A, "
                f"Homogeneity={t3_result.sei_evolution.homogeneity_score:.3f}"
            )

        # Final consolidated score compilation
        gwp = kwargs.get("gwp_value", 1.0)
        score = self._scoring_engine.compute_score(molecule_input, t1_result, t2_result, t3_result, gwp)  # type: ignore[union-attr]
        results["score"] = score  # type: ignore[assignment]
        results["tier_timings"] = tier_timings  # type: ignore[assignment]

        # Print scorecard
        print(f"\n{self._scoring_engine.print_scorecard(score)}")

        # Performance report
        total_ms = (time.perf_counter() - pipeline_start) * 1000
        timing_lines = []
        for tier, t_ms in tier_timings.items():
            timing_lines.append(f"    {tier}: {t_ms:.1f}ms")
        if timing_lines:
            print(f"\n[Aurelius v5.2] Performance: total={total_ms:.1f}ms | " + " | ".join(timing_lines))

        return results

    def screen_batch(self, smiles_list: list[str], n_workers: int = 1, **kwargs: Any) -> list[dict[str, Any]]:
        """Screen a batch of molecules through the full pipeline.

        When ``n_workers`` is greater than 1, molecules are screened
        in parallel using ``ProcessPoolExecutor`` with the ``spawn``
        multiprocessing context to avoid MPS context crashes in forked
        processes.

        RDKit objects are never passed across process boundaries; only
        SMILES strings are pickled, and molecule reconstruction happens
        inside each worker process.

        Args:
            smiles_list: List of SMILES strings to screen.
            n_workers: Number of parallel workers.  Default ``1`` means
                sequential execution (no parallelism).
            **kwargs: Extra keyword arguments forwarded to
                ``screen_molecule`` for each molecule.

        Returns:
            List of per-molecule pipeline results.
        """
        if n_workers < 1 or n_workers == 1:
            return [self.screen_molecule(smiles, **kwargs) for smiles in smiles_list]

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            all_results = pool.map(
                lambda smiles: self.screen_molecule(smiles, **kwargs),
                smiles_list,
            )
        return all_results
