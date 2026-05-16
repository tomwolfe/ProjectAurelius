"""Aurelius v5.2 Pipeline Orchestrator.

Coordinates the full three-tier screening pipeline:
  Tier 1: MLX-NA Filter (ChemVLM-2 MX4)
  Tier 2: MatterSim-MT (torch.compile Graph Mode)
  Tier 3: GCMD Digital Twin (Arrhenius kMC)

Then computes the Aurelius Score v5.2 (S_A_v5.2).
"""

from __future__ import annotations

import gc
import time

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    mx = None  # type: ignore
    HAS_MLX = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    HAS_TORCH = False

from aurelius.config import M5ProConfig, apply_global_config
from aurelius.memory.manager import (
    MetalShaderConfig,
    QuantizationConfig,
    ZeroCopyMemoryManager,
)
from aurelius.scoring.engine import AureliusScoringEngine
from aurelius.screening.tier1_mlx_filter import MLXNAFilter
from aurelius.screening.tier2_mattersim import MatterSimMTSimulator
from aurelius.screening.tier3_gcmtwin import GCMDigitalTwin, GCMDTConfig
from aurelius.solvation.engine import MWSESolvationEngine
from aurelius.types import (
    DesolvationPathResult,
    MLXFilterResult,
    MoleculeInput,
    Tier2Result,
)


class AureliusPipeline:
    """Full Aurelius v5.2 screening pipeline orchestrator.

    Coordinates all three tiers and computes the final Aurelius Score.
    """

    def __init__(
        self,
        config: M5ProConfig | None = None,
        use_real_models: bool = True,
    ) -> None:
        """Initialize the Aurelius pipeline.

        Args:
            config: Pipeline configuration. If None, loads default.
            use_real_models: If True, Tier 1 loads/trains on real data.
                If False, uses synthetic data (demo mode).
        """
        self.config = config or apply_global_config()
        self._memory_manager: ZeroCopyMemoryManager | None = None
        self._mlx_filter: MLXNAFilter | None = None
        self._mattersim_sim: MatterSimMTSimulator | None = None
        self._gcmtwin: GCMDigitalTwin | None = None
        self._scoring_engine: AureliusScoringEngine | None = None
        self._solvation_engine: MWSESolvationEngine | None = None
        self.has_mlx = HAS_MLX
        self.has_torch = HAS_TORCH
        self._use_real_models = use_real_models

    def initialize(self) -> None:
        """Initialize all pipeline components."""
        print("\n" + "=" * 60)
        print("  PROJECT AURELIUS v5.2 - Pipeline Initialization")
        print("  The 2nm Fusion Edition | M5 Pro Neural Accelerators")
        print("=" * 60 + "\n")

        # Phase 1: Memory manager
        self._memory_manager = ZeroCopyMemoryManager(
            quant_config=QuantizationConfig(
                precision=self.config.chemvlm_quantization,
            ),
            shader_config=MetalShaderConfig(),
        )
        self._memory_manager.initialize_accelerator()
        self._memory_manager.load_precompiled_shaders()

        # Phase 2: MWSE Solvation engine
        self._solvation_engine = MWSESolvationEngine(
            kex_window_ps=self.config.kex_screening_window_ps,
        )

        # Phase 3: Screening tiers
        if self.config.tier1_mlxfilter_enabled:
            self._mlx_filter = MLXNAFilter(
                quantization_format=self.config.chemvlm_quantization,
                use_real_models=self._use_real_models,
            )
            mode = "REAL" if self._use_real_models else "SYNTHETIC (demo)"
            print(f"[Aurelius v5.2] Tier 1 (MLX-NA): ENABLED [{mode}]")

        if self.config.tier2_mattersim_enabled:
            self._mattersim_sim = MatterSimMTSimulator(
                barrier_threshold_eV=self.config.desolvation_barrier_threshold_eV,
            )
            device = self._mattersim_sim._select_device() if self._mattersim_sim else "cpu"
            print(f"[Aurelius v5.2] Tier 2 (MatterSim-MT): ENABLED (device={device})")

        if self.config.tier3_gcmtwin_enabled:
            self._gcmtwin = GCMDigitalTwin(
                gcmtwin_config=GCMDTConfig(
                    max_simulation_steps=self.config.turquant_max_context,
                ),
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

        print(f"\n[Aurelius v5.2] Pipeline ready. "
              f"Viability threshold: {self._scoring_engine.viability_threshold}\n")

    def load_models(self, chemvlm2_path: str = "models/chemvlm2",
                    mattersim_path: str = "models/mattersim_mt",
                    gcmtwin_path: str = "models/gcmd_dt") -> None:
        """Load all models into the memory manager."""
        if self._memory_manager:
            self._memory_manager.load_chemvlm2(chemvlm2_path)
            self._memory_manager.load_mattersim(mattersim_path)
            self._memory_manager.load_gcmtwin(gcmtwin_path)

        if self._mlx_filter:
            self._mlx_filter.load_model(chemvlm2_path)

        if self._mattersim_sim:
            self._mattersim_sim.initialize(mattersim_path)

    def _generate_failed_run(self, smiles: str, reason: str, **kwargs) -> dict:
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
            n_md_cycles=kwargs.get("n_md_cycles", 500),
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

        score = self._scoring_engine.compute_score(
            molecule_input, failed_tier1, failed_tier2, None, 1.0
        )

        return {
            "tier1": failed_tier1,
            "tier2": failed_tier2,
            "tier3": None,
            "score": score,
        }

    def screen_molecule(self, smiles: str, **kwargs) -> dict:
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
            n_md_cycles=kwargs.get("n_md_cycles", 500),
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
            print(f"\n  Tier 1 Result: {t1_result.molecule_smiles} "
                  f"-> {'VIABLE' if t1_result.is_viable else 'REJECTED'} "
                  f"(confidence={t1_result.confidence_score:.3f}, "
                  f"time={t1_result.inference_time_ms:.1f}ms)")
            if not t1_result.is_viable:
                print(f"[Aurelius Pipeline] Short-circuiting: {smiles} failed Tier 1.")
                results["tier_timings"] = tier_timings
                return self._generate_failed_run(smiles, "Failed Tier 1 Structural Filter", **kwargs)

        # MWSE Solvation analysis
        mwse_state = None
        if self._solvation_engine:
            mwse_state = self._solvation_engine.evaluate_mwse_state(
                molecule_input.ion_type, molecule_input.solvent_type
            )
            results["mwse"] = mwse_state

        # Tier 2: MatterSim-MT
        t2_result = None
        if self._mattersim_sim:
            t2_start = time.perf_counter()
            t2_result = self._mattersim_sim.simulate_desolvation(
                smiles,
                molecule_input.ion_type,
                molecule_input.solvent_type,
                molecule_input.n_md_cycles,
            )
            tier_timings["tier2_ms"] = (time.perf_counter() - t2_start) if 't2_start' in dir() else t2_result.simulation_time_ms
            results["tier2"] = t2_result
            print(f"  Tier 2 Result: {t2_result.molecule_smiles} "
                  f"-> {'VIABLE' if t2_result.is_viable else 'REJECTED'} "
                  f"(barrier={t2_result.desolvation_path.barrier_height_eV:.3f} eV, "
                  f"time={t2_result.simulation_time_ms:.1f}ms)")

            # HARD SHORT-CIRCUIT: Explicit early exit
            if not t2_result.is_viable:
                print(f"[Aurelius Pipeline] Short-circuiting: {smiles} failed Tier 2 viability.")
                results["tier_timings"] = tier_timings
                return self._generate_failed_run(
                    smiles, f"Failed Tier 2 Solvation (Barrier: {t2_result.desolvation_path.barrier_height_eV} eV)", **kwargs
                )

        # CROSS-TIER HARDWARE CLEANUP
        gc.collect()
        if self.has_mlx:
            mx.metal.clear_cache()
        if self.has_torch:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if hasattr(torch.backends, "cuda") and torch.backends.cuda.is_built():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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
            results["tier3"] = t3_result
            print(f"  Tier 3 Result: {t3_result.molecule_smiles} "
                  f"-> SEI: {t3_result.sei_evolution.thickness_angstrom:.1f}A, "
                  f"Homogeneity={t3_result.sei_evolution.homogeneity_score:.3f}")

        # POST-TIER-3 MEMORY CLEANUP
        gc.collect()
        if self.has_mlx:
            mx.metal.clear_cache()
        if self.has_torch:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if hasattr(torch.backends, "cuda") and torch.backends.cuda.is_built():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Final consolidated score compilation
        gwp = kwargs.get("gwp_value", 1.0)
        score = self._scoring_engine.compute_score(
            molecule_input, t1_result, t2_result, t3_result, gwp
        )
        results["score"] = score
        results["tier_timings"] = tier_timings

        # Print scorecard
        print(f"\n{self._scoring_engine.print_scorecard(score)}")

        # Performance report
        total_ms = (time.perf_counter() - pipeline_start) * 1000
        timing_lines = []
        for tier, t_ms in tier_timings.items():
            timing_lines.append(f"    {tier}: {t_ms:.1f}ms")
        if timing_lines:
            print(f"\n[Aurelius v5.2] Performance: total={total_ms:.1f}ms | " + " | ".join(timing_lines))

        # Memory budget report
        if self._memory_manager:
            budget = self._memory_manager.get_memory_budget()
            print(f"\n[Aurelius v5.2] Memory Budget: "
                  f"{budget['chemvlm2_footprint_gb']}GB ChemVLM-2, "
                  f"{budget['remaining_gb']}GB remaining")

        return results

    def screen_batch(self, smiles_list: list[str], **kwargs) -> list[dict]:
        """Screen a batch of molecules through the full pipeline."""
        all_results = []
        for smiles in smiles_list:
            result = self.screen_molecule(smiles, **kwargs)
            all_results.append(result)
        return all_results

    def get_memory_budget(self) -> dict:
        """Get current memory allocation status."""
        if self._memory_manager:
            return self._memory_manager.get_memory_budget()
        return {}
