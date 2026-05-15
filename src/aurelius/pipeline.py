"""Aurelius v5.1 Pipeline Orchestrator.

Coordinates the full three-tier screening pipeline:
  Tier 1: MLX-NA Filter (ChemVLM-2 MX4)
  Tier 2: MatterSim-MT (torch.compile Graph Mode)
  Tier 3: GCMD Digital Twin (TurboQuant KV-Compression)

Then computes the Aurelius Score v5.1 (S_A_v5.1).
"""

from typing import Optional

from aurelius.config import apply_global_config, M5ProConfig
from aurelius.memory.manager import (
    MetalShaderConfig,
    QuantizationConfig,
    ZeroCopyMemoryManager,
)
from aurelius.solvation.engine import MWSESolvationEngine
from aurelius.screening.tier1_mlx_filter import MLXNAFilter
from aurelius.screening.tier2_mattersim import MatterSimMTSimulator
from aurelius.screening.tier3_gcmtwin import GCMDigitalTwin, TurboQuantConfig
from aurelius.scoring.engine import AureliusScoringEngine, MoleculeInput


class AureliusPipeline:
    """Full Aurelius v5.1 screening pipeline orchestrator.

    Coordinates all three tiers and computes the final Aurelius Score.
    """

    def __init__(self, config: Optional[M5ProConfig] = None):
        self.config = config or apply_global_config()
        self._memory_manager: Optional[ZeroCopyMemoryManager] = None
        self._mlx_filter: Optional[MLXNAFilter] = None
        self._mattersim_sim: Optional[MatterSimMTSimulator] = None
        self._gcmtwin: Optional[GCMDigitalTwin] = None
        self._scoring_engine: Optional[AureliusScoringEngine] = None
        self._solvation_engine: Optional[MWSESolvationEngine] = None

    def initialize(self) -> None:
        """Initialize all pipeline components."""
        print("\n" + "=" * 60)
        print("  PROJECT AURELIUS v5.1 - Pipeline Initialization")
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
            )
            print("[Aurelius v5.1] Tier 1 (MLX-NA): ENABLED")

        if self.config.tier2_mattersim_enabled:
            self._mattersim_sim = MatterSimMTSimulator(
                barrier_threshold_eV=self.config.desolvation_barrier_threshold_eV,
            )
            print("[Aurelius v5.1] Tier 2 (MatterSim-MT): ENABLED")

        if self.config.tier3_gcmtwin_enabled:
            self._gcmtwin = GCMDigitalTwin(
                turboquant_config=TurboQuantConfig(
                    max_context_tokens=self.config.turquant_max_context,
                ),
            )
            print("[Aurelius v5.1] Tier 3 (GCMD Digital Twin): ENABLED")

        # Phase 4: Scoring engine
        self._scoring_engine = AureliusScoringEngine(
            weight_sigma=self.config.weight_sigma,
            weight_desolvation=self.config.weight_desolvation_barrier,
            weight_sei_homogeneity=self.config.weight_sei_homogeneity,
            weight_mx_synthesis=self.config.weight_mx_synthesis_score,
            weight_gwp=self.config.weight_gwp,
        )

        print(f"\n[Aurelius v5.1] Pipeline ready. "
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

    def screen_molecule(self, smiles: str, **kwargs) -> dict:
        """Run the complete three-tier screening pipeline on a molecule.

        Returns a dict with all tier results and the final Aurelius score.
        """
        if not self._scoring_engine:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

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

        # Tier 1: MLX-NA Filter
        tier1_result = None
        if self._mlx_filter:
            tier1_result = self._mlx_filter.screen_molecule(smiles)
            results["tier1"] = tier1_result
            print(f"\n  Tier 1 Result: {tier1_result.molecule_smiles} "
                  f"-> {'VIABLE' if tier1_result.is_viable else 'REJECTED'} "
                  f"(confidence={tier1_result.confidence_score:.3f}, "
                  f"time={tier1_result.inference_time_ms:.1f}ms)")

        # MWSE Solvation analysis
        mwse_state = None
        if self._solvation_engine:
            mwse_state = self._solvation_engine.evaluate_mwse_state(
                molecule_input.ion_type, molecule_input.solvent_type
            )
            results["mwse"] = mwse_state

        # Tier 2: MatterSim-MT
        tier2_result = None
        if self._mattersim_sim:
            tier2_result = self._mattersim_sim.simulate_desolvation(
                smiles,
                molecule_input.ion_type,
                molecule_input.solvent_type,
                molecule_input.n_md_cycles,
            )
            results["tier2"] = tier2_result
            print(f"  Tier 2 Result: {tier2_result.molecule_smiles} "
                  f"-> {'VIABLE' if tier2_result.is_viable else 'REJECTED'} "
                  f"(barrier={tier2_result.desolvation_path.barrier_height_eV:.3f} eV, "
                  f"time={tier2_result.simulation_time_ms:.1f}ms)")

        # Tier 3: GCMD Digital Twin
        tier3_result = None
        if self._gcmtwin:
            tier3_result = self._gcmtwin.simulate_sei_evolution(
                smiles,
                molecule_input.solvent_type,
                molecule_input.salt_type,
                molecule_input.voltage_cutoff,
                molecule_input.max_sei_time_ps,
            )
            results["tier3"] = tier3_result
            print(f"  Tier 3 Result: {tier3_result.molecule_smiles} "
                  f"-> SEI: {tier3_result.sei_evolution.thickness_angstrom:.1f}Å, "
                  f"Homogeneity={tier3_result.sei_evolution.homogeneity_score:.3f}")

        # Phase 4: Compute Aurelius Score
        gwp = kwargs.get("gwp_value", 1.0)
        score = self._scoring_engine.compute_score(
            molecule_input, tier1_result, tier2_result, tier3_result, gwp
        )
        results["score"] = score

        # Print scorecard
        print(f"\n{self._scoring_engine.print_scorecard(score)}")

        # Memory budget report
        if self._memory_manager:
            budget = self._memory_manager.get_memory_budget()
            print(f"\n[Aurelius v5.1] Memory Budget: "
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
