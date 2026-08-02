"""Aurelius Pipeline Orchestrator.

Coordinates a streamlined two-step discovery pipeline:
  1. Filter — Quick structural validity (Tier 1) check with LogP and MW gates.
  2. Oracle — Multi-objective property evaluation via hybrid quantum/GC
     correction (HOMO, LUMO, Dielectric proxy, Viscosity proxy, SA Score).
  3. Score — Multi-objective composite with Al corrosion penalty.
     Each objective is a callable that transforms a raw value to a sub-score.

All stages accept a pre-parsed MoleculeContext to enforce single-point parsing.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

try:
    from rdkit import Chem
except ImportError:
    raise ImportError(
        "RDKit not found. Install via: conda install -c conda-forge rdkit"
    ) from None

from aurelius.filter import Filter
from aurelius.kernel_loader import JSONKernelLoader, KernelLoader, _load_demo_kernel
from aurelius.mixer import screen_mixture as _screen_mixture
from aurelius.scorer import (
    _OBJECTIVES,
    Objective,
    _apply_domain_penalty,
    _check_al_corrosion_risk,
    _check_building_block_grounding,
    _check_hydrolytic_instability,
    _check_hypofluorite_instability,
)
from aurelius.scorer import (
    compute_score as _compute_score,
)
from aurelius.scorer import (
    format_score as _format_score,
)
from aurelius.scoring.oracle import (
    PropertyOracle,
    _compute_sei_fracture_toughness_proxy,
    predict_gas_evolution_proxy,
)
from aurelius.scoring.oracle.gc import BasePropertyModel
from aurelius.types import MoleculeContext

logger = logging.getLogger(__name__)


DEFAULT_DOMAIN_BOUNDARIES: dict[str, tuple[float, float]] = {
    "mw": (30.0, 500.0),
    "logp": (-2.0, 6.0),
    "tpsa": (0.0, 200.0),
    "hba": (1, 12),
    "hbd": (0, 6),
    "ring_count": (1, 6),
}


def check_kernel_health(ctx: MoleculeContext) -> bool:
    """Compare molecule properties against the default kernel domain boundary.

    Returns True if the molecule is inside the domain, False if outside.
    Logs a warning when outside the domain.
    """
    inside = True
    for prop, (lo, hi) in DEFAULT_DOMAIN_BOUNDARIES.items():
        value = getattr(ctx, prop, None)
        if value is not None and not (lo <= value <= hi):
            logger.warning(
                "Molecule outside default kernel domain: %s=%s (domain [%s, %s]). "
                "Consider retuning kernel parameters for better accuracy.",
                prop, value, lo, hi,
            )
            inside = False
    return inside


__all__ = [
    "AureliusPipeline",
    "KernelLoader",
    "JSONKernelLoader",
    "_load_demo_kernel",
    "_OBJECTIVES",
    "Objective",
    "DEFAULT_DOMAIN_BOUNDARIES",
    "check_kernel_health",
]


class AureliusPipeline:
    """Full Aurelius screening pipeline orchestrator.

    Coordinates the Filter -> Oracle -> Score pipeline.
    All stages accept a pre-parsed MoleculeContext to avoid redundant
    RDKit parsing.
    """

    def __init__(
        self,
        use_real_models: bool = True,
        property_pack: BasePropertyModel | None = None,
        kernel_loader: KernelLoader | None = None,
        solvent: str | None = "ether",
    ) -> None:
        self._filter: Filter | None = None
        self._use_real_models = use_real_models
        self._oracle: PropertyOracle | None = None
        self._property_pack = property_pack
        self._kernel_loader: KernelLoader = kernel_loader or JSONKernelLoader()
        self._solvent = solvent

    def initialize(self) -> None:
        """Initialise all pipeline components."""
        import rdkit  # noqa: F401

        if self._use_real_models:
            try:
                self._filter = Filter()
                logger.info("Tier 1 (Filter): ENABLED")
            except Exception as exc:
                logger.warning("Tier 1 (Filter): DISABLED - %s", exc)
                self._filter = None

        self._oracle = PropertyOracle(property_pack=self._property_pack, solvent=self._solvent)
        oracle_cache = "oracle_cache.joblib"
        if not self._oracle.load(oracle_cache):
            logger.info("Oracle (PropertyOracle): no cache found — using GC model directly.")
        else:
            logger.info("Oracle (PropertyOracle): loaded from cache (%s).", oracle_cache)
        logger.info("Oracle (PropertyOracle): ENABLED")

    def load_kernel(self, kernel_path: str) -> dict[str, Any] | None:
        """Load a kernel via the configured ``KernelLoader``.

        Args:
            kernel_path: Path to a kernel JSON file.

        Returns:
            Dict of kernel parameters on successful load, or ``None``
            to indicate the caller should use defaults.
        """
        result = self._kernel_loader.load(kernel_path)
        if result is None:
            logger.warning(
                "No valid kernel loaded — using default parameters.",
            )
        return result

    def _generate_failed_run(self, smiles: str, reason: str) -> dict[str, Any]:
        t1_result = {
            "molecule_smiles": smiles,
            "is_viable": False,
            "inference_time_ms": 0.0,
        }
        return {
            "tier1": t1_result,
            "tier2": None,
            "score": {
                "total_score": 0.0,
                "is_viable": False,
                "rejection_reasons": [reason],
            },
        }

    def screen_molecule(
        self,
        ctx: MoleculeContext,
    ) -> dict[str, Any]:
        """Run a single pre-parsed molecule through the Filter -> Oracle pipeline.

        Args:
            ctx: Pre-parsed MoleculeContext.

        Returns:
            Dict with tier results and the final Aurelius score.

        Raises:
            TypeError: If ctx is not a MoleculeContext.
            RuntimeError: If pipeline not initialised.
        """
        if not isinstance(ctx, MoleculeContext):
            raise TypeError(
                f"AureliusPipeline.screen_molecule() requires a MoleculeContext, "
                f"got {type(ctx).__name__}. Use MoleculeContext.from_smiles() first."
            )
        if not self._oracle:
            raise RuntimeError(
                "Pipeline not initialised. Call pipeline.initialize() first before "
                "running screen_molecule() or screen_smiles()."
            )

        smiles = ctx.smiles
        logger.info("Processing: %s", smiles)
        pipeline_start = time.perf_counter()

        t1_result = None
        if self._filter:
            t1_start = time.perf_counter()
            t1_result = self._filter.screen(ctx)
            tier_timings: dict[str, float] = {}
            tier_timings["tier1_ms"] = (time.perf_counter() - t1_start) * 1000
            results: dict[str, Any] = {"tier1": t1_result}
            logger.info(
                "Tier 1 Result: %s -> %s (time=%.1fms)",
                smiles,
                "VIABLE" if t1_result.get("is_viable", False) else "REJECTED",
                t1_result.get("inference_time_ms", 0.0),
            )
            if not t1_result.get("is_viable", True):
                logger.warning("Short-circuiting: %s failed Tier 1.", smiles)
                return self._generate_failed_run(smiles, "Failed Tier 1 Structural Filter")
        else:
            results = {}
            tier_timings = {}

        check_kernel_health(ctx)

        t2_result = None
        homo_eV = -99.0
        lumo_eV = -99.0
        dielectric_proxy = 0.0
        viscosity_proxy = 99.0
        sei_fracture_toughness_proxy = 0.0
        if self._oracle:
            t2_start = time.perf_counter()
            oracle_result = self._oracle.evaluate(ctx)
            tier_timings["tier2_ms"] = (time.perf_counter() - t2_start) * 1000

            homo_eV = oracle_result.get("homo_eV", -99.0)
            lumo_eV = oracle_result.get("lumo_eV", -99.0)
            dielectric_proxy = oracle_result.get("dielectric_proxy", 0.0)
            viscosity_proxy = oracle_result.get("viscosity_proxy", 99.0)
            li_solvation_proxy = oracle_result.get("li_solvation_proxy", 0.0)
            li_binding_energy_kcal = oracle_result.get("li_binding_energy_kcal", 0.0)
            ced_proxy = oracle_result.get("ced_proxy", 0.0)
            hydrolysis_risk_proxy = oracle_result.get("hydrolysis_risk_proxy", 0.0)

            domain_penalty = oracle_result.get("domain_penalty", 1.0)

            sei_fracture_toughness_proxy = _compute_sei_fracture_toughness_proxy(ctx)
            gas_evolution_proxy = predict_gas_evolution_proxy(ctx)

            t2_result = dict(oracle_result)
            t2_result["homo_eV"] = homo_eV
            t2_result["lumo_eV"] = lumo_eV
            t2_result["gap_eV"] = oracle_result.get("gap_eV", 0.0)
            t2_result["sei_fracture_toughness_proxy"] = sei_fracture_toughness_proxy
            t2_result["gas_evolution_proxy"] = gas_evolution_proxy
            t2_result.setdefault("domain_applicable", True)
            t2_result.setdefault("domain_reason", "")
            t2_result["domain_penalty"] = domain_penalty
            t2_result.setdefault("quantum_confidence", "unknown")
            t2_result.setdefault("li_binding_energy_kcal", li_binding_energy_kcal)
            results["tier2"] = t2_result
            logger.info(
                "Oracle Result: %s -> HOMO=%.3f LUMO=%.3f Dielectric=%.3f Viscosity=%.3f LiSolv=%.3f",
                smiles, homo_eV, lumo_eV, dielectric_proxy, viscosity_proxy, li_solvation_proxy,
            )

        quantum_confidence = t2_result.get("quantum_confidence", "unknown") if t2_result else "unknown"
        score = _compute_score(
            homo_eV, lumo_eV,
            dielectric_proxy=dielectric_proxy,
            viscosity_proxy=viscosity_proxy,
            li_solvation_proxy=li_solvation_proxy,
            li_binding_energy_kcal=li_binding_energy_kcal,
            ced_proxy=ced_proxy,
            sei_fracture_toughness_proxy=sei_fracture_toughness_proxy,
            gas_evolution_proxy=gas_evolution_proxy,
            hydrolysis_risk_proxy=hydrolysis_risk_proxy,
            ctx=ctx,
            quantum_confidence=quantum_confidence,
        )

        score = _apply_domain_penalty(score, t2_result)
        results["score"] = score

        logger.debug("Scorecard:\n%s", _format_score(score))

        total_ms = (time.perf_counter() - pipeline_start) * 1000
        timing_lines = []
        for tier, t_ms in tier_timings.items():
            timing_lines.append(f"    {tier}: {t_ms:.1f}ms")
        if timing_lines:
            logger.info("Performance: total=%.1fms | %s", total_ms, " | ".join(timing_lines))

        return results

    def screen_mixture(
        self,
        ctx1: MoleculeContext,
        ctx2: MoleculeContext,
        frac1: float = 0.5,
        ctx3: MoleculeContext | None = None,
        frac2: float | None = None,
    ) -> dict[str, Any]:
        """Score a binary or ternary mixture using thermodynamic mixing rules with synergy bonus.

        Delegates to ``aurelius.mixer.screen_mixture``.
        """
        return _screen_mixture(self.screen_molecule, ctx1, ctx2, frac1, ctx3, frac2)

    @staticmethod
    def _check_hydrolytic_instability(mol: Chem.Mol) -> float:
        return _check_hydrolytic_instability(mol)

    @staticmethod
    def _check_hypofluorite_instability(mol: Chem.Mol) -> float:
        return _check_hypofluorite_instability(mol)

    @staticmethod
    def _check_al_corrosion_risk(mol: Chem.Mol) -> float:
        return _check_al_corrosion_risk(mol)

    @staticmethod
    def _check_building_block_grounding(mol: Chem.Mol) -> float:
        return _check_building_block_grounding(mol)

    @staticmethod
    def _compute_score(
        homo_eV: float = -99.0,
        lumo_eV: float = -99.0,
        dielectric_proxy: float = 0.0,
        viscosity_proxy: float = 99.0,
        li_solvation_proxy: float = 0.0,
        li_binding_energy_kcal: float = 0.0,
        ced_proxy: float = 0.0,
        sei_fracture_toughness_proxy: float = 0.0,
        gas_evolution_proxy: float = 0.0,
        hydrolysis_risk_proxy: float = 0.0,
        ctx: MoleculeContext | None = None,
        quantum_confidence: str = "unknown",
    ) -> dict[str, Any]:
        return _compute_score(
            homo_eV, lumo_eV,
            dielectric_proxy=dielectric_proxy,
            viscosity_proxy=viscosity_proxy,
            li_solvation_proxy=li_solvation_proxy,
            li_binding_energy_kcal=li_binding_energy_kcal,
            ced_proxy=ced_proxy,
            sei_fracture_toughness_proxy=sei_fracture_toughness_proxy,
            gas_evolution_proxy=gas_evolution_proxy,
            hydrolysis_risk_proxy=hydrolysis_risk_proxy,
            ctx=ctx,
            quantum_confidence=quantum_confidence,
        )

    @staticmethod
    def _apply_domain_penalty(score: dict[str, Any], t2_result: dict[str, Any] | None) -> dict[str, Any]:
        return _apply_domain_penalty(score, t2_result)

    @staticmethod
    def _apply_penalties(
        total_score: float, lumo_eV: float, ctx: MoleculeContext | None
    ) -> float:
        from aurelius.scorer import _apply_penalties
        return _apply_penalties(total_score, lumo_eV, ctx)

    @staticmethod
    def _build_rejection_reasons(
        total_score: float,
        sub_scores: dict[str, float],
        raw_values: dict[str, float],
        is_viable: bool,
        ctx: MoleculeContext | None,
    ) -> list[str]:
        from aurelius.scorer import _build_rejection_reasons
        return _build_rejection_reasons(total_score, sub_scores, raw_values, is_viable, ctx)

    @staticmethod
    def _format_score(score: dict[str, Any]) -> str:
        return _format_score(score)

    def screen_smiles(self, smiles: str) -> dict[str, Any]:
        """Convenience: parse a SMILES string then screen it.

        Args:
            smiles: SMILES string.

        Returns:
            Result dict (same as screen_molecule).
        """
        ctx = MoleculeContext.from_smiles(smiles)
        if ctx is None:
            return self._generate_failed_run(smiles, "Invalid SMILES — parsing failed")
        return self.screen_molecule(ctx)

    def screen_batch(
        self,
        contexts: list[MoleculeContext],
        n_workers: int = 1,
    ) -> list[dict[str, Any]]:
        """Screen a batch of pre-parsed molecules through the full pipeline.

        Args:
            contexts: List of MoleculeContext objects.
            n_workers: Number of parallel workers.

        Returns:
            List of result dicts.
        """
        if n_workers < 1 or n_workers == 1:
            return [self.screen_molecule(ctx) for ctx in contexts]

        results: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(self.screen_molecule, ctx): i
                for i, ctx in enumerate(contexts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        return [results[i] for i in range(len(contexts))]
