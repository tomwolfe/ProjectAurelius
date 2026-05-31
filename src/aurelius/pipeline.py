"""Aurelius Pipeline Orchestrator.

Coordinates a streamlined two-step discovery pipeline:
  1. **Filter** — Quick structural validity (Tier 1) check with LogP and MW gates.
  2. **Oracle** — Multi-objective property evaluation via fragment-additivity
     (HOMO, LUMO, Dielectric proxy, Viscosity proxy, SA Score).

The multi-objective composite score weights five objectives for electrolyte fitness.
All stages accept a pre-parsed RDKit Mol object to avoid redundant parsing.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer

from aurelius.config import AureliusConfig, apply_global_config
from aurelius.constants import (
    DIELECTRIC_SIGMOID_STEEPNESS,
    DIELECTRIC_TARGET,
    HOMO_SIGMOID_STEEPNESS,
    HOMO_THRESHOLD,
    LUMO_SIGMA,
    LUMO_TARGET,
    SA_SIGMOID_STEEPNESS,
    SA_THRESHOLD,
    SCORE_WEIGHT_DIELECTRIC,
    SCORE_WEIGHT_HOMO,
    SCORE_WEIGHT_LUMO,
    SCORE_WEIGHT_SA,
    SCORE_WEIGHT_VISCOSITY,
    VIABILITY_THRESHOLD,
    VISCOSITY_SIGMOID_STEEPNESS,
    VISCOSITY_THRESHOLD,
)
from aurelius.scoring.oracle import PropertyOracle
from aurelius.screening.tier1 import Filter
from aurelius.types import MoleculeContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hydrolytically unstable SMARTS patterns
# ---------------------------------------------------------------------------
# These motifs degrade in battery electrolyte environments via hydrolysis
# or decomposition, making them unsuitable for long-cycle-life cells.
_HYDROLYTICALLY_UNSTABLE_SMARTS: list[tuple[str, str, float]] = [
    ("[CX3](=[OX1])[OX2][CX3](=[OX1])[OX2]", "anhydride", 0.3),
    ("[CX3](=[OX1])[OX2][CX2]#[N]", "acyl_cyanide", 0.4),
    ("[SX4](=[OX1])(=[OX1])[OX2][CX3](=[OX1])", "sulfonate_ester", 0.2),
    ("[PX4](=[OX1])([OX2][CX4])[OX2][CX4]", "phosphate_ester", 0.15),
    ("[Si]([OX2])[OX2]", "silyl_ether", 0.3),
    ("[CX3](=[OX1])[OX2][CX2]=[CX2]", "enol_ester", 0.35),
    ("[#6][CX3](=[OX1])[OX2][CX3](=[OX1])[#6]", "geminal_diester", 0.2),
]


class AureliusPipeline:
    """Full Aurelius screening pipeline orchestrator.

    Coordinates the Filter -> Oracle pipeline and computes
    the multi-objective composite Aurelius Score. All stages accept a
    pre-parsed MoleculeContext to avoid redundant RDKit Mol object creation.
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
        self._filter: Filter | None = None
        self._use_real_models = use_real_models
        self._oracle: PropertyOracle | None = None

    def initialize(self) -> None:
        """Initialise all pipeline components."""
        try:
            import rdkit  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "RDKit is required for pipeline initialisation. "
                "Install with: pip install rdkit"
            ) from None

        # Phase 1: Structural filter
        if self._use_real_models:
            try:
                self._filter = Filter()
                logger.info("Tier 1 (Filter): ENABLED")
            except Exception as exc:
                logger.warning("Tier 1 (Filter): DISABLED - %s", exc)
                self._filter = None

        # Phase 2: Oracle for property evaluation (pure GC — no training needed)
        self._oracle = PropertyOracle()
        oracle_cache = "oracle_cache.joblib"
        if not self._oracle.load(oracle_cache):
            logger.info("Oracle (PropertyOracle): no cache found — using GC model directly.")
        else:
            logger.info("Oracle (PropertyOracle): loaded from cache (%s).", oracle_cache)
        logger.info("Oracle (PropertyOracle): ENABLED")

    def _generate_failed_run(self, smiles: str, reason: str) -> dict[str, Any]:
        """Generate a failed run result dict for early-exit scenarios."""
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

    @staticmethod
    def _parse_once(smiles: str) -> MoleculeContext | None:
        """Parse SMILES to Mol exactly once and return a MoleculeContext.

        Args:
            smiles: SMILES string.

        Returns:
            MoleculeContext or None if parsing fails.
        """
        return MoleculeContext.from_smiles(smiles)

    def screen_molecule(
        self,
        smiles_or_ctx: str | MoleculeContext,
    ) -> dict[str, Any]:
        """Run a single molecule through the Filter -> Oracle pipeline.

        Accepts either a SMILES string or a pre-parsed MoleculeContext.
        If a SMILES string is provided, it is parsed into a MoleculeContext
        exactly once at the start, and the Mol object is reused across
        all pipeline stages.

        Args:
            smiles_or_ctx: SMILES string or MoleculeContext.

        Returns:
            Dict with tier results and the final Aurelius score.
        """
        if not self._oracle:
            raise RuntimeError("Pipeline not initialised. Call initialise() first.")

        # Parse SMILES -> Mol exactly once
        if isinstance(smiles_or_ctx, MoleculeContext):
            ctx = smiles_or_ctx
            smiles = ctx.smiles
        else:
            smiles = smiles_or_ctx
            parsed = self._parse_once(smiles)
            if parsed is None:
                return self._generate_failed_run(smiles, "Invalid SMILES — parsing failed")
            ctx = parsed

        logger.info("Processing: %s", smiles)
        pipeline_start = time.perf_counter()

        # Step 1: Filter (structural validity + LogP + MW)
        t1_result = None
        if self._filter:
            t1_start = time.perf_counter()
            t1_result = self._filter.screen_molecule(smiles, mol=ctx.mol)
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

        # Step 2: Oracle (property evaluation — accepts pre-parsed Mol)
        t2_result = None
        homo_eV = -99.0
        lumo_eV = -99.0
        dielectric_proxy = 0.0
        viscosity_proxy = 99.0
        if self._oracle:
            t2_start = time.perf_counter()
            oracle_result = self._oracle.evaluate(smiles, mol=ctx.mol)
            tier_timings["tier2_ms"] = (time.perf_counter() - t2_start) * 1000

            homo_eV = oracle_result.get("homo_eV", -99.0)
            lumo_eV = oracle_result.get("lumo_eV", -99.0)
            dielectric_proxy = oracle_result.get("dielectric_proxy", 0.0)
            viscosity_proxy = oracle_result.get("viscosity_proxy", 99.0)

            t2_result = {
                "homo_eV": homo_eV,
                "lumo_eV": lumo_eV,
                "gap_eV": oracle_result.get("gap_eV", 0.0),
                "dielectric_proxy": dielectric_proxy,
                "viscosity_proxy": viscosity_proxy,
                "domain_applicable": oracle_result.get("domain_applicable", True),
                "domain_reason": oracle_result.get("domain_reason", ""),
            }
            results["tier2"] = t2_result
            logger.info(
                "Oracle Result: %s -> HOMO=%.3f LUMO=%.3f Dielectric=%.3f Viscosity=%.3f",
                smiles,
                homo_eV,
                lumo_eV,
                dielectric_proxy,
                viscosity_proxy,
            )

        # Step 3: Score computation (uses pre-parsed Mol)
        score = self._compute_score(
            homo_eV, lumo_eV,
            dielectric_proxy=dielectric_proxy,
            viscosity_proxy=viscosity_proxy,
            smiles=smiles, mol=ctx.mol,
        )
        results["score"] = score

        logger.debug("Scorecard:\n%s", self._format_score(score))

        total_ms = (time.perf_counter() - pipeline_start) * 1000
        timing_lines = []
        for tier, t_ms in tier_timings.items():
            timing_lines.append(f"    {tier}: {t_ms:.1f}ms")
        if timing_lines:
            logger.info("Performance: total=%.1fms | %s", total_ms, " | ".join(timing_lines))

        return results

    @staticmethod
    def _check_hydrolytic_instability(mol: Chem.Mol) -> float:
        """Check for hydrolytically unstable motifs and return a penalty multiplier.

        Scans the molecule for SMARTS patterns known to degrade in battery
        electrolyte environments. Returns a multiplier in [0.5, 1.0].
        """
        penalty = 1.0
        for smarts, name, severity in _HYDROLYTICALLY_UNSTABLE_SMARTS:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is not None and mol.HasSubstructMatch(pattern):
                penalty *= (1.0 - severity)
                logger.debug("Hydrolytic instability detected: %s (penalty %.2f)", name, severity)
        return max(penalty, 0.5)

    @staticmethod
    def _gaussian_lumo(lumo_eV: float) -> float:
        """Gaussian reward for LUMO in the electrochemical stability window.

        Centered at -1.0 eV with sigma=0.75, rewarding LUMO ∈ [-1.75, -0.25] eV
        which covers the typical SEI formation window for Li/Na electrolytes.
        """
        return float(np.exp(-0.5 * ((lumo_eV - LUMO_TARGET) / LUMO_SIGMA) ** 2))

    @staticmethod
    def _sigmoid_homo(homo_eV: float) -> float:
        """Sigmoid penalty for HOMO above threshold.

        When HOMO < HOMO_THRESHOLD (oxidative stability), score -> 1.0.
        When HOMO > HOMO_THRESHOLD, score falls to 0.0.
        """
        return float(1.0 / (1.0 + np.exp(HOMO_SIGMOID_STEEPNESS * (homo_eV - HOMO_THRESHOLD))))

    @staticmethod
    def _sigmoid_dielectric(dielectric_proxy: float) -> float:
        """Sigmoid reward for high dielectric constant (salt dissolution capability).

        Returns a value in [0, 1] where higher dielectric -> higher reward.
        """
        return float(1.0 / (1.0 + np.exp(-DIELECTRIC_SIGMOID_STEEPNESS * (dielectric_proxy - DIELECTRIC_TARGET))))

    @staticmethod
    def _sigmoid_viscosity(viscosity_proxy: float) -> float:
        """Sigmoid penalty for high viscosity (poor ion mobility).

        Returns a value in [0, 1] where higher viscosity -> lower score.
        """
        return float(1.0 / (1.0 + np.exp(VISCOSITY_SIGMOID_STEEPNESS * (viscosity_proxy - VISCOSITY_THRESHOLD))))

    @staticmethod
    def _sigmoid_sa(sa_score: float) -> float:
        """Sigmoid penalty for difficult synthetic accessibility.

        Returns a value in [0, 1] where harder synthesis -> lower score.
        """
        return float(1.0 / (1.0 + np.exp(SA_SIGMOID_STEEPNESS * (sa_score - SA_THRESHOLD))))

    def _compute_score(
        self,
        homo_eV: float = -99.0,
        lumo_eV: float = -99.0,
        dielectric_proxy: float = 0.0,
        viscosity_proxy: float = 99.0,
        smiles: str | None = None,
        mol: Chem.Mol | None = None,
    ) -> dict[str, Any]:
        """Compute the multi-objective composite Aurelius Score.

        Five weighted objectives:
          - **LUMO reward** (30%): Gaussian centered at LUMO_TARGET eV (SEI formation)
          - **HOMO penalty** (20%): Sigmoid penalizing HOMO > HOMO_THRESHOLD eV (oxidative stability)
          - **Dielectric reward** (25%): Sigmoid rewarding high dielectric (salt dissolution)
          - **Viscosity penalty** (15%): Sigmoid penalizing high viscosity (ion mobility)
          - **SA penalty** (10%): Sigmoid penalizing poor synthetic accessibility

        Each resolved sub-score is in [0, 1]. The weighted sum is mapped to [0, 100].

        Args:
            homo_eV: Predicted HOMO energy in eV.
            lumo_eV: Predicted LUMO energy in eV.
            dielectric_proxy: Predicted dielectric constant proxy.
            viscosity_proxy: Predicted viscosity proxy.
            smiles: Optional SMILES for context.
            mol: Pre-parsed RDKit Mol object for substructure checks.

        Returns:
            Dict with ``total_score``, ``is_viable``, ``sub_scores``,
            and ``rejection_reasons``.
        """
        sub_scores: dict[str, float] = {}

        g_lumo = self._gaussian_lumo(lumo_eV)
        s_homo = self._sigmoid_homo(homo_eV)
        sub_scores["lumo_reward"] = round(g_lumo, 4)
        sub_scores["homo_penalty"] = round(s_homo, 4)

        s_dielectric = self._sigmoid_dielectric(dielectric_proxy)
        sub_scores["dielectric_reward"] = round(s_dielectric, 4)

        s_viscosity = self._sigmoid_viscosity(viscosity_proxy)
        sub_scores["viscosity_penalty"] = round(s_viscosity, 4)

        # Compute SA score from RDKit sascorer (replaces custom heuristic)
        sa_score: float = 5.0  # neutral default
        if mol is not None:
            try:
                sa_score = float(sascorer.calculateScore(mol))
            except Exception:
                sa_score = 5.0
        elif smiles is not None:
            m = Chem.MolFromSmiles(smiles)
            if m is not None:
                try:
                    sa_score = float(sascorer.calculateScore(m))
                except Exception:
                    sa_score = 5.0
        s_sa = self._sigmoid_sa(sa_score)
        sub_scores["sa_penalty"] = round(s_sa, 4)

        # Weighted composite
        total_score = 100.0 * (
            SCORE_WEIGHT_LUMO * g_lumo
            + SCORE_WEIGHT_HOMO * s_homo
            + SCORE_WEIGHT_DIELECTRIC * s_dielectric
            + SCORE_WEIGHT_VISCOSITY * s_viscosity
            + SCORE_WEIGHT_SA * s_sa
        )

        # Hydrolytic instability penalty (multiplicative, applied after weighting)
        if mol is not None:
            hydro_penalty = self._check_hydrolytic_instability(mol)
            total_score *= hydro_penalty
        elif smiles is not None:
            m = Chem.MolFromSmiles(smiles)
            if m is not None:
                hydro_penalty = self._check_hydrolytic_instability(m)
                total_score *= hydro_penalty

        total_score = float(np.clip(total_score, 0.0, 100.0))

        is_viable = total_score >= VIABILITY_THRESHOLD

        rejection_reasons: list[str] = []
        if not is_viable:
            reasons = []
            if g_lumo < 0.3:
                reasons.append(f"LUMO={lumo_eV:.3f}eV (poor SEI formation)")
            if s_homo < 0.3:
                reasons.append(f"HOMO={homo_eV:.3f}eV (oxidative instability)")
            if s_dielectric < 0.3:
                reasons.append(f"dielectric_proxy={dielectric_proxy:.3f} (poor salt dissolution)")
            if s_viscosity < 0.3:
                reasons.append(f"viscosity_proxy={viscosity_proxy:.3f} (poor ion mobility)")
            if s_sa < 0.3:
                reasons.append(f"SA score={sa_score:.2f} (hard to synthesize)")
            rejection_reasons.append(
                f"Aurelius Score {total_score:.1f} below threshold: {'; '.join(reasons)}"
            )

        return {
            "total_score": total_score,
            "is_viable": is_viable,
            "sub_scores": sub_scores,
            "sa_score": round(sa_score, 4),
            "rejection_reasons": rejection_reasons,
        }

    @staticmethod
    def _format_score(score: dict[str, Any]) -> str:
        """Format a score dict for logging."""
        total = score.get("total_score", 0.0)
        viable = score.get("is_viable", False)
        return f"Score: {total:.1f}/100 {'VIABLE' if viable else 'REJECTED'}"

    def screen_batch(
        self,
        smiles_or_ctx_list: list[str | MoleculeContext],
        n_workers: int = 1,
    ) -> list[dict[str, Any]]:
        """Screen a batch of molecules through the full pipeline.

        When ``n_workers`` is greater than 1, molecules are screened
        in parallel using ``ThreadPoolExecutor``.
        """
        if n_workers < 1 or n_workers == 1:
            return [self.screen_molecule(smi) for smi in smiles_or_ctx_list]

        results: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(self.screen_molecule, smi): i
                for i, smi in enumerate(smiles_or_ctx_list)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        return [results[i] for i in range(len(smiles_or_ctx_list))]
