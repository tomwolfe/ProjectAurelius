"""Aurelius Pipeline Orchestrator.

Coordinates a streamlined two-step discovery pipeline:
  1. **Filter** — Quick structural validity (Tier 1) check.
  2. **Oracle** — Evaluate HOMO/LUMO frontier orbital energies via the PropertyOracle.

The results are then fed back to the RF surrogate for Bayesian optimisation.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

from aurelius.config import AureliusConfig, apply_global_config
from aurelius.scoring.oracle import PropertyOracle
from aurelius.screening.tier1 import Filter
from aurelius.utils.dependencies import HAS_RDKIT

logger = logging.getLogger(__name__)


class AureliusPipeline:
    """Full Aurelius screening pipeline orchestrator.

    Coordinates the Filter -> Oracle pipeline and computes
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

        # Phase 2: Oracle for property evaluation (with cache check)
        self._oracle = PropertyOracle()
        oracle_cache = "oracle_cache.joblib"
        if not self._oracle.load(oracle_cache):
            logger.info("Oracle (PropertyOracle): no cache found — training from scratch.")
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

        lumo_score = self._oracle.predict_normalized_lumo(smiles) if self._oracle else 0.0
        total_score = lumo_score
        is_viable = total_score >= 50.0

        return {
            "tier1": t1_result,
            "tier2": None,
            "score": {
                "total_score": total_score,
                "lumo_score": lumo_score,
                "is_viable": is_viable,
                "rejection_reasons": [reason],
            },
        }

    def screen_molecule(self, smiles: str) -> dict[str, Any]:
        """Run a single molecule through the Filter -> Oracle pipeline.

        Returns a dict with tier results and the final Aurelius score.
        Includes per-tier timing metrics for performance monitoring.
        """
        if not self._oracle:
            raise RuntimeError("Pipeline not initialised. Call initialise() first.")

        logger.info("Processing: %s", smiles)
        pipeline_start = time.perf_counter()

        # Step 1: Filter (structural validity + SA score)
        t1_result = None
        if self._filter:
            t1_start = time.perf_counter()
            t1_result = self._filter.screen_molecule(smiles)
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

        # Step 2: Oracle (property evaluation)
        t2_result = None
        lumo_score = 0.0
        homo_eV = -99.0
        lumo_eV = -99.0
        domain_applicable = True
        domain_penalty = 1.0
        if self._oracle:
            t2_start = time.perf_counter()
            oracle_result = self._oracle.evaluate(smiles)
            lumo_score = self._oracle.predict_normalized_lumo(smiles)
            tier_timings["tier2_ms"] = (time.perf_counter() - t2_start) * 1000

            homo_eV = oracle_result.get("homo_eV", -99.0)
            lumo_eV = oracle_result.get("lumo_eV", -99.0)
            domain_applicable = oracle_result.get("domain_applicable", True)
            domain_reason = oracle_result.get("domain_reason", "")
            domain_penalty = oracle_result.get("domain_penalty", 1.0)

            t2_result = {
                "homo_eV": homo_eV,
                "lumo_eV": lumo_eV,
                "gap_eV": oracle_result.get("gap_eV", 0.0),
                "score_eV": oracle_result.get("score_eV", 0.0),
                "lumo_score": lumo_score,
                "domain_applicable": domain_applicable,
                "domain_reason": domain_reason,
                "domain_penalty": domain_penalty,
            }
            results["tier2"] = t2_result
            logger.info(
                "Property Oracle Result: %s -> HOMO=%.3f LUMO=%.3f gap=%.3f "
                "lumo_score=%.1f",
                smiles,
                homo_eV,
                lumo_eV,
                t2_result["gap_eV"],
                lumo_score,
            )

        score = self._compute_score(
            lumo_score, homo_eV, lumo_eV,
            smiles=smiles, domain_applicable=domain_applicable,
            domain_penalty=domain_penalty,
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
    def _compute_sa_score(smiles: str) -> float:
        """Compute synthetic accessibility score in [0, 1] (1 = easy to synthesise).

        Uses RDKit's sascorer if available, otherwise falls back to a
        fragment-complexity heuristic based on rings, stereocenters,
        and rotatable bonds.

        Returns:
            0.0–1.0 where 1.0 = trivially synthesizable.
        """
        if not HAS_RDKIT:
            return 1.0
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 1.0

        # Try RDKit Contrib sascorer first
        try:
            from rdkit.Contrib.SA_Score import sascorer  # type: ignore[import-not-found]

            raw = sascorer.calculateScore(mol)
            # sascorer returns ~1 (easy) to ~10 (hard); map to [0, 1]
            return max(0.0, 1.0 - (raw - 1.0) / 9.0)
        except ImportError:
            pass

        # Fallback heuristic: penalize structural complexity
        from rdkit.Chem import rdMolDescriptors

        n_rings = rdMolDescriptors.CalcNumRings(mol)
        n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
        n_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        n_het = rdMolDescriptors.CalcNumHeterocycles(mol)

        # Heuristic: fewer rings, rotors, stereocenters, and heterocycles = easier
        complexity = n_rings * 0.15 + n_rot * 0.05 + n_stereo * 0.2 + n_het * 0.1
        raw = min(complexity, 5.0) / 5.0  # normalize to [0, 1]
        return max(0.0, 1.0 - raw)

    @staticmethod
    def _gaussian_lumo(lumo_eV: float) -> float:
        """Gaussian reward for LUMO in the electrochemical stability window.

        Centered at -1.0 eV with sigma=0.75, rewarding LUMO ∈ [-1.75, -0.25] eV
        which covers the typical SEI formation window for Li/Na electrolytes.
        """
        return float(np.exp(-0.5 * ((lumo_eV + 1.0) / 0.75) ** 2))

    @staticmethod
    def _sigmoid_homo(homo_eV: float) -> float:
        """Sigmoid penalty for HOMO above -6.0 eV.

        When HOMO < -6.0 eV (oxidative stability), score → 1.0.
        When HOMO > -6.0 eV, score falls to 0.0 with steepness k=5.
        """
        return float(1.0 / (1.0 + np.exp(5.0 * (homo_eV + 6.0))))

    def _compute_score(
        self,
        lumo_score: float,
        homo_eV: float = -99.0,
        lumo_eV: float = -99.0,
        smiles: str | None = None,
        domain_applicable: bool = True,
        domain_penalty: float = 1.0,
    ) -> dict[str, Any]:
        """Compute the final Aurelius Score using Gaussian penalty approach.

        Scoring:
          - LUMO Gaussian reward centered at -1.0 eV, sigma=0.75 (SEI formation)
          - HOMO sigmoid penalty for oxidative stability (threshold -6.0 eV)
          - Synthetic accessibility penalty for novel but synthesisable molecules
          - Domain applicability penalty for OOD molecules (element-based or
            fingerprint-based)

        The final score is in [0, 100].

        Args:
            lumo_score: Normalized LUMO score from oracle (0-100).
            homo_eV: Predicted HOMO energy in eV.
            lumo_eV: Predicted LUMO energy in eV.
            smiles: Optional SMILES for synthetic accessibility penalty.
            domain_applicable: Whether molecule is within QM9 applicability domain.
            domain_penalty: Score multiplier from OOD checks (1.0 = in-domain,
                0.5 = element OOD, 0.9 = fingerprint OOD, 0.0 = hard reject).

        Returns:
            Dict with ``total_score``, ``lumo_score``,
            ``is_viable``, and ``rejection_reasons``.
        """
        g_lumo = self._gaussian_lumo(lumo_eV)
        s_homo = self._sigmoid_homo(homo_eV)

        total_score = 100.0 * g_lumo * s_homo

        # SA penalty — penalise molecules that are impossible to synthesise
        if smiles is not None:
            sa = self._compute_sa_score(smiles)
            total_score *= sa

        # Domain applicability penalty
        # The RF model was trained only on QM9 (CHON ± trace F). Battery
        # electrolytes with heavy fluorination, sulfonation, or other
        # unseen elements cause the RF to hallucinate. The domain_penalty
        # applies a score reduction proportional to OOD severity:
        #   1.0  = in-domain (no penalty)
        #   0.9  = fingerprint OOD (mild Tanimoto dissimilarity)
        #   0.5  = element OOD (F>3, any S/P/etc — model extrapolating)
        #   0.0  = invalid SMILES (hard reject)
        if not domain_applicable and domain_penalty < 1.0:
            penalty_pct = int((1.0 - domain_penalty) * 100)
            total_score *= domain_penalty
            logger.warning(
                "OOD penalty applied (%d%% reduction): domain_penalty=%.1f",
                penalty_pct, domain_penalty,
            )

        total_score = float(np.clip(total_score, 0.0, 100.0))

        is_viable = total_score >= 50.0

        rejection_reasons: list[str] = []
        if not is_viable:
            rejection_reasons.append(
                f"Aurelius Score {total_score:.1f} below viability threshold "
                f"(g_lumo={g_lumo:.3f}, s_homo={s_homo:.3f})"
            )

        return {
            "total_score": total_score,
            "lumo_score": lumo_score,
            "is_viable": is_viable,
            "rejection_reasons": rejection_reasons,
        }

    @staticmethod
    def _format_score(score: dict[str, Any]) -> str:
        """Format a score dict for logging."""
        total = score.get("total_score", 0.0)
        viable = score.get("is_viable", False)
        return f"Score: {total:.1f}/100 {'VIABLE' if viable else 'REJECTED'}"

    def screen_batch(self, smiles_list: list[str], n_workers: int = 1) -> list[dict[str, Any]]:
        """Screen a batch of molecules through the full pipeline.

        When ``n_workers`` is greater than 1, molecules are screened
        in parallel using ``ThreadPoolExecutor``.
        """
        if n_workers < 1 or n_workers == 1:
            return [self.screen_molecule(smiles) for smiles in smiles_list]

        results: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(self.screen_molecule, smiles): i
                for i, smiles in enumerate(smiles_list)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        return [results[i] for i in range(len(smiles_list))]
