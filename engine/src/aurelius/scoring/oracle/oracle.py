"""PropertyOracle — Hybrid fragment-additivity + quantum chemistry oracle.

Usage:
    from aurelius.scoring.oracle.oracle import PropertyOracle
    from aurelius.types import MoleculeContext

    ctx = MoleculeContext.from_smiles("CC(=O)OC1=CC=CC=C1")
    result = oracle.evaluate(ctx)
    print(result["homo_eV"])            # e.g. -7.6 (quantum)
    print(result["li_solvation_proxy"])  # e.g. 2.3 (GC)
"""

from __future__ import annotations

import logging
from typing import Any

from aurelius.cache import CacheBackend, DictCache
from aurelius.constants import (
    SEI_LUMO_LOWER,
    SEI_LUMO_UPPER,
    SEI_MOTIF_PENALTY_FACTOR,
    STABLE_SEI_MOTIFS,
)
from aurelius.scoring.oracle.gc import (
    _DATA_SOURCE,
    _UQ_PENALTY,
    _UQ_THRESHOLD_FRACTION,
    BasePropertyModel,
    ElectrolytePack,
    GcUqEnsemble,
    compute_gc_domain_penalty,
)
from aurelius.scoring.oracle.quantum import (
    QuantumOracle,
    compute_quantum_domain_penalty,
)
from aurelius.scoring.oracle.surrogate import SurrogateQuantumOracle
from aurelius.types import MoleculeContext

logger = logging.getLogger(__name__)


def _evaluate_sei_motif(ctx: MoleculeContext, lumo: float) -> tuple[float, str]:
    """Check if a molecule has known stable SEI-forming motifs.

    Physical basis: A molecule with LUMO in the SEI formation window
    (> SEI_LUMO_LOWER and < SEI_LUMO_UPPER) is thermodynamically capable
    of reductive decomposition to form an SEI. However, without known
    stable SEI-forming functional groups (e.g., cyclic carbonates, CF3,
    sultones), the resulting SEI may be unstable or poorly passivating.
    This check penalises molecules in the SEI window that lack these motifs.

    Returns:
        (penalty_multiplier, reason_string)
        Multiplier in [SEI_MOTIF_PENALTY_FACTOR, 1.0]; 1.0 = has stable motif or
        LUMO outside SEI window.
    """
    if not (SEI_LUMO_LOWER < lumo < SEI_LUMO_UPPER):
        return 1.0, ""
    mol = ctx.mol
    for pattern in STABLE_SEI_MOTIFS:
        if pattern is not None and mol.HasSubstructMatch(pattern):
            return 1.0, ""
    return SEI_MOTIF_PENALTY_FACTOR, "Lacks stable SEI-forming motif"


class PropertyOracle:
    """Multi-objective property oracle with a hybrid physics model.

    Architecture:
      - HOMO / LUMO / Gap: QuantumOracle (xTB preferred, TOM fallback)
      - Dielectric proxy: GC fragment-additivity + TPSA-based cap
      - Viscosity proxy: GC fragment-additivity + MW + rotatable bonds
      - Li+ Solvation proxy: GC fragment-additivity
    """

    def __init__(
        self,
        use_xtb: bool = True,
        use_surrogate: bool = True,
        use_gc_uq: bool = True,
        property_pack: BasePropertyModel | None = None,
        l2_cache: CacheBackend | None = None,
    ) -> None:
        self._quantum = QuantumOracle(use_xtb=use_xtb)
        self._use_surrogate = use_surrogate
        self._surrogate: SurrogateQuantumOracle | None = (
            SurrogateQuantumOracle() if use_surrogate else None
        )
        self._use_gc_uq = use_gc_uq
        self._gc_uq: GcUqEnsemble | None = GcUqEnsemble() if use_gc_uq else None
        self._property_pack: BasePropertyModel = property_pack or ElectrolytePack()
        self._cache: dict[str, dict[str, Any]] = {}
        self._disk_cache: CacheBackend = l2_cache or DictCache()
        self._n_surrogate_skips = 0

    @property
    def quantum_method(self) -> str:
        return self._quantum.method

    def _run_surrogate(self, ctx: MoleculeContext) -> tuple[float, float, float, bool]:
        """Run surrogate pre-filter. Returns (surrogate_penalty, s_homo, s_lumo, skip_quantum).

        Two gates control whether the surrogate prediction is trusted:
          1. Structural novelty gate: if the molecule's max Tanimoto similarity
             to the calibration pool is below ``similarity_threshold``, the
             surrogate is considered untrustworthy and quantum is forced.
          2. HOMO threshold gate: if surrogate predicts HOMO > -5.0 eV,
             skip quantum (classic fast-skip path).
        """
        if self._surrogate is None:
            return 1.0, -99.0, 99.0, False
        try:
            if self._surrogate.is_structurally_novel(ctx):
                logger.info(
                    "Surrogate: molecule structurally novel (max Tanimoto < %.2f) — "
                    "forcing quantum oracle",
                    self._surrogate.similarity_threshold,
                )
                return 1.0, -99.0, 99.0, False
        except Exception:
            pass

        try:
            s_homo, s_lumo = self._surrogate.predict(ctx)
            penalty = self._surrogate.compute_penalty(s_homo)
            if penalty < 1.0:
                self._n_surrogate_skips += 1
                logger.info(
                    "Surrogate: HOMO=%.3f eV > %.1f threshold — skipping quantum oracle",
                    s_homo, -5.0,
                )
                return penalty, s_homo, s_lumo, True
            return penalty, s_homo, s_lumo, False
        except Exception:
            logger.debug("Surrogate pre-filter unavailable")
            return 1.0, -99.0, 99.0, False

    def _compute_quantum(self, ctx: MoleculeContext, skip_quantum: bool, s_homo: float, s_lumo: float) -> tuple[float, float, float, str, Any]:
        """Compute or skip quantum evaluation.

        Two-tier architecture:
          1. xTB (via QuantumOracle) — preferred real QM
          2. TOM — closed-form topological fallback (last resort)
        """
        if skip_quantum:
            gap = s_lumo - s_homo
            return s_homo, s_lumo, gap, "surrogate", "surrogate"

        # Tier 1: xTB (if available)
        if self._quantum._use_xtb:
            qr = self._quantum.evaluate(ctx.mol)
            if "conformer_variance" in qr:
                gap = qr["lumo_eV"] - qr["homo_eV"]
                return qr["homo_eV"], qr["lumo_eV"], gap, "xTB (Boltzmann-weighted)", qr.get("quantum_confidence", "xtb")

        # Tier 2: TOM fallback
        qr = self._quantum.evaluate(ctx.mol)
        gap = qr["lumo_eV"] - qr["homo_eV"]
        return qr["homo_eV"], qr["lumo_eV"], gap, "TOM (Topological Orbital Model)", qr.get("quantum_confidence", "tom_low")

    def _compute_uq_penalty(self, ctx: MoleculeContext) -> tuple[float, float, float]:
        """Compute GC uncertainty penalty — graded by number of flagged properties.

        Returns:
            (penalty, diel_std, visc_std) where penalty in [0.81, 1.0].
        """
        if self._gc_uq is None:
            return 1.0, 0.0, 0.0
        try:
            _diel_mean, diel_std, _ = self._gc_uq.predict_dielectric(ctx)
            _visc_mean, visc_std, _ = self._gc_uq.predict_viscosity(ctx)
            n_flags = 0
            if diel_std > max(1.0, abs(_diel_mean)) * _UQ_THRESHOLD_FRACTION:
                n_flags += 1
            if visc_std > max(1.0, abs(_visc_mean)) * _UQ_THRESHOLD_FRACTION:
                n_flags += 1
            if n_flags > 0:
                return _UQ_PENALTY ** n_flags, diel_std, visc_std
            return 1.0, diel_std, visc_std
        except Exception:
            logger.debug("GC UQ unavailable")
        return 1.0, 0.0, 0.0

    def _compute_gc_properties(self, ctx: MoleculeContext) -> dict[str, float]:
        """Compute all GC-based bulk property proxies for a molecule.

        Delegates to the configured property pack (default: ElectrolytePack).
        """
        return self._property_pack.predict_all(ctx)

    def _build_domain(self, ctx: MoleculeContext, skip_quantum: bool, surrogate_penalty: float, s_homo: float, uq_penalty: float) -> tuple[float, str, bool]:
        """Build domain penalty and reason string."""
        q_penalty, q_reason = (1.0, "skipped — surrogate") if skip_quantum else compute_quantum_domain_penalty(ctx)
        gc_penalty, gc_reason = compute_gc_domain_penalty(ctx)
        domain_penalty = min(q_penalty, gc_penalty, surrogate_penalty, uq_penalty)
        reasons: list[str] = []
        if q_penalty < 1.0 and not skip_quantum:
            reasons.append(f"quantum: {q_reason}")
        if gc_penalty < 1.0:
            reasons.append(f"GC: {gc_reason}")
        if surrogate_penalty < 1.0:
            reasons.append(f"surrogate: HOMO={s_homo:.2f} eV > threshold")
        if uq_penalty < 1.0:
            reasons.append("High UQ Variance")
        return domain_penalty, "; ".join(reasons) if reasons else _DATA_SOURCE, domain_penalty >= 0.85

    def _apply_sei_penalty(self, ctx: MoleculeContext, lumo: float, domain_penalty: float, domain_reason_str: str) -> tuple[float, str, bool]:
        """Apply SEI motif penalty and integrate into domain penalty."""
        sei_penalty, sei_reason = _evaluate_sei_motif(ctx, lumo)
        if sei_penalty >= 1.0:
            return domain_penalty, domain_reason_str, domain_penalty >= 0.85
        domain_penalty *= sei_penalty
        if domain_reason_str and domain_reason_str != _DATA_SOURCE:
            domain_reason_str += "; " + sei_reason
        else:
            domain_reason_str = sei_reason
        return domain_penalty, domain_reason_str, domain_penalty >= 0.85

    def _assemble_result(
        self,
        smiles: str,
        homo: float, lumo: float, gap: float,
        gc_props: dict[str, float],
        domain_penalty: float, domain_reason_str: str, domain_applicable: bool,
        quantum_method: str, quantum_confidence_val: str,
        skip_quantum: bool,
        diel_std: float = 0.0,
        visc_std: float = 0.0,
    ) -> dict[str, Any]:
        """Assemble the final evaluation result dict."""
        result: dict[str, Any] = {
            "homo_eV": round(homo, 4),
            "lumo_eV": round(lumo, 4),
            "gap_eV": round(gap, 4),
            "domain_applicable": domain_applicable,
            "domain_drift_risk": domain_penalty < 0.85,
            "domain_reason": domain_reason_str,
            "domain_penalty": round(domain_penalty, 4),
            "quantum_method": quantum_method,
            "quantum_confidence": quantum_confidence_val,
        }
        # Merge property pack-specific proxies
        for key, value in gc_props.items():
            result[key] = round(value, 4)

        # Standard electrolyte uncertainty flag (only if dielectric/viscosity present)
        dielectric = gc_props.get("dielectric_proxy", 0.0)
        viscosity = gc_props.get("viscosity_proxy", 0.0)
        result["uncertainty_flag"] = (
            diel_std > abs(dielectric) * _UQ_THRESHOLD_FRACTION
            or visc_std > abs(viscosity) * _UQ_THRESHOLD_FRACTION
        )
        if self._surrogate is not None:
            result["surrogate_skipped"] = skip_quantum
        return result

    def _cache_key(self, smiles: str) -> str:
        return f"{smiles}::{self._property_pack.name}"

    def evaluate(self, ctx: MoleculeContext) -> dict[str, Any]:
        if not isinstance(ctx, MoleculeContext):
            raise TypeError(
                f"PropertyOracle.evaluate() requires a MoleculeContext, got {type(ctx).__name__}. "
                "Use MoleculeContext.from_smiles() to parse SMILES first."
            )
        smiles = ctx.smiles
        key = self._cache_key(smiles)
        if key in self._cache:
            return self._cache[key]
        cached = self._disk_cache.get(key)
        if cached is not None:
            self._cache[key] = cached
            return cached
        # Fallback for legacy caches (pre-pack-keying)
        if key != smiles:
            cached = self._disk_cache.get(smiles)
            if cached is not None:
                self._cache[key] = cached
                return cached

        surrogate_penalty, s_homo, s_lumo, skip_quantum = self._run_surrogate(ctx)
        homo, lumo, gap, quantum_method, quantum_confidence_val = self._compute_quantum(ctx, skip_quantum, s_homo, s_lumo)
        uq_penalty, diel_std, visc_std = self._compute_uq_penalty(ctx)

        gc_props = self._compute_gc_properties(ctx)

        domain_penalty, domain_reason_str, domain_applicable = self._build_domain(
            ctx, skip_quantum, surrogate_penalty, s_homo, uq_penalty,
        )
        domain_penalty, domain_reason_str, domain_applicable = self._apply_sei_penalty(
            ctx, lumo, domain_penalty, domain_reason_str,
        )

        if domain_penalty < 0.85:
            logger.warning(
                "Warning: Molecule is outside the calibrated domain of the current kernel. "
                "Consider retuning via Aurelius Certification Lab. "
                "(smiles=%s, domain_penalty=%.4f, reason=%s)",
                smiles, domain_penalty, domain_reason_str,
            )

        result = self._assemble_result(
            smiles, homo, lumo, gap,
            gc_props,
            domain_penalty, domain_reason_str, domain_applicable,
            quantum_method, quantum_confidence_val,
            skip_quantum,
            diel_std=diel_std,
            visc_std=visc_std,
        )
        self._cache[key] = result
        self._disk_cache[key] = result
        return result

    def evaluate_smiles(self, smiles: str) -> dict[str, Any]:
        ctx = MoleculeContext.from_smiles(smiles)
        if ctx is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return self.evaluate(ctx)

    def save(self, path: str = "oracle_cache.joblib") -> None:
        self._cache.clear()
        for key in self._disk_cache:
            self._cache[key] = self._disk_cache[key]
        import joblib
        payload: dict[str, Any] = {
            "cache": self._cache,
            "data_source": _DATA_SOURCE,
        }
        joblib.dump(payload, path)
        logger.info("PropertyOracle: cache saved to %s", path)

    def load(self, path: str = "oracle_cache.joblib") -> bool:
        try:
            import joblib
            payload = joblib.load(path)
        except (FileNotFoundError, Exception) as exc:
            logger.debug("PropertyOracle: no cached oracle at %s (%s)", path, exc)
            return False

        loaded_cache = payload.get("cache")
        if loaded_cache is not None:
            self._cache.update(loaded_cache)
            for key, value in loaded_cache.items():
                self._disk_cache[key] = value
        logger.info("PropertyOracle: cache loaded from %s", path)
        return True

    def clear_cache(self) -> None:
        self._cache.clear()
        self._disk_cache.clear()
        self._quantum.clear_cache()
