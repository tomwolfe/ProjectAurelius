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
    GcUqEnsemble,
    compute_gc_domain_penalty,
    predict_ced_proxy,
    predict_dielectric_proxy,
    predict_ionic_conductivity_proxy,
    predict_li_dissociation_proxy,
    predict_li_solvation_proxy,
    predict_viscosity_proxy,
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
    ) -> None:
        self._quantum = QuantumOracle(use_xtb=use_xtb)
        self._use_surrogate = use_surrogate
        self._surrogate: SurrogateQuantumOracle | None = (
            SurrogateQuantumOracle() if use_surrogate else None
        )
        self._use_gc_uq = use_gc_uq
        self._gc_uq: GcUqEnsemble | None = GcUqEnsemble() if use_gc_uq else None
        self._cache: dict[str, dict[str, Any]] = {}
        self._n_surrogate_skips = 0

    @property
    def quantum_method(self) -> str:
        return self._quantum.method

    def _run_surrogate(self, ctx: MoleculeContext) -> tuple[float, float, float, bool]:
        """Run surrogate pre-filter. Returns (surrogate_penalty, s_homo, s_lumo, skip_quantum)."""
        if self._surrogate is None:
            return 1.0, -99.0, 99.0, False
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

    def _compute_uq_penalty(self, ctx: MoleculeContext) -> float:
        """Compute GC uncertainty penalty."""
        if self._gc_uq is None:
            return 1.0
        try:
            _diel_mean, diel_std, _ = self._gc_uq.predict_dielectric(ctx)
            _visc_mean, visc_std, _ = self._gc_uq.predict_viscosity(ctx)
            if diel_std > max(1.0, abs(_diel_mean)) * _UQ_THRESHOLD_FRACTION:
                return _UQ_PENALTY
            if visc_std > max(1.0, abs(_visc_mean)) * _UQ_THRESHOLD_FRACTION:
                return _UQ_PENALTY
        except Exception:
            logger.debug("GC UQ unavailable")
        return 1.0

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

    def evaluate(self, ctx: MoleculeContext) -> dict[str, Any]:
        if not isinstance(ctx, MoleculeContext):
            raise TypeError(
                f"PropertyOracle.evaluate() requires a MoleculeContext, got {type(ctx).__name__}. "
                "Use MoleculeContext.from_smiles() to parse SMILES first."
            )
        smiles = ctx.smiles
        if smiles in self._cache:
            return self._cache[smiles]

        surrogate_penalty, s_homo, s_lumo, skip_quantum = self._run_surrogate(ctx)
        homo, lumo, gap, quantum_method, quantum_confidence_val = self._compute_quantum(ctx, skip_quantum, s_homo, s_lumo)
        uq_penalty = self._compute_uq_penalty(ctx)

        dielectric = predict_dielectric_proxy(ctx)
        viscosity = predict_viscosity_proxy(ctx)
        li_solvation = predict_li_solvation_proxy(ctx)
        li_dissociation = predict_li_dissociation_proxy(ctx)
        ced = predict_ced_proxy(ctx)
        conductivity = predict_ionic_conductivity_proxy(dielectric, viscosity, li_solvation)

        domain_penalty, domain_reason_str, domain_applicable = self._build_domain(
            ctx, skip_quantum, surrogate_penalty, s_homo, uq_penalty,
        )

        sei_penalty, sei_reason = _evaluate_sei_motif(ctx, lumo)
        if sei_penalty < 1.0:
            domain_penalty *= sei_penalty
            if domain_reason_str and domain_reason_str != _DATA_SOURCE:
                domain_reason_str += "; " + sei_reason
            else:
                domain_reason_str = sei_reason
            domain_applicable = domain_penalty >= 0.85

        result: dict[str, Any] = {
            "homo_eV": round(homo, 4),
            "lumo_eV": round(lumo, 4),
            "gap_eV": round(gap, 4),
            "dielectric_proxy": round(dielectric, 4),
            "viscosity_proxy": round(viscosity, 4),
            "li_solvation_proxy": round(li_solvation, 4),
            "li_dissociation_proxy": round(li_dissociation, 4),
            "ced_proxy": round(ced, 4),
            "conductivity_proxy": round(conductivity, 4),
            "domain_applicable": domain_applicable,
            "domain_reason": domain_reason_str,
            "domain_penalty": round(domain_penalty, 4),
            "quantum_method": quantum_method,
            "quantum_confidence": quantum_confidence_val,
        }
        if self._surrogate is not None:
            result["surrogate_skipped"] = skip_quantum
        self._cache[smiles] = result
        return result

    def evaluate_smiles(self, smiles: str) -> dict[str, Any]:
        ctx = MoleculeContext.from_smiles(smiles)
        if ctx is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return self.evaluate(ctx)

    def save(self, path: str = "oracle_cache.joblib") -> None:
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
        logger.info("PropertyOracle: cache loaded from %s", path)
        return True

    def clear_cache(self) -> None:
        self._cache.clear()
        self._quantum.clear_cache()
