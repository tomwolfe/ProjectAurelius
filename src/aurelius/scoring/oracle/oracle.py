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

from aurelius.scoring.oracle.gc import (
    _DATA_SOURCE,
    compute_gc_domain_penalty,
    predict_dielectric_proxy,
    predict_ionic_conductivity_proxy,
    predict_li_solvation_proxy,
    predict_viscosity_proxy,
)
from aurelius.scoring.oracle.quantum import (
    QuantumOracle,
    compute_quantum_domain_penalty,
)
from aurelius.types import MoleculeContext

logger = logging.getLogger(__name__)


class PropertyOracle:
    """Multi-objective property oracle with a hybrid physics model.

    Architecture:
      - HOMO / LUMO / Gap: QuantumOracle (xTB preferred, TOM fallback)
      - Dielectric proxy: GC fragment-additivity + TPSA-based cap
      - Viscosity proxy: GC fragment-additivity + MW + rotatable bonds
      - Li+ Solvation proxy: GC fragment-additivity
    """

    def __init__(self, use_xtb: bool = True) -> None:
        self._quantum = QuantumOracle(use_xtb=use_xtb)
        self._cache: dict[str, dict[str, Any]] = {}

    @property
    def quantum_method(self) -> str:
        return self._quantum.method

    def evaluate(self, ctx: MoleculeContext) -> dict[str, Any]:
        if not isinstance(ctx, MoleculeContext):
            raise TypeError(
                f"PropertyOracle.evaluate() requires a MoleculeContext, got {type(ctx).__name__}. "
                "Use MoleculeContext.from_smiles() to parse SMILES first."
            )

        smiles = ctx.smiles
        if smiles in self._cache:
            return self._cache[smiles]

        quantum_result = self._quantum.evaluate(ctx.mol)
        homo = quantum_result["homo_eV"]
        lumo = quantum_result["lumo_eV"]
        gap = lumo - homo

        dielectric = predict_dielectric_proxy(ctx)
        viscosity = predict_viscosity_proxy(ctx)
        li_solvation = predict_li_solvation_proxy(ctx)
        conductivity = predict_ionic_conductivity_proxy(dielectric, viscosity, li_solvation)

        # Domain of applicability penalties
        q_penalty, q_reason = compute_quantum_domain_penalty(ctx)
        gc_penalty, gc_reason = compute_gc_domain_penalty(ctx)
        domain_penalty = min(q_penalty, gc_penalty)
        domain_reasons: list[str] = []
        if q_penalty < 1.0:
            domain_reasons.append(f"quantum: {q_reason}")
        if gc_penalty < 1.0:
            domain_reasons.append(f"GC: {gc_reason}")
        domain_applicable = domain_penalty >= 0.85
        domain_reason_str = "; ".join(domain_reasons) if domain_reasons else _DATA_SOURCE

        # Apply physical sanity bounds to ensure realistic predictions
        sanity_warning: list[str] = []
        clamped_values = {
            "dielectric_proxy": dielectric,
            "viscosity_proxy": viscosity,
            "homo_eV": homo,
            "lumo_eV": lumo,
        }

        # Apply bounds per property
        if dielectric < 1.0:
            clamped_values["dielectric_proxy"] = 1.0
            sanity_warning.append("dielectric_proxy below physical minimum (clamped to 1.0)")
        elif dielectric > 100.0:
            clamped_values["dielectric_proxy"] = 100.0
            sanity_warning.append("dielectric_proxy above physical maximum (clamped to 100.0)")

        if viscosity < 0.1:
            clamped_values["viscosity_proxy"] = 0.1
            sanity_warning.append("viscosity_proxy below physical minimum (clamped to 0.1)")
        elif viscosity > 50.0:
            clamped_values["viscosity_proxy"] = 50.0
            sanity_warning.append("viscosity_proxy above physical maximum (clamped to 50.0)")

        if homo < -12.0:
            clamped_values["homo_eV"] = -12.0
            sanity_warning.append("homo_eV below physical minimum (clamped to -12.0)")
        elif homo > -3.0:
            clamped_values["homo_eV"] = -3.0
            sanity_warning.append("homo_eV above physical maximum (clamped to -3.0)")

        if lumo < -5.0:
            clamped_values["lumo_eV"] = -5.0
            sanity_warning.append("lumo_eV below physical minimum (clamped to -5.0)")
        elif lumo > 5.0:
            clamped_values["lumo_eV"] = 5.0
            sanity_warning.append("lumo_eV above physical maximum (clamped to 5.0)")

        result: dict[str, Any] = {
            "homo_eV": round(clamped_values["homo_eV"], 4),
            "lumo_eV": round(clamped_values["lumo_eV"], 4),
            "gap_eV": round(clamped_values["lumo_eV"] - clamped_values["homo_eV"], 4),
            "dielectric_proxy": round(clamped_values["dielectric_proxy"], 4),
            "viscosity_proxy": round(clamped_values["viscosity_proxy"], 4),
            "li_solvation_proxy": round(li_solvation, 4),
            "conductivity_proxy": round(conductivity, 4),
            "domain_applicable": domain_applicable,
            "domain_reason": domain_reason_str,
            "domain_penalty": round(domain_penalty, 4),
            "quantum_method": self._quantum.method,
            "quantum_confidence": quantum_result.get("quantum_confidence", "unknown"),
            "sanity_warning": sanity_warning,
        }

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
