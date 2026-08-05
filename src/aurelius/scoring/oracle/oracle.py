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

import numpy as np
from rdkit import Chem

from aurelius.scoring.oracle.conformal import get_conformal_predictor
from aurelius.scoring.oracle.gc import (
    _DATA_SOURCE,
    _count_branch_points,
    _count_fragments_batch,
    _count_stereocenters,
    compute_gc_domain_penalty,
    predict_dielectric_proxy,
    predict_dielectric_proxy_batch,
    predict_ionic_conductivity_proxy,
    predict_ionic_conductivity_proxy_batch,
    predict_li_solvation_proxy,
    predict_li_solvation_proxy_batch,
    predict_viscosity_proxy,
    predict_viscosity_proxy_batch,
)
from aurelius.scoring.oracle.quantum import (
    QuantumOracle,
    compute_quantum_domain_penalty,
    predict_tom_orbitals_batch,
)
from aurelius.types import MoleculeContext
from aurelius.utils.device import get_device, to_device, batch_tanimoto as _batch_tanimoto

logger = logging.getLogger(__name__)

_PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "dielectric_proxy": (1.0, 100.0),
    "viscosity_proxy": (0.1, 50.0),
    "homo_eV": (-12.0, -3.0),
    "lumo_eV": (-5.0, 5.0),
}


def _fp_to_numpy(fp: Any, n_bits: int = 2048) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Convert a single RDKit fingerprint to a 1D numpy array.

    Uses numpy fancy indexing instead of a Python for-loop over bits.
    """
    n_on = fp.GetNumOnBits()
    if n_on == 0:
        return np.zeros(n_bits, dtype=np.float32)
    on_bits = np.fromiter(fp.GetOnBits(), dtype=np.int32, count=n_on)
    arr = np.zeros(n_bits, dtype=np.float32)
    arr[on_bits] = 1.0
    return arr


def _fp_batch_to_numpy(fps: list[Any], n_bits: int = 2048) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Convert a list of RDKit fingerprints to a 2D numpy array.

    Uses numpy fancy indexing per fingerprint instead of a Python
    for-loop over individual bits.
    """
    n = len(fps)
    arr = np.zeros((n, n_bits), dtype=np.float32)
    for i, fp in enumerate(fps):
        n_on = fp.GetNumOnBits()
        if n_on == 0:
            continue
        on_bits = np.fromiter(fp.GetOnBits(), dtype=np.int32, count=n_on)
        arr[i, on_bits] = 1.0
    return arr


def _tanimoto_batch_numpy(fps: list[Any], n_bits: int = 2048) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Compute pairwise Tanimoto similarity matrix using numpy (CPU fallback).

    Uses vectorized numpy operations: intersection = arr @ arr.T,
    union = sum(arr, axis=1)[:, None] + sum(arr, axis=1)[None, :].
    Tanimoto = intersection / union, clipped to [0, 1].

    This is the single correct path for all batch sizes since RDKit's
    BulkTanimotoSimilarity does not accept a list-of-fingerprints as
    the first argument in RDKit 2026+.
    """
    n = len(fps)
    if n <= 1:
        return np.ones((n, n), dtype=np.float32)
    arr = _fp_batch_to_numpy(fps, n_bits=n_bits)
    intersections = arr @ arr.T
    sums = arr.sum(axis=1, keepdims=True) + arr.sum(axis=1, keepdims=True).T
    sums[sums == 0] = 1.0
    return np.clip(intersections / sums, 0.0, 1.0).astype(np.float32)


def _tanimoto_batch_mps(fps: list[Any], n_bits: int = 2048) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Compute pairwise Tanimoto similarity using MPS (Apple GPU).

    Uses ``__import__("torch")`` to avoid hard dependency on torch.
    Falls back to numpy if torch is not available or MPS is not present.
    """
    torch = __import__("torch")
    if not torch.backends.mps.is_available():
        return _tanimoto_batch_numpy(fps, n_bits=n_bits)
    arr = _fp_batch_to_numpy(fps, n_bits=n_bits)
    tensor = torch.from_numpy(arr).to("mps")
    intersections = tensor @ tensor.T
    sums = tensor.sum(dim=1, keepdim=True) + tensor.sum(dim=1, keepdim=True).T
    sums = torch.where(sums == 0, torch.ones_like(sums), sums)
    result = torch.clamp(intersections / sums, 0.0, 1.0).float()
    return result.cpu().numpy()


def _tanimoto_batch_mlx(fps: list[Any], n_bits: int = 2048) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Compute pairwise Tanimoto similarity using MLX (Apple GPU).

    Uses ``__import__("mlx")`` to avoid hard dependency on mlx.
    Falls back to numpy if mlx is not available.
    """
    mlx = __import__("mlx")
    mlx_core = __import__("mlx.core")
    arr = _fp_batch_to_numpy(fps, n_bits=n_bits)
    tensor = mlx_core.array(arr)
    intersections = tensor @ tensor.T
    sums = tensor.sum(axis=1, keepdims=True) + tensor.sum(axis=1, keepdims=True).T
    sums = mlx_core.where(sums == 0, mlx_core.ones_like(sums), sums)
    result = mlx_core.clip(intersections / sums, 0.0, 1.0).astype(np.float32)
    return result


def batch_tanimoto_similarity(fps: list[Any], n_bits: int = 2048, device: str | None = None) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Compute pairwise Tanimoto similarity matrix for a batch of fingerprints.

    Selects the best available backend:
      - MPS (Apple GPU) if torch with MPS is available
      - MLX (Apple GPU) if mlx is available
      - Numpy (CPU) otherwise

    Uses vectorized numpy matrix operations for all batch sizes.
    RDKit's BulkTanimotoSimilarity is not used because it does not
    accept a list-of-fingerprints as the first argument in RDKit 2026+.

    Args:
        fps: List of RDKit fingerprint objects.
        n_bits: Number of bits in each fingerprint.
        device: Optional device override ("mps", "mlx", "cpu").
            If None, auto-detects via ``get_device()``.

    Returns:
        2D numpy array of shape (n, n) with Tanimoto similarities.
    """
    if device is None:
        device = get_device()
    if device == "mps":
        return _tanimoto_batch_mps(fps, n_bits=n_bits)
    if device == "mlx":
        return _tanimoto_batch_mlx(fps, n_bits=n_bits)
    return _tanimoto_batch_numpy(fps, n_bits=n_bits)


def _apply_physical_bounds(
    raw: dict[str, float],
) -> tuple[dict[str, float], list[str]]:
    """Clamp each property to its physical bounds, collecting warnings.

    Returns the clamped dict and a list of warning messages for any value
    that fell outside its bounds.
    """
    clamped: dict[str, float] = {}
    warnings_list: list[str] = []
    for key, value in raw.items():
        lo, hi = _PHYSICAL_BOUNDS.get(key, (float("-inf"), float("inf")))
        if value < lo:
            clamped[key] = lo
            warnings_list.append(f"{key} below physical minimum (clamped to {lo})")
        elif value > hi:
            clamped[key] = hi
            warnings_list.append(f"{key} above physical maximum (clamped to {hi})")
        else:
            clamped[key] = value
    return clamped, warnings_list


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

        clamped_values, sanity_warning = _apply_physical_bounds({
            "dielectric_proxy": dielectric,
            "viscosity_proxy": viscosity,
            "homo_eV": homo,
            "lumo_eV": lumo,
        })

        cp = get_conformal_predictor()
        intervals = {
            "homo": cp.predict_interval("homo", clamped_values["homo_eV"]),
            "lumo": cp.predict_interval("lumo", clamped_values["lumo_eV"]),
            "dielectric": cp.predict_interval(
                "dielectric", clamped_values["dielectric_proxy"]
            ),
            "viscosity": cp.predict_interval(
                "viscosity", clamped_values["viscosity_proxy"]
            ),
        }
        conformal_confidence = cp.confidence_discount(intervals)

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
            "conformal_intervals": {
                prop: [round(lo, 4), round(hi, 4)]
                for prop, (lo, hi) in intervals.items()
            },
            "conformal_confidence": round(conformal_confidence, 4),
        }

        self._cache[smiles] = result
        return result

    def predict_batch_properties(self, contexts: list[MoleculeContext]) -> dict[str, np.ndarray[Any, np.dtype[np.float32]]]:
        """Batch-predict properties for a list of molecules.

        Vectorizes fragment counting, fingerprint generation,
        quantum orbital prediction, and property prediction to
        leverage MPS/MLX acceleration where available.

        Args:
            contexts: List of pre-parsed MoleculeContext objects.

        Returns:
            Dict mapping property names to 1D numpy arrays.
            Keys: dielectric_proxy, viscosity_proxy, li_solvation_proxy,
                  conductivity_proxy, tpsa, mw, fp (2048-dim fingerprint matrix),
                  homo_eV, lumo_eV, gap_eV.
        """
        n = len(contexts)
        if n == 0:
            return {
                "dielectric_proxy": np.array([], dtype=np.float32),
                "viscosity_proxy": np.array([], dtype=np.float32),
                "li_solvation_proxy": np.array([], dtype=np.float32),
                "conductivity_proxy": np.array([], dtype=np.float32),
                "homo_eV": np.array([], dtype=np.float32),
                "lumo_eV": np.array([], dtype=np.float32),
                "gap_eV": np.array([], dtype=np.float32),
            }

        # Batch fingerprint matrix
        fps = [ctx.get_ecfp4() for ctx in contexts]
        fp_matrix = _fp_batch_to_numpy(fps, n_bits=2048)

        # Batch fragment counting
        counts = _count_fragments_batch(contexts)

        # Batch GC property prediction
        tpsa_values = np.array([ctx.tpsa for ctx in contexts], dtype=np.float32)
        mw_values = np.array([ctx.mw for ctx in contexts], dtype=np.float32)
        n_rotatable = np.array([ctx.rotatable_bonds for ctx in contexts], dtype=np.int32)

        # Compute branch points and stereocenters in batch
        n_branch = np.array([_count_branch_points(ctx.mol) for ctx in contexts], dtype=np.int32)
        n_stereo = np.array([_count_stereocenters(ctx.mol) for ctx in contexts], dtype=np.int32)

        dielectric = predict_dielectric_proxy_batch(counts, tpsa_values)
        viscosity = predict_viscosity_proxy_batch(counts, mw_values, n_rotatable, n_branch, n_stereo)
        li_solvation = predict_li_solvation_proxy_batch(counts, mw_values)
        conductivity = predict_ionic_conductivity_proxy_batch(dielectric, viscosity, li_solvation)

        # Vectorized TOM batch evaluation for HOMO/LUMO
        homo_array, lumo_array = predict_tom_orbitals_batch([ctx.mol for ctx in contexts])
        gap_array = lumo_array - homo_array

        return {
            "dielectric_proxy": dielectric,
            "viscosity_proxy": viscosity,
            "li_solvation_proxy": li_solvation,
            "conductivity_proxy": conductivity,
            "tpsa": tpsa_values,
            "mw": mw_values,
            "fp": fp_matrix,
            "homo_eV": homo_array,
            "lumo_eV": lumo_array,
            "gap_eV": gap_array,
        }

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

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
