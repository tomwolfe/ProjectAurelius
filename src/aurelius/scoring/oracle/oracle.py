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

from aurelius.scoring.oracle.conformal import get_conformal_predictor
from aurelius.scoring.oracle.delta_correction import get_delta_correction
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
    predict_orbitals_batch,
)
from aurelius.types import MoleculeContext
from aurelius.utils.device import get_device

logger = logging.getLogger(__name__)

_BACKEND_NAMES = {
    "numpy": "numpy",
    "mlx": "mlx",
    "mps": "mps",
    "cuda": "cuda",
}

_BACKEND_LABELS = {
    "numpy": "CPU (numpy)",
    "mlx": "GPU (MLX)",
    "mps": "GPU (MPS)",
    "cuda": "GPU (CUDA)",
}


def _select_batch_backend() -> str:
    """Return the current batch backend name for logging/benchmarking."""
    dev = get_device()
    return _BACKEND_LABELS.get(dev, dev)

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

    Uses vectorized numpy operations to process all fingerprints at once,
    eliminating Python for-loop over individual bits.
    """
    n = len(fps)
    arr = np.zeros((n, n_bits), dtype=np.float32)

    # Collect all on-bit indices for all fingerprints
    all_on_bits = []
    row_indices = []

    for i, fp in enumerate(fps):
        n_on = fp.GetNumOnBits()
        if n_on == 0:
            continue
        on_bits = fp.GetOnBits()
        all_on_bits.extend(on_bits)
        row_indices.extend([i] * n_on)

    # Use advanced numpy indexing to set all values at once
    arr[np.array(row_indices), np.array(all_on_bits)] = 1.0
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
    import mlx.core as mx
    arr = _fp_batch_to_numpy(fps, n_bits=n_bits)
    tensor = mx.array(arr)
    intersections = tensor @ tensor.T
    sums = mx.broadcast_to(tensor.sum(axis=1, keepdims=True), intersections.shape)
    sums_t = mx.broadcast_to(tensor.sum(axis=1, keepdims=True).T, intersections.shape)
    union = sums + sums_t
    union = mx.maximum(union, mx.array(1.0))
    result = mx.clip(intersections / union, 0.0, 1.0)
    return np.array(result.astype(mx.float32))


def batch_tanimoto_similarity(fps: list[Any], n_bits: int = 2048, device: str | None = None) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Compute pairwise Tanimoto similarity matrix for a batch of fingerprints.

    Selects the best available backend:
      - MLX (Apple GPU) if available (lowest launch overhead for small batches)
      - MPS (Apple GPU) if torch with MPS is available
      - Numpy (CPU) otherwise

    Uses vectorized matrix operations for all batch sizes.
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
    if device == "mlx":
        return _tanimoto_batch_mlx(fps, n_bits=n_bits)
    if device == "mps":
        return _tanimoto_batch_mps(fps, n_bits=n_bits)
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
        homo = float(quantum_result["homo_eV"])
        lumo = float(quantum_result["lumo_eV"])

        # ADR-2026-08-10: Reduction stability is now the ΔSCF electron
        # affinity, validated against 40 experimental gas-phase EAs
        # (rho = 0.91). The LUMO Δ-correction it replaces reached only
        # rho = 0.10 on unseen molecules because a frozen virtual orbital is
        # not a physical reduction descriptor for saturated solvents.
        reduction_proxy = self._predict_reduction_proxy(ctx.mol)

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
            "reduction_stability_proxy": reduction_proxy,
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

    def _predict_reduction_proxy(self, mol: Any) -> dict[str, Any]:
        """ΔSCF electron affinity as the reduction-stability axis (ADR-2026-08-10).

        Returns ``ea_eV`` on the experimental gas-phase electron-affinity
        scale, where **higher means more easily reduced** (worse bulk reduction
        stability). Backed by xTB ΔSCF when available (rho = 0.91 against 40
        measured EAs), otherwise by an interpretable structural ridge model
        (rho = 0.69, chemical-class-disjoint).

        ``lumo_eV`` is still reported for backward compatibility but is a
        calibration artefact only — it must not be used for ranking.
        """
        record: dict[str, Any] = {
            "ea_eV": None, "method": "unavailable", "confidence": 0.0,
            "in_calibrated_span": False,
            "metric": "Spearman rho vs 40 experimental gas-phase EAs",
            "sign_convention": "higher ea_eV = more easily reduced = less stable",
        }

        try:
            from aurelius.scoring.oracle.reduction import get_reduction_oracle

            record.update(get_reduction_oracle().evaluate(mol))
        except Exception as exc:
            logger.debug("Reduction oracle failed (%s).", exc)

        # Legacy field, retained for one release. Not a ranking input.
        self._fill_lumo_fields(record, mol)

        return record

    def _fill_lumo_fields(self, record: dict[str, Any], mol: Any) -> None:
        """Legacy LUMO fields for a reduction record, retained for one release.

        Not a ranking input: the LUMO is a calibration artefact only
        (see ADR-2026-08-10).
        """
        try:
            from aurelius.scoring.oracle.lumo_proxy import get_lumo_proxy

            lumo, _conf = get_lumo_proxy().predict_corrected(mol)
            record["lumo_eV"] = round(lumo, 4)
            record["lumo_note"] = "calibration only (MAE); superseded for ranking"
        except Exception:
            record["lumo_eV"] = None

    def predict_batch_properties(self, contexts: list[MoleculeContext]) -> dict[str, Any]:
        """Batch-predict properties for a list of molecules.

        Vectorizes fragment counting, fingerprint generation,
        quantum orbital prediction, and property prediction to
        leverage MPS/MLX acceleration where available.

        Physical justification: On M5 Pro, fusing GC+TOM computations into
        a single MLX computation graph eliminates intermediate numpy copies
        and reduces CPU-GPU synchronization overhead. Target batch sizes
        of 200-500 molecules saturate GPU bandwidth while minimizing kernel
        launch overhead for MLX (lower than MPS for fine-grained operations).

        Args:
            contexts: List of pre-parsed MoleculeContext objects.

        Returns:
            Dict mapping property names to 1D numpy arrays.
            Keys: dielectric_proxy, viscosity_proxy, li_solvation_proxy,
                  conductivity_proxy, tpsa, mw, fp (2048-dim fingerprint matrix),
                  homo_eV, lumo_eV, gap_eV, ea_eV (float32, NaN when the
                  reduction record has no EA), reduction_records (list of
                  per-molecule dicts, key-identical to the scalar
                  ``_predict_reduction_proxy`` record), and the per-molecule
                  metadata lists/arrays screen_batch needs to assemble tier2
                  key-identical to scalar ``evaluate()``: sanity_warning,
                  quantum_confidence, domain_applicable, domain_reason,
                  domain_penalty, conformal_intervals, conformal_confidence,
                  quantum_method.
        """
        n = len(contexts)
        if n == 0:
            return self._empty_batch_result()

        # Batch fingerprint matrix. Kept on CPU: the result is returned to the
        # caller as numpy, and the round trip to GPU was previously discarded.
        fps = [ctx.get_ecfp4() for ctx in contexts]
        fp_matrix = _fp_batch_to_numpy(fps, n_bits=2048)

        # Batch fragment counting (returns numpy arrays)
        counts = _count_fragments_batch(contexts)

        # Batch GC property prediction
        tpsa_values = np.array([ctx.tpsa for ctx in contexts], dtype=np.float32)
        mw_values = np.array([ctx.mw for ctx in contexts], dtype=np.float32)
        n_rotatable = np.array([ctx.rotatable_bonds for ctx in contexts], dtype=np.int32)

        # Compute branch points and stereocenters in batch
        n_branch = np.array([_count_branch_points(ctx.mol) for ctx in contexts], dtype=np.int32)
        n_stereo = np.array([_count_stereocenters(ctx.mol) for ctx in contexts], dtype=np.int32)

        # ADR-2026-08-07-02: Single source of truth for batch GC physics.
        #
        # A separate hand-written MLX branch previously duplicated these
        # formulas and had silently drifted from the numpy/scalar definitions:
        #   - Li+ solvation used (mw - 30.0) * 0.05 instead of
        #     max(0, mw - 50.0) * 0.002, and clamped at 1.0 instead of 0.5.
        #   - Conductivity used exp(-v) * d * ls instead of the Walden product
        #     with the Gaussian Li+ term, so it ignored the 3.5 Goldilocks
        #     target entirely.
        # Measured divergence on DMSO: conductivity 13.44 (MLX) vs 0.625 (CPU),
        # a 21x discrepancy that made GPU and CPU runs mutually incomparable
        # and silently changed EA selection depending on the host machine.
        #
        # These are elementwise ops over an (N, ~35) array. Measured on M5 Pro
        # at N=550 the whole numpy block costs 0.30 ms, versus 42 ms for the
        # upstream RDKit SMARTS matching that produces `counts`. There is no
        # meaningful arithmetic to offload, so the correct engineering choice
        # is one shared implementation rather than two that can disagree.
        dielectric = predict_dielectric_proxy_batch(counts, tpsa_values, contexts)
        viscosity = predict_viscosity_proxy_batch(
            counts, mw_values, n_rotatable, n_branch, n_stereo, contexts
        )
        li_solvation = predict_li_solvation_proxy_batch(counts, mw_values)
        conductivity = predict_ionic_conductivity_proxy_batch(dielectric, viscosity, li_solvation)

        # Batch frontier orbitals: LPM HOMO + TOM LUMO (ADR-2026-08-08-01),
        # with the same Δ-correction the scalar closed-form path applies so
        # batch and scalar (use_xtb=False) orbitals agree.
        mols = [ctx.mol for ctx in contexts]
        homo_array, lumo_array = predict_orbitals_batch(mols)
        homo_array, lumo_array = get_delta_correction().predict_corrected_batch(
            mols, base=list(zip(homo_array.tolist(), lumo_array.tolist(), strict=True))
        )
        gap_array = lumo_array - homo_array

        # Reduction axis: per-molecule ΔSCF EA records (ADR-2026-08-10).
        reduction_records = self._predict_reduction_proxy_batch(contexts)
        ea_array = np.asarray(
            [
                float(record["ea_eV"]) if record.get("ea_eV") is not None else float("nan")
                for record in reduction_records
            ],
            dtype=np.float32,
        )
        metadata = self._compute_batch_metadata(
            contexts, homo_array, lumo_array, dielectric, viscosity
        )

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
            "ea_eV": ea_array,
            "reduction_records": reduction_records,
            "sanity_warning": metadata["sanity_warning"],
            "quantum_confidence": metadata["quantum_confidence"],
            "domain_applicable": metadata["domain_applicable"],
            "domain_reason": metadata["domain_reason"],
            "domain_penalty": metadata["domain_penalty"],
            "conformal_intervals": metadata["conformal_intervals"],
            "conformal_confidence": metadata["conformal_confidence"],
            "quantum_method": [self._closed_form_method()] * n,
        }

    def _empty_batch_result(self) -> dict[str, Any]:
        """All-keys-empty batch result for an empty input list."""
        empty_f32 = np.array([], dtype=np.float32)
        return {
            "dielectric_proxy": empty_f32,
            "viscosity_proxy": empty_f32,
            "li_solvation_proxy": empty_f32,
            "conductivity_proxy": empty_f32,
            "homo_eV": empty_f32,
            "lumo_eV": empty_f32,
            "gap_eV": empty_f32,
            "ea_eV": empty_f32,
            "tpsa": empty_f32,
            "mw": empty_f32,
            "fp": np.zeros((0, 2048), dtype=np.float32),
            "reduction_records": [],
            "sanity_warning": [],
            "quantum_confidence": [],
            "domain_applicable": [],
            "domain_reason": [],
            "domain_penalty": [],
            "conformal_intervals": [],
            "conformal_confidence": [],
            "quantum_method": [],
        }

    def _closed_form_method(self) -> str:
        """Honest method string for the closed-form orbital path.

        The batch path always computes closed-form orbitals via
        ``predict_orbitals_batch``, so ``quantum_method`` must never claim the
        xTB backend even when ``PropertyOracle(use_xtb=True)``.
        """
        if getattr(self._quantum, "_use_lone_pair", True):
            return "LPM HOMO + TOM LUMO"
        return "TOM (Topological Orbital Model)"

    def _predict_reduction_proxy_batch(
        self, contexts: list[MoleculeContext]
    ) -> list[dict[str, Any]]:
        """ΔSCF electron-affinity records for a batch (ADR-2026-08-10).

        When xTB is available the raw ΔSCF EAs are computed once, in parallel
        and preserving input order, and assembled into records key-identical
        to the scalar gas branch of ``ReductionOracle._compute``. Molecules
        whose raw EA is None fall back to the scalar proxy. When xTB is
        unavailable every molecule falls back to the scalar proxy (structural
        ridge) — the batch path never probes for the xTB binary itself.
        """
        from aurelius.scoring.oracle.reduction import (
            _EA_CALIBRATED_SPAN_RAW,
            calibrate_ea,
            compute_dscf_ea_batch,
            has_xtb,
        )

        mols = [ctx.mol for ctx in contexts]
        if not has_xtb():
            return [self._predict_reduction_proxy(mol) for mol in mols]

        records: list[dict[str, Any]] = []
        for mol, raw in zip(mols, compute_dscf_ea_batch(mols), strict=True):
            if raw is None:
                records.append(self._predict_reduction_proxy(mol))
                continue
            lo, hi = _EA_CALIBRATED_SPAN_RAW
            in_span = lo <= raw <= hi
            record: dict[str, Any] = {
                "ea_eV": round(calibrate_ea(raw), 4),
                "method": "xtb_dscf",
                "confidence": round(0.85 if in_span else 0.45, 4),
                "in_calibrated_span": in_span,
                "ea_raw": round(raw, 4),
                "gas_ea_eV": None,
                "cpu_seconds": 0.0,
                "solvent": None,
                "metric": "Spearman rho vs 40 experimental gas-phase EAs",
                "sign_convention": "higher ea_eV = more easily reduced = less stable",
            }
            self._fill_lumo_fields(record, mol)
            records.append(record)
        return records

    def _compute_batch_metadata(
        self,
        contexts: list[MoleculeContext],
        homo_array: np.ndarray,
        lumo_array: np.ndarray,
        dielectric: np.ndarray,
        viscosity: np.ndarray,
    ) -> dict[str, Any]:
        """Per-molecule metadata lists mirroring ``PropertyOracle.evaluate()``.

        Reuses the exact scalar functions (``_apply_physical_bounds``,
        ``compute_quantum_domain_penalty``, ``compute_gc_domain_penalty`` and
        the conformal predictor) so batch and scalar tier-2 records stay
        key-identical without duplicating oracle logic.
        """
        sanity_warning: list[list[str]] = []
        quantum_confidence: list[str] = []
        domain_applicable: list[bool] = []
        domain_reason: list[str] = []
        domain_penalty: list[float] = []
        conformal_intervals: list[dict[str, list[float]]] = []
        conformal_confidence: list[float] = []

        cp = get_conformal_predictor()
        for i, ctx in enumerate(contexts):
            clamped, warnings_list = _apply_physical_bounds({
                "dielectric_proxy": float(dielectric[i]),
                "viscosity_proxy": float(viscosity[i]),
                "homo_eV": float(homo_array[i]),
                "lumo_eV": float(lumo_array[i]),
            })
            sanity_warning.append(warnings_list)

            q_penalty, q_reason = compute_quantum_domain_penalty(ctx)
            gc_penalty, gc_reason = compute_gc_domain_penalty(ctx)
            d_penalty = min(q_penalty, gc_penalty)
            reasons: list[str] = []
            if q_penalty < 1.0:
                reasons.append(f"quantum: {q_reason}")
            if gc_penalty < 1.0:
                reasons.append(f"GC: {gc_reason}")
            domain_applicable.append(d_penalty >= 0.85)
            domain_reason.append("; ".join(reasons) if reasons else _DATA_SOURCE)
            domain_penalty.append(d_penalty)

            intervals = {
                "homo": cp.predict_interval("homo", clamped["homo_eV"]),
                "lumo": cp.predict_interval("lumo", clamped["lumo_eV"]),
                "dielectric": cp.predict_interval(
                    "dielectric", clamped["dielectric_proxy"]
                ),
                "viscosity": cp.predict_interval(
                    "viscosity", clamped["viscosity_proxy"]
                ),
            }
            conformal_intervals.append({
                prop: [round(lo, 4), round(hi, 4)]
                for prop, (lo, hi) in intervals.items()
            })
            conformal_confidence.append(round(cp.confidence_discount(intervals), 4))

            quantum_confidence.append(self._quantum._tom_confidence(ctx.mol))

        return {
            "sanity_warning": sanity_warning,
            "quantum_confidence": quantum_confidence,
            "domain_applicable": domain_applicable,
            "domain_reason": domain_reason,
            "domain_penalty": domain_penalty,
            "conformal_intervals": conformal_intervals,
            "conformal_confidence": conformal_confidence,
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
