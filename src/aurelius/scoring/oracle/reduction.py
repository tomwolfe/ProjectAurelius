"""Reduction-stability oracle: ΔSCF electron affinity (ADR-2026-08-10).

Physical justification
----------------------
The reduction axis was previously ranked by a frontier LUMO energy — first from
multi-source DFT labels, then from internally consistent xTB labels. Both
failed on unseen molecules (Spearman ρ = 0.06 → 0.10). Cleaning the labels did
not help because the problem is the *observable*, not the provenance.

Koopmans' theorem is strong for the occupied space and weak for the virtual
space. Ionisation removes an electron from a bound, localised lone pair, which
is why the Lone-Pair Model reaches ρ = 0.94 against 88 NIST ionisation
energies. Electron *attachment* to a saturated carbonate or ether does not
populate a bound orbital at all: the lowest virtual orbital of such a molecule
is a discretised continuum function whose energy tracks the basis set rather
than the chemistry. Ranking by it is close to ranking by basis diffuseness.

Measured against 40 directly determined gas-phase electron affinities
(permutation control: |ρ| 95th percentile = 0.31):

    TOM LUMO, negated               ρ = +0.34   (at the noise floor)
    structural ridge, class-disjoint ρ = +0.69
    ΔSCF EA (xTB)                   ρ = +0.91

This module therefore computes the ΔSCF vertical electron affinity

    EA = E(neutral, N e⁻) − E(anion, N+1 e⁻)          [both at neutral geometry]

with two GFN2-xTB single points. This is a genuine energy difference between
two variationally optimised electronic states, so it includes the orbital
relaxation that Koopmans' theorem discards — which is precisely the term that
dominates for weakly binding closed-shell solvents.

Sign convention: **higher EA = the molecule accepts an electron more readily =
less reduction-stable**. For an electrolyte solvent, low EA is desirable in the
bulk; a deliberately high-EA additive (FEC, VC) is what forms the SEI.

Cost and caching
----------------
Two single points, ~30 ms per molecule for solvent-sized systems. Results are
cached by canonical SMILES in a JSON file that survives process restarts, in
the same style as ``xtb_single_point.py``.

Fallback
--------
When xTB is absent (CI, most laptops) a ridge model over interpretable
electron-accepting structural features is used. It is fitted on the same clean
experimental set and reports a distinct ``method`` so the two paths can never
be silently confused. The old TOM LUMO is *not* used as a fallback: at ρ = 0.34
against a 0.31 permutation bar it carries essentially no ranking information.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem

from aurelius.scoring.oracle.quantum import _find_xtb_binary, _generate_xyz, has_xtb
from aurelius.types import MoleculeContext

logger = logging.getLogger(__name__)

HARTREE_EV = 27.211386245988

_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "data", "experimental_electron_affinity.json",
)

DEFAULT_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "..", "reduction_cache.json",
)

_TOTAL_ENERGY_RE = re.compile(r"TOTAL ENERGY\s+([-+]?\d+\.\d+)\s+Eh")

_XTB_TIMEOUT = 120

# OLS affine map from raw xTB ΔSCF EA onto the experimental EA scale.
# Refreshed by ``scripts/calibrate_reduction.py``; an affine map cannot change
# Spearman rho, so this sets units only and is not a ranking fit.
_EA_CALIBRATION: tuple[float, float] = (0.6590, -2.9176)

# OLS affine map from raw ALPB-corrected ΔSCF EA onto the solution-phase
# CV onset scale (ADR-2026-08-11). Solution-phase EA is shifted by solvation
# stabilization of the anion; this corrects the ALPB-corrected xTB values
# onto the experimental solution reference (1M LiPF6 EC:DMC vs Li/Li+).
_EA_SOLUTION_CALIBRATION: tuple[float, float] = (0.8842, -1.4210)

# Raw-xTB ΔSCF span covered by the experimental gas-phase calibration set.
# Predictions outside it are flagged: they are bounded extrapolations, not
# measurements.
_EA_CALIBRATED_SPAN_RAW: tuple[float, float] = (2.92, 8.82)

# Raw ALPB-corrected ΔSCF EA span covered by the solution-phase calibration
# set (10 entries: EC, PC, DMC, DEC, FEC, VC, DME, THF, ACN, sulfolane).
_EA_SOLUTION_CALIBRATED_SPAN_RAW: tuple[float, float] = (1.65, 5.85)

# xTB --alpb accepts named solvents. Map a predicted dielectric constant to
# the nearest ALPB solvent so the ΔSCF EA is evaluated in a medium matching
# the molecule's own polarity. For battery electrolytes the relevant range
# is ε ≈ 2–40 (linear carbonates to acetonitrile); outside it we fall back
# to acetonitrile as the standard reference solvent for CV calibration.
_ALPB_SOLVENT_BY_DIELECTRIC: list[tuple[str, float]] = [
    ("hexane", 1.9),
    ("toluene", 2.4),
    ("thf", 7.6),
    ("dcm", 9.1),
    ("acetone", 20.7),
    ("ethanol", 24.3),
    ("acetonitrile", 37.5),
    ("dmso", 46.7),
    ("water", 80.0),
]


def solvent_from_dielectric(epsilon: float) -> str:
    """Map a predicted dielectric constant to the nearest xTB ALPB solvent.

    Uses absolute distance on the ε scale. Clamps to the nearest named
    solvent rather than extrapolating beyond the ends.
    """
    best, best_dist = "acetonitrile", float("inf")
    for name, eps in _ALPB_SOLVENT_BY_DIELECTRIC:
        d = abs(eps - epsilon)
        if d < best_dist:
            best, best_dist = name, d
    return best


@dataclass
class ReductionResult:
    """Reduction-stability estimate for one molecule."""

    ea_eV: float | None
    method: str  # "xtb_dscf" | "structural_ridge" | "unavailable"
    confidence: float
    in_calibrated_span: bool
    ea_raw: float | None = None
    cpu_seconds: float = 0.0

    solvent: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ea_eV": None if self.ea_eV is None else round(self.ea_eV, 4),
            "method": self.method,
            "confidence": round(self.confidence, 4),
            "in_calibrated_span": self.in_calibrated_span,
            "ea_raw": None if self.ea_raw is None else round(self.ea_raw, 4),
            "cpu_seconds": round(self.cpu_seconds, 4),
            "solvent": self.solvent,
        }


def load_experimental_ea() -> list[dict[str, Any]]:
    """Load the clean experimental electron-affinity set.

    Returns all entries (both gas-phase and solution-phase). Use
    ``load_experimental_ea_gas()`` or ``load_experimental_ea_solution()``
    for phase-specific subsets.
    """
    try:
        with open(_DATA_PATH) as fh:
            doc = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("Experimental EA set unavailable at %s", _DATA_PATH)
        return []
    entries = doc.get("entries", []) if isinstance(doc, dict) else doc
    return [e for e in entries if Chem.MolFromSmiles(e.get("smiles", "")) is not None]


# ---------------------------------------------------------------------------
# Structural fallback features
# ---------------------------------------------------------------------------
# Each feature is an electron-accepting structural motif with a known direction
# of effect on electron affinity. Keeping them nameable is the point: the
# fallback has to stay auditable when xTB is missing.

_ACCEPTOR_SMARTS: list[tuple[str, str]] = [
    ("nitro", "[N+](=O)[O-]"),
    ("nitrile", "[NX1]#[CX2]"),
    ("carbonyl", "[CX3]=[OX1]"),
    ("quinoid", "O=C1C=CC(=O)C=C1"),
    ("sulfonyl", "[SX4](=O)(=O)"),
    ("conj_c_c", "C=C"),
    ("fluorine", "[F]"),
    ("chlorine", "[Cl]"),
    ("azo", "[NX2]=[NX2]"),
    ("anhydride", "[CX3](=O)[OX2][CX3](=O)"),
]

_COMPILED_ACCEPTORS: list[tuple[str, Chem.Mol]] = [
    (name, pat) for name, smarts in _ACCEPTOR_SMARTS
    if (pat := Chem.MolFromSmarts(smarts)) is not None
]


def structural_features(mol: Chem.Mol) -> np.ndarray:
    """Interpretable electron-accepting descriptors for the xTB-free fallback.

    Counts are square-rooted because acceptor effects on EA saturate: the
    second nitro group adds less than the first, the same sub-linear behaviour
    the LPM inductive term models on the ionisation side.
    """
    from rdkit.Chem import Descriptors, rdMolDescriptors

    feats: list[float] = []
    for _name, pat in _COMPILED_ACCEPTORS:
        feats.append(float(np.sqrt(len(mol.GetSubstructMatches(pat)))))

    n_arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    n_heavy = max(mol.GetNumHeavyAtoms(), 1)
    n_pi = sum(1 for b in mol.GetBonds() if b.GetIsAromatic() or b.GetBondTypeAsDouble() > 1.0)

    feats.extend([
        float(n_arom_rings),
        float(np.sqrt(n_pi)),
        float(n_pi) / n_heavy,
        float(Descriptors.MolWt(mol)) / 100.0,
        float(Descriptors.TPSA(mol)) / 50.0,
    ])
    return np.asarray(feats, dtype=np.float64)


class _StructuralEAModel:
    """Ridge regression on interpretable acceptor features.

    This exists so the reduction axis degrades to something with a *positive*
    correlation when xTB is unavailable, rather than to the anti-correlated
    TOM LUMO it previously fell back to.
    """

    def __init__(self, entries: list[dict[str, Any]] | None = None) -> None:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        self._entries = entries if entries is not None else load_experimental_ea()
        self._ok = False
        if len(self._entries) < 10:
            return

        X, y = [], []
        for e in self._entries:
            mol = Chem.MolFromSmiles(e["smiles"])
            if mol is None:
                continue
            X.append(structural_features(mol))
            y.append(float(e["ea_eV"]))

        self._X = np.vstack(X)
        self._y = np.asarray(y)
        self._scaler = StandardScaler().fit(self._X)
        self._model = Ridge(alpha=1.0).fit(self._scaler.transform(self._X), self._y)
        self._residual_std = float(
            np.std(self._y - self._model.predict(self._scaler.transform(self._X)))
        ) or 1.0
        self._ok = True

    @property
    def available(self) -> bool:
        return self._ok

    def predict(self, mol: Chem.Mol) -> tuple[float, float]:
        """Return (ea_eV, confidence)."""
        x = self._scaler.transform(structural_features(mol).reshape(1, -1))
        ea = float(self._model.predict(x)[0])
        # Confidence falls off with distance from the fitted feature cloud.
        d = float(np.linalg.norm(x))
        conf = float(np.clip(1.0 / (1.0 + 0.15 * d), 0.05, 0.7))
        return ea, conf

    def loo_metrics(self) -> dict[str, float]:
        """Leave-one-out ρ and MAE — the honest fallback accuracy."""
        from scipy.stats import spearmanr
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        preds = np.zeros_like(self._y)
        for i in range(len(self._y)):
            mask = np.ones(len(self._y), dtype=bool)
            mask[i] = False
            sc = StandardScaler().fit(self._X[mask])
            m = Ridge(alpha=1.0).fit(sc.transform(self._X[mask]), self._y[mask])
            preds[i] = m.predict(sc.transform(self._X[i].reshape(1, -1)))[0]
        return {
            "spearman_rho": float(spearmanr(preds, self._y).statistic),
            "mae_eV": float(np.mean(np.abs(preds - self._y))),
            "n": int(len(self._y)),
        }


# ---------------------------------------------------------------------------
# xTB ΔSCF path
# ---------------------------------------------------------------------------

def _run_xtb_energy(
    xyz: str, charge: int, uhf: int, solvent: str | None = None, threads: int = 1
) -> float | None:
    """Single-point total energy in Hartree, or None on any failure."""
    xtb_bin = _find_xtb_binary()
    if xtb_bin is None:
        return None

    workdir = tempfile.mkdtemp(prefix="aurelius_ea_")
    xyz_path = os.path.join(workdir, "input.xyz")
    try:
        with open(xyz_path, "w") as fh:
            fh.write(xyz)

        cmd = [xtb_bin, "--gfn", "2", "--sp", xyz_path,
               "--chrg", str(charge), "--uhf", str(uhf), "-P", str(threads)]
        if solvent:
            cmd += ["--alpb", solvent]

        env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
        proc = subprocess.run(
            cmd, cwd=workdir, capture_output=True, text=True,
            timeout=_XTB_TIMEOUT, env=env,
        )
        matches = _TOTAL_ENERGY_RE.findall(proc.stdout)
        return float(matches[-1]) if matches else None
    except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
        logger.debug("xTB EA single point failed: %s", exc)
        return None
    finally:
        with contextlib.suppress(OSError):
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)


def compute_dscf_ea(mol: Chem.Mol, solvent: str | None = None, threads: int = 1) -> float | None:
    """Raw ΔSCF vertical electron affinity in eV (uncalibrated).

    Both single points use the neutral ETKDG geometry, so this is the vertical
    quantity — the appropriate one for a fast electron-transfer step, and the
    only one xTB evaluates reliably for anions whose relaxed geometry is
    unbound.
    """
    xyz = _generate_xyz(mol)
    e_neutral = _run_xtb_energy(xyz, 0, 0, solvent=solvent, threads=threads)
    if e_neutral is None:
        return None
    e_anion = _run_xtb_energy(xyz, -1, 1, solvent=solvent, threads=threads)
    if e_anion is None:
        return None
    return (e_neutral - e_anion) * HARTREE_EV


def compute_dscf_ea_batch(
    mols: list[Chem.Mol],
    solvent: str | None = None,
    max_workers: int | None = None,
) -> list[float | None]:
    """ΔSCF EA for many molecules, parallelised across cores.

    Each molecule needs two xTB single points, and for solvent-sized systems
    each one is dominated by process start-up rather than by the SCF itself
    (~60 ms wall, of which the actual calculation is ~15 ms). Threading a
    single xTB call therefore buys nothing — measured identical wall time at
    ``-P 1`` and ``-P 8`` — but the calls are completely independent, so
    running them across a process pool scales nearly linearly until the core
    count is saturated.

    Falls back to serial evaluation when the pool cannot be created, and
    preserves input order.
    """
    if not mols:
        return []

    from concurrent.futures import ThreadPoolExecutor

    if max_workers is None:
        max_workers = min(len(mols), max(1, (os.cpu_count() or 4) - 1))

    def _one(mol: Chem.Mol) -> float | None:
        try:
            return compute_dscf_ea(mol, solvent=solvent, threads=1)
        except Exception:  # pragma: no cover - defensive
            return None

    try:
        # Threads, not processes: every worker blocks in subprocess.run, so the
        # GIL is released and there is no pickling cost for RDKit molecules.
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(_one, mols))
    except Exception:  # pragma: no cover - defensive
        return [_one(m) for m in mols]


def calibrate_ea(raw_ea: float) -> float:
    """Map a raw xTB ΔSCF EA onto the experimental EA scale."""
    a, b = _EA_CALIBRATION
    return a * raw_ea + b


def calibrate_ea_solution(raw_ea: float, solvent: str | None = None) -> float:
    """Map a raw ALPB-corrected xTB ΔSCF EA onto the solution-phase scale.

    Uses the solution-phase calibration affine map fitted on 10-20 CV onset
    measurements (ADR-2026-08-11). The ALPB solvation model already shifts
    the raw ΔSCF energy; this calibration corrects any residual offset between
    ALPB-xTB and experimental CV onset.

    Args:
        raw_ea: Raw ALPB-corrected xTB ΔSCF EA in eV.
        solvent: ALPB solvent name (unused, calibration is global but
            available for future solvent-specific refinement).

    Returns:
        Calibrated solution-phase EA in eV (vs vacuum, i.e. on the same scale
        as gas-phase EA + 1.39 eV reference conversion).
    """
    a, b = _EA_SOLUTION_CALIBRATION
    return a * raw_ea + b


def load_experimental_ea_solution() -> list[dict[str, Any]]:
    """Load solution-phase experimental EA entries from the calibration set."""
    return [e for e in load_experimental_ea() if e.get("phase") == "solution"]


def load_experimental_ea_gas() -> list[dict[str, Any]]:
    """Load gas-phase experimental EA entries (excludes solution-phase)."""
    return [e for e in load_experimental_ea() if e.get("phase") == "gas"]


class ReductionOracle:
    """Reduction-stability oracle with xTB ΔSCF primary and ridge fallback.

    Usage::

        oracle = ReductionOracle()
        result = oracle.evaluate(ctx)   # dict: ea_eV, method, confidence, ...

    ``ea_eV`` is on the experimental gas-phase electron-affinity scale, where
    **higher means more easily reduced** (worse bulk reduction stability).
    """

    def __init__(
        self,
        cache_path: str = DEFAULT_CACHE_PATH,
        solvent: str | None = None,
        threads: int = 1,
    ) -> None:
        self._cache_path = cache_path
        self._solvent = solvent
        self._threads = threads
        self._cache = self._load_cache()
        self._fallback: _StructuralEAModel | None = None
        self._hits = 0
        self._misses = 0
        self._dirty = False

    @classmethod
    def with_auto_solvent(
        cls,
        mol: Chem.Mol,
        cache_path: str = DEFAULT_CACHE_PATH,
        threads: int = 1,
    ) -> ReductionOracle:
        """Create an oracle with solvent auto-selected from predicted dielectric.

        Uses the Kirkwood-Fröhlich dielectric proxy to pick the nearest
        ALPB solvent, falling back to "acetonitrile" when the proxy is
        unavailable.
        """
        solvent = "acetonitrile"
        try:
            from aurelius.scoring.oracle.gc import predict_dielectric_proxy
            from aurelius.types import MoleculeContext

            ctx = mol if isinstance(mol, MoleculeContext) else MoleculeContext.from_smiles(
                Chem.MolToSmiles(mol)
            )
            if ctx is not None:
                eps = predict_dielectric_proxy(ctx)
                solvent = solvent_from_dielectric(eps)
        except Exception:
            pass
        return cls(cache_path=cache_path, solvent=solvent, threads=threads)

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        try:
            with open(self._cache_path) as fh:
                loaded = json.load(fh)
            return loaded if isinstance(loaded, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _get_fallback(self) -> _StructuralEAModel:
        if self._fallback is None:
            self._fallback = _StructuralEAModel()
        return self._fallback

    @property
    def method(self) -> str:
        return "xtb_dscf" if has_xtb() else "structural_ridge"

    def evaluate(self, ctx: MoleculeContext | Chem.Mol) -> dict[str, Any]:
        """Return the reduction-stability record for a molecule."""
        mol = ctx.mol if isinstance(ctx, MoleculeContext) else ctx
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        key = f"{smiles}|{self._solvent}"

        cached = self._cache.get(key)
        if cached is not None and cached.get("method") == self.method:
            self._hits += 1
            return dict(cached)

        self._misses += 1
        result = self._compute(mol).to_dict()
        self._cache[key] = result
        self._dirty = True
        return result

    def _compute(self, mol: Chem.Mol) -> ReductionResult:
        import time

        start = time.perf_counter()
        if has_xtb():
                 raw = compute_dscf_ea(mol, solvent=self._solvent, threads=self._threads)
                 if raw is not None:
                    lo, hi = _EA_CALIBRATED_SPAN_RAW
                    in_span = lo <= raw <= hi
                    if self._solvent is not None:
                        calibrated = self._calibrate_solution_phase(raw, self._solvent)
                        sol_lo, sol_hi = _EA_SOLUTION_CALIBRATED_SPAN_RAW
                        in_span = sol_lo <= raw <= sol_hi
                        confidence = 0.90 if in_span else 0.45
                    else:
                        calibrated = calibrate_ea(raw)
                        confidence = 0.85 if in_span else 0.45
                    return ReductionResult(
                        ea_eV=calibrated,
                        method="xtb_dscf",
                        # Extrapolation beyond the calibrated span is reported but
                        # discounted rather than hidden.
                        confidence=confidence,
                        in_calibrated_span=in_span,
                        ea_raw=raw,
                        cpu_seconds=time.perf_counter() - start,
                        solvent=self._solvent,
                    )

        fallback = self._get_fallback()
        if fallback.available:
            ea, conf = fallback.predict(mol)
            return ReductionResult(
                ea_eV=ea, method="structural_ridge", confidence=conf,
                in_calibrated_span=True, ea_raw=None,
                cpu_seconds=time.perf_counter() - start,
                solvent=self._solvent,
            )

        return ReductionResult(
            ea_eV=None, method="unavailable", confidence=0.0,
            in_calibrated_span=False,
            cpu_seconds=time.perf_counter() - start,
            solvent=self._solvent,
        )

    def _calibrate_solution_phase(self, raw_ea: float, solvent: str | None) -> float:
        """Apply solution-phase calibration to a raw ALPB-corrected ΔSCF EA.

        ADR-2026-08-11: ALPB solvation is already implemented in ``_compute``.
        This method adds the *calibration* step that maps ALPB-xTB energies
        onto the experimental solution-phase CV scale, using the affine map
        fitted on the solution-phase calibration set added in this ADR.

        When ``solvent`` is None (gas-phase mode), the gas-phase calibration
        is applied instead, preserving backward compatibility.
        """
        if solvent is None:
            return calibrate_ea(raw_ea)
        return calibrate_ea_solution(raw_ea, solvent)

    def flush(self) -> None:
        """Persist the cache to disk if anything changed."""
        if not self._dirty:
            return
        try:
            directory = os.path.dirname(self._cache_path) or "."
            os.makedirs(directory, exist_ok=True)
            with open(self._cache_path, "w") as fh:
                json.dump(self._cache, fh, indent=2)
            self._dirty = False
        except OSError as exc:
            logger.warning("Could not write reduction cache: %s", exc)

    def report(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "method": self.method,
            "cache_entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total else 0.0,
        }


_DEFAULT: ReductionOracle | None = None


def get_reduction_oracle() -> ReductionOracle:
    """Process-wide singleton reduction oracle (lazy init)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ReductionOracle()
    return _DEFAULT


def predict_reduction_stability(mol: Chem.Mol) -> dict[str, Any]:
    """Convenience wrapper: reduction-stability record for one molecule."""
    return get_reduction_oracle().evaluate(mol)
