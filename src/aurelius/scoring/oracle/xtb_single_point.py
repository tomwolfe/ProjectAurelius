"""Tier-2.5 xTB single-point oracle with SMILES-keyed caching.

ADR-2026-08-08-02: Project Aurelius v11 escalated to xTB only for molecules
flagged ``tom_low`` by the conjugation-based domain of applicability — a
handful per generation, chosen by a heuristic about pi-system length rather
than by how good the molecule is. In practice this meant the most
promising candidates (high-scoring, well inside the DoA) were scored by the
closed-form model and never touched real quantum mechanics.

This module implements the mandatory mid-tier gate described in the project
plan: every Tier-1 survivor gets a fast **single-point** xTB GFN2-xTB
calculation (not a geometry optimisation — an SP is ~10x cheaper and is the
whole point of a bridge tier) before any ORCA work-up. ORCA is reserved for
the top tier only and is handled by :class:`DFTValidator`.

Three design constraints drove the implementation:

1. **Aggressive caching, keyed by canonical SMILES.** The evolutionary
   search revisits near-identical molecules across generations, and the same
   SMILES recurs in nearly every benchmark rerun. The cache is a plain JSON
   file so it survives process restarts and cross-checks against the
   existing ``dft_cache.json`` convention. A cache hit turns a 0.3-2 s QM
   call into a 50 µs disk read.

2. **Graceful degradation.** xTB is not installed in the test environment, on
   the CI runner, or on many laptops. When it is absent the gate must return a
   well-formed record marked ``"xtb_unavailable"`` so the rest of the
   pipeline is unaffected, never raise.

3. **No new dependencies.** xTB is invoked via subprocess exactly as
   :mod:`aurelius.scoring.oracle.quantum` already does; the cache is JSON.

The cache layout mirrors ``dft_cache.json`` so the two can be merged by a
single ``json.load`` if desired, but they are kept separate deliberately:
geometry optimisation and single-point have different convergence
semantics and different error models, and conflating them would silently
corrupt the grounding score.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

from rdkit import Chem

from aurelius.scoring.oracle.quantum import _find_xtb_binary, _generate_xyz, has_xtb
from aurelius.types import MoleculeContext

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "..", "xtb_cache.json",
)

_XTB_SINGLE_POINT_TIMEOUT = 120


@dataclass
class XTBResult:
    """Cached result of an xTB single-point calculation.

    ``homo_eV``/``lumo_eV`` are None when xTB was unavailable or failed, so
    callers must explicitly check rather than assuming the field is present.
    """

    homo_eV: float | None
    lumo_eV: float | None
    dipole_D: float
    xtb_method: str
    convergence: str
    cpu_seconds: float
    source: str = "computed"  # "computed" | "xtb_unavailable" | "parse_failure" | "timeout"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_key(mol: Chem.Mol) -> str:
    """Cache key: canonical SMILES, invariant to input ordering."""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _load_cache(cache_path: str) -> dict[str, dict[str, Any]]:
    try:
        with open(cache_path) as fh:
            loaded = json.load(fh)
        return loaded if isinstance(loaded, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache_path: str, cache: dict[str, dict[str, Any]]) -> None:
    try:
        directory = os.path.dirname(cache_path) or "."
        os.makedirs(directory, exist_ok=True)
        with open(cache_path, "w") as fh:
            json.dump(cache, fh, indent=2, default=str)
    except OSError as exc:
        logger.warning("Could not write xTB cache to %s: %s", cache_path, exc)


class XTBSinglePointOracle:
    """Single-point xTB GFN2-xTB evaluator with SMILES-keyed caching.

    Usage:

        oracle = XTBSinglePointOracle()
        result = oracle.evaluate(ctx)        # dict with homo/lumo/method/source
        oracle.report()                       # cache hit/miss statistics

    The oracle is cheap to construct and holds its cache in memory; call
    ``load()`` to seed it from the on-disk file, and the cache is persisted
    lazily on the first write.
    """

    def __init__(self, cache_path: str = DEFAULT_CACHE_PATH) -> None:
        self._cache_path = cache_path
        self._cache: dict[str, dict[str, Any]] = {}
        self._hits = 0
        self._misses = 0
        self._load()

    def _load(self) -> None:
        self._cache = _load_cache(self._cache_path)
        logger.debug(
            "xTB SP cache loaded: %d entries from %s", len(self._cache), self._cache_path
        )

    @property
    def n_cached(self) -> int:
        return len(self._cache)

    def evaluate(self, ctx: MoleculeContext | str) -> dict[str, Any]:
        """Evaluate a molecule, returning cached or freshly-computed results.

        Returns a dict (not :class:`XTBResult`) so callers never have to
        import the dataclass — the schema mirrors the existing
        ``DFTValidator`` output so it slots into the same reporting path.

        Keys: ``homo_eV``, ``lumo_eV``, ``dipole_D``, ``xtb_method``,
        ``convergence``, ``cpu_seconds``, ``source``.
        """
        mol = MoleculeContext.from_smiles(ctx).mol if isinstance(ctx, str) else ctx.mol
        key = _canonical_key(mol)

        if key in self._cache:
            self._hits += 1
            return dict(self._cache[key])

        self._misses += 1
        result = self._compute(mol)
        self._cache[key] = result.to_dict()
        return result.to_dict()

    def _compute(self, mol: Chem.Mol) -> XTBResult:
        """Run a cold xTB single-point or record its absence."""
        if not has_xtb():
            return XTBResult(
                homo_eV=None, lumo_eV=None, dipole_D=0.0,
                xtb_method="GFN2-xTB", convergence="not_run",
                cpu_seconds=0.0, source="xtb_unavailable",
            )

        xyz = _generate_xyz(mol)
        xtb_bin = _find_xtb_binary()
        start = time.perf_counter()
        try:
            result = self._run_sp(xtb_bin, xyz, mol)
        except Exception as exc:
            logger.debug("xTB single-point failed for %s: %s", _canonical_key(mol), exc)
            result = XTBResult(
                homo_eV=None, lumo_eV=None, dipole_D=0.0,
                xtb_method="GFN2-xTB", convergence="failure",
                cpu_seconds=round(time.perf_counter() - start, 4),
                source="parse_failure",
            )
        result.cpu_seconds = round(time.perf_counter() - start, 4)
        self._persist()
        return result

    @staticmethod
    def _run_sp(xtb_bin: str, xyz: str, mol: Chem.Mol) -> XTBResult:
        import subprocess
        import tempfile

        workdir = tempfile.mkdtemp(prefix="aurelius_xtbsp_")
        xyz_path = os.path.join(workdir, "input.xyz")
        with open(xyz_path, "w") as fh:
            fh.write(xyz)

        from aurelius.scoring.oracle.quantum import _parse_xtb_output

        proc = subprocess.run(
            [xtb_bin, "--gfn", "2", "--sp", xyz_path],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=_XTB_SINGLE_POINT_TIMEOUT,
        )
        parsed = _parse_xtb_output(proc.stdout + proc.stderr)
        if parsed is None:
            return XTBResult(
                homo_eV=None, lumo_eV=None, dipole_D=0.0,
                xtb_method="GFN2-xTB", convergence="parse_failure",
                cpu_seconds=0.0, source="parse_failure",
            )
        return XTBResult(
            homo_eV=parsed["homo_eV"],
            lumo_eV=parsed["lumo_eV"],
            dipole_D=parsed.get("dipole_D", 0.0),
            xtb_method="GFN2-xTB",
            convergence="converged",
            cpu_seconds=0.0,
            source="computed",
        )

    def _persist(self) -> None:
        _save_cache(self._cache_path, self._cache)

    def report(self) -> dict[str, Any]:
        total = self._hits + self._misses
        hit_rate = (self._hits / total) if total else 0.0
        return {
            "cache_path": self._cache_path,
            "cache_entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
        }

    def flush(self) -> None:
        """Persist the in-memory cache to disk immediately."""
        self._persist()


@contextlib.contextmanager
def temporary_oracle(temp_dir: str) -> XTBSinglePointOracle:
    """Oracle backed by an on-disk cache in a throwaway directory.

    Convenience for tests that should not touch the user's real
    ``xtb_cache.json``.
    """
    os.makedirs(temp_dir, exist_ok=True)
    oracle = XTBSinglePointOracle(cache_path=os.path.join(temp_dir, "xtb_cache.json"))
    try:
        yield oracle
    finally:
        oracle.flush()
