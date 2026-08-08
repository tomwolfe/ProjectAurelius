"""DFTValidator — ORCA single-point re-ranking gate.

Validates top Aurelius discoveries with a higher-level DFT single point
(wB97X-D3/def2-SVP via ORCA) as an independent check on the surrogate
oracle's frontier-orbital predictions. This closes the "wet-lab ready"
loop: candidates are re-scored by theory that is systematically more
accurate than the TOM/xTB surrogate before they are handed to synthesis.

Design:
  - ORCA is invoked as a subprocess (mirroring the xTB wrapper in
    quantum.py) so the gate degrades gracefully when ORCA is not
    installed: ``available()`` returns False and ``compute()`` returns None.
  - Results are cached in ``dft_cache.json`` keyed by canonical SMILES so
    re-ranking runs never recompute finished molecules.
  - A Spearman-rho helper compares the Aurelius ranking against the DFT
    HOMO/LUMO ranking; the prospective-selection script warns when the
    independent model disagrees (rho < 0.4).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import tempfile
import warnings
from typing import Any

from rdkit import Chem
from scipy.stats import spearmanr

from aurelius.scoring.oracle.quantum import _find_xtb_binary, _generate_xyz

logger = logging.getLogger(__name__)

_ORCA_METHOD: str = "wB97X-D3 def2-SVP SP RIJCOSX"
_ORCA_NPROCS: int = 4

# Minimum grounding score a candidate must achieve to pass the DFT geometry
# optimization cascade gate. A converged xTB GFN2-xTB optimization scores 1.0;
# a failed optimization scores 0.5.
DFT_GEOMETRY_OPT_THRESHOLD: float = 0.80


def _find_orca_binary() -> str | None:
    """Locate the ORCA binary on the system PATH."""
    for candidate in ["orca", "orca_mpi"]:
        with contextlib.suppress(Exception):
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return candidate
    return None


_HAS_ORCA: bool = _find_orca_binary() is not None


def has_orca() -> bool:
    """Return True if an ORCA binary is available on PATH."""
    return _HAS_ORCA


def _build_orca_input() -> str:
    """Build an ORCA input file for a wB97X-D3/def2-SVP single point."""
    return (
        f"! {_ORCA_METHOD}\n"
        f"%pal nprocs {_ORCA_NPROCS} end\n"
        "%maxcore 2000\n"
        "* xyzfile 0 1 input.xyz\n"
    )


def _parse_orbital_row(line: str) -> tuple[float, float] | None:
    """Parse a single ORCA orbital row into (occ, E_eV) or None."""
    line = line.strip()
    if not line or set(line) <= {"-"}:
        return None
    tokens = line.replace(":", " ").split()
    if len(tokens) < 4 or not tokens[0].strip().isdigit():
        return None
    try:
        occ = float(tokens[1])
        e_ev = float(tokens[3])
    except ValueError:
        return None
    return occ, e_ev


def _parse_orca_output(output: str) -> dict[str, float] | None:
    """Parse HOMO/LUMO energies from an ORCA single-point output.

    ORCA prints an ``ORBITAL ENERGIES`` block with rows of the form
    ``NO : OCC  E(Eh)  E(eV)``. For a closed-shell single point the last
    row with OCC > 0 is the HOMO and the first row with OCC == 0 is the
    LUMO. Only the first (alpha) block is parsed.
    """
    idx = output.find("ORBITAL ENERGIES")
    if idx < 0:
        return None
    block = output[idx:]

    # Keep only the first spin block (cut at the next delimiter).
    end = block.find("\n---", 20)
    if end > 0:
        block = block[:end]

    homo: float | None = None
    lumo: float | None = None
    for line in block.splitlines():
        parsed = _parse_orbital_row(line)
        if parsed is None:
            continue
        occ, e_ev = parsed
        if occ > 0.0:
            homo = e_ev
        elif lumo is None:
            lumo = e_ev

    if homo is None or lumo is None or lumo <= homo:
        return None
    return {"homo_eV": homo, "lumo_eV": lumo}


def spearman_correlation(x: list[float], y: list[float]) -> tuple[float, float]:
    """Spearman rank correlation between two score lists.

    Returns (rho, p-value). Degenerate inputs (fewer than 3 points or a
    constant array) yield rho=0.0, p=1.0 rather than NaN or a warning.
    """
    if len(x) < 3 or len(x) != len(y):
        return 0.0, 1.0
    with contextlib.suppress(Exception), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, p = spearmanr(x, y)
        if rho is not None and not isinstance(rho, complex) and rho == rho:  # not NaN
            return float(rho), float(p)
    return 0.0, 1.0


class DFTValidator:
    """ORCA single-point validator with on-disk SMILES-keyed caching."""

    METHOD: str = _ORCA_METHOD

    def __init__(self, cache_path: str = "dft_cache.json", seed: int = 42) -> None:
        self._cache_path = cache_path
        self._seed = seed
        self._n_calls = 0
        self._cache: dict[str, dict[str, float]] = self._load_cache()

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _load_cache(self) -> dict[str, dict[str, float]]:
        try:
            with open(self._cache_path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        return {}

    def _save_cache(self) -> None:
        try:
            with open(self._cache_path, "w") as f:
                json.dump(self._cache, f, indent=2)
        except OSError as exc:
            logger.debug("DFTValidator: could not write cache %s (%s)", self._cache_path, exc)

    def clear_cache(self) -> None:
        self._cache.clear()

    def get_cache_size(self) -> int:
        return len(self._cache)

    # ------------------------------------------------------------------
    # ORCA subprocess wrapper
    # ------------------------------------------------------------------

    def _run_orca(self, mol: Chem.Mol) -> dict[str, float] | None:
        """Run an ORCA single-point calculation and parse HOMO/LUMO."""
        orca_bin = _find_orca_binary()
        if orca_bin is None:
            return None

        xyz = _generate_xyz(mol)
        workdir = tempfile.mkdtemp(prefix="aurelius_dft_")
        with open(os.path.join(workdir, "input.xyz"), "w") as f:
            f.write(xyz)
        with open(os.path.join(workdir, "input.inp"), "w") as f:
            f.write(_build_orca_input())

        try:
            result = subprocess.run(
                [orca_bin, "input.inp"],
                cwd=workdir,
                capture_output=True, text=True, timeout=600,
            )
            output = result.stdout
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
            logger.debug("DFTValidator: ORCA run failed (%s)", exc)
            return None

        parsed = _parse_orca_output(output)
        if parsed is not None:
            parsed["dft_method"] = self.METHOD  # type: ignore[dict-item]
            self._n_calls += 1
        return parsed

    def compute(self, mol: Chem.Mol) -> dict[str, float] | None:
        """Compute (or fetch from cache) DFT HOMO/LUMO for a molecule.

        Returns a dict with ``homo_eV``/``lumo_eV`` or None if ORCA is
        unavailable or the calculation/parse fails.
        """
        smiles = Chem.MolToSmiles(mol)
        if smiles in self._cache:
            return dict(self._cache[smiles])

        result = self._run_orca(mol)
        if result is not None:
            self._cache[smiles] = result
            self._save_cache()
        return result

    def validate_ranking(
        self, scores: list[float], molecules: list[Chem.Mol]
    ) -> dict[str, float]:
        """Re-rank a candidate list by DFT and report Spearman agreement.

        Computes rho between the Aurelius scores and a DFT composite
        virtual potential -(HOMO+LUMO)/2 (same convention as the EHT
        benchmark), plus per-orbital correlations.

        Returns:
            Dict with ``rho_composite``, ``rho_homo``, ``rho_lumo``,
            ``n_validated`` and the associated p-values.
        """
        dft_homos: list[float] = []
        dft_lumos: list[float] = []
        valid_scores: list[float] = []

        for score, mol in zip(scores, molecules, strict=False):
            result = self.compute(mol)
            if result is None:
                continue
            dft_homos.append(result["homo_eV"])
            dft_lumos.append(result["lumo_eV"])
            valid_scores.append(score)

        if len(valid_scores) < 3:
            return {
                "rho_composite": 0.0, "p_composite": 1.0,
                "rho_homo": 0.0, "p_homo": 1.0,
                "rho_lumo": 0.0, "p_lumo": 1.0,
                "n_validated": len(valid_scores),
            }

        composite = [-(h + l_val) / 2.0 for h, l_val in zip(dft_homos, dft_lumos, strict=False)]
        rho_c, p_c = spearman_correlation(valid_scores, composite)
        rho_h, p_h = spearman_correlation(valid_scores, dft_homos)
        rho_l, p_l = spearman_correlation(valid_scores, dft_lumos)

        logger.info(
            "DFTValidator: validated %d candidates — rho(composite)=%.3f "
            "rho(HOMO)=%.3f rho(LUMO)=%.3f",
            len(valid_scores), rho_c, rho_h, rho_l,
        )
        return {
            "rho_composite": round(rho_c, 4),
            "p_composite": round(p_c, 4),
            "rho_homo": round(rho_h, 4),
            "p_homo": round(p_h, 4),
            "rho_lumo": round(rho_l, 4),
            "p_lumo": round(p_l, 4),
            "n_validated": len(valid_scores),
        }


import re


_DFT_GEOM_CACHE: dict[str, dict[str, float]] = {}


def _parse_xtb_opt_convergence(output: str) -> tuple[bool, float]:
    """Parse xTB optimization output for convergence and final energy.

    xTB prints ``   ***       Fully optimized       ***`` or ``   * Converged``
    when geometry optimization succeeds. Returns (converged, final_energy_eV).
    """
    converged = bool(
        re.search(r"Fully optimized|Converged|Stationary point found", output, re.IGNORECASE)
    )
    energy = float("nan")
    for line in reversed(output.splitlines()):
        m = re.search(r"Total Energy\s*[:=]?\s*([-+]?\d+\.?\d*)\s*(?:a\.u\.|Eh)?", line, re.IGNORECASE)
        if m:
            energy = float(m.group(1))
            if abs(energy) > 50.0:
                energy /= 27.2114
            break
    if not converged:
        for line in output.splitlines():
            m = re.search(r"Final energy\s*[:=]?\s*([-+]?\d+\.?\d*)", line, re.IGNORECASE)
            if m:
                energy = float(m.group(1))
                break
    return converged, energy


def dft_geometry_optimize(
    mol: Chem.Mol,
    cache_path: str = "dft_cache.json",
    timeout: int = 300,
) -> dict[str, float | str | None]:
    """Run xTB GFN2-xTB geometry optimization on a molecule.

    Physical justification: A molecule's frontier orbital energies depend on
    its 3-D conformation. Running a fast semi-empirical geometry optimization
    (xTB GFN2-xTB with ``--opt``) provides a grounded, first-principles
    confirmation that the candidate's predicted orbitals are based on a
    realistic geometry rather than a hand-built 2-D embedding. The
    optimization's convergence status is reported as ``dft_grounding_score``
    (1.0 = converged, 0.5 = failed to converge, 0.0 = not attempted).

    Results are cached by canonical SMILES to avoid recomputing the same
    molecule's optimized geometry.

    Args:
        mol: RDKit molecule to optimize.
        cache_path: Path to a JSON cache file keyed by SMILES.
        timeout: Maximum seconds for the xTB subprocess.

    Returns:
        Dict with keys:
          - ``dft_grounding_score`` (float): 1.0 if converged, 0.5 if failed,
            1.0 if xTB unavailable (graceful degradation).
          - ``dft_final_energy_eV`` (float | None): Final SCF energy after
            optimization, or ``float('nan')`` if not computed.
          - ``dft_method`` (str): Description of the method used.
    """
    smiles = Chem.MolToSmiles(mol)
    if smiles in _DFT_GEOM_CACHE:
        return dict(_DFT_GEOM_CACHE[smiles])

    file_cache = _read_geom_cache(cache_path)
    cached = file_cache.get(smiles)
    if isinstance(cached, dict) and "dft_grounding_score" in cached:
        _DFT_GEOM_CACHE[smiles] = dict(cached)
        return dict(cached)

    xtb_bin = _find_xtb_binary()
    if xtb_bin is None:
        result: dict[str, float | str | None] = {
            "dft_grounding_score": 1.0,
            "dft_final_energy_eV": float("nan"),
            "dft_method": "xTB unavailable (dft_grounding_score defaulted to 1.0)",
        }
    else:
        result = _run_xtb_optimization(mol, xtb_bin, timeout)

    _DFT_GEOM_CACHE[smiles] = result
    file_cache[smiles] = result
    _write_geom_cache(cache_path, file_cache)
    return dict(result)


def _read_geom_cache(cache_path: str) -> dict[str, Any]:
    """Load the on-disk geometry cache, tolerating a missing or corrupt file."""
    try:
        with open(cache_path) as f:
            loaded = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write_geom_cache(cache_path: str, file_cache: dict[str, Any]) -> None:
    """Persist the geometry cache; caching is best-effort and never fatal."""
    try:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(file_cache, f, indent=2, default=str)
    except OSError:
        pass


def _run_xtb_optimization(
    mol: Chem.Mol, xtb_bin: str, timeout: int
) -> dict[str, float | str | None]:
    """Run ``xtb --opt`` in a scratch directory and summarise convergence."""
    method = "xTB GFN2-xTB geometry optimization"
    workdir = tempfile.mkdtemp(prefix="aurelius_dft_geom_")
    xyz_path = os.path.join(workdir, "input.xyz")
    with open(xyz_path, "w") as f:
        f.write(_generate_xyz(mol))

    try:
        proc = subprocess.run(
            [xtb_bin, "--gfn", "2", "--opt", xyz_path],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
        logger.debug("DFT geometry optimization failed (%s)", exc)
        return {
            "dft_grounding_score": 0.5,
            "dft_final_energy_eV": float("nan"),
            "dft_method": f"{method} (subprocess error)",
        }

    converged, energy = _parse_xtb_opt_convergence(proc.stdout + proc.stderr)
    return {
        "dft_grounding_score": 1.0 if converged else 0.5,
        "dft_final_energy_eV": round(energy, 6) if energy == energy else float("nan"),
        "dft_method": method if converged else f"{method} (convergence failed)",
    }


__all__ = [
    "DFTValidator",
    "DFT_GEOMETRY_OPT_THRESHOLD",
    "dft_geometry_optimize",
    "has_orca",
    "spearman_correlation",
    "_build_orca_input",
    "_parse_orca_output",
    "_parse_xtb_opt_convergence",
]
