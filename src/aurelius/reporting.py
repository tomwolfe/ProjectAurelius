"""Wet-lab candidate reporting for Project Aurelius.

Promotes ``scripts/select_prospective_candidates.py`` into a first-class
``aurelius report`` pipeline entry point so that wet-lab handoff is a core
feature rather than an afterthought.

Public API:
    ReportingEngine()          -- build an engine with an initialized pipeline
    engine.generate_report(...) -- run (or reuse) a discovery loop and emit
                                   markdown + CSV handoff artifacts
    aurelius report ...        -- CLI wrapper, see ``__main__.py``
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from rdkit import Chem
from rdkit.Chem import AllChem

from aurelius.agent.loop import AgentConfig, run_screening
from aurelius.agent.mutation.retrosynthetic import compute_synthesis_feasibility, has_plausible_route
from aurelius.constants import HYDROLYTICALLY_UNSTABLE_PATTERNS
from aurelius.pipeline import AureliusPipeline
from aurelius.scoring.oracle.dft_validator import dft_geometry_optimize
from aurelius.types import MoleculeContext

try:
    from rdkit.DataStructs import TanimotoSimilarity
except Exception:  # pragma: no cover - RDKit always present when pipeline runs
    TanimotoSimilarity = None  # type: ignore[assignment]

# Cascade filtering thresholds for wet-lab decision readiness.
# The DFT geometry-optimization gate (dft_grounding_score >= 0.80) is mandatory
# unless the caller passes ``skip_dft=True`` to ``generate_report()``.
# Note: ``synthesis_depth`` is retained for backward compatibility in result
# serialisation, but the primary selection objective is now the continuous
# ``synthesizability_complexity`` score in [0, 1].
CANDIDATE_CASCADE = [
    ("is_viable", True, "Candidate must be viable (is_viable=True)"),
    ("combined_grounding_score", 0.75, "Combined grounding score must be >= 0.75"),
    ("synthesizability_complexity", 0.6, "Synthesizability complexity must be >= 0.6"),
    ("domain_penalty", 0.95, "Domain penalty must be >= 0.95"),
    ("novelty_to_seed", 0.3, "Novelty to seed must be >= 0.3"),
]

# Default DFT geometry-optimization threshold and cache path.
DFT_GEOMETRY_OPT_THRESHOLD = 0.80
DFT_CACHE_PATH = "dft_cache.json"

# Defaults for a de novo discovery run.
DEFAULT_N_GENERATIONS = 50
DEFAULT_BATCH_SIZE = 50
DEFAULT_TOP_N = 20
SYNTHESIS_FEASIBILITY_THRESHOLD = 0.6
NOVELTY_THRESHOLD = 0.3


def _load_known_electrolytes() -> list[str]:
    """Load known commercial electrolyte SMILES from the data directory."""
    data_path = Path(__file__).parent / "data" / "known_electrolytes.json"
    try:
        with open(data_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _ecfp4(smi: str) -> Any | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def _tanimoto(a: Any, b: Any) -> float:
    if TanimotoSimilarity is None:
        return 0.0
    return float(TanimotoSimilarity(a, b))


def _find_nearest_known(
    candidate_smi: str,
    known: list[str],
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """Nearest known commercial electrolytes by Tanimoto similarity."""
    if not known or TanimotoSimilarity is None:
        return []
    cand_fp = _ecfp4(candidate_smi)
    if cand_fp is None:
        return []

    sims: list[tuple[str, float]] = []
    for smi in known:
        known_fp = _ecfp4(smi)
        if known_fp is None:
            continue
        sims.append((smi, _tanimoto(cand_fp, known_fp)))

    sims.sort(key=lambda x: -x[1])
    return [{"smiles": s, "similarity": round(v, 4)} for s, v in sims[:top_n]]


def _generate_synthesis_hints(mol: Chem.Mol) -> str:
    """Brief synthesis hints based on functional-group presence."""
    hints = []
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3](=[OX1])[OX2]")):
        hints.append("carbonate/ester")
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[OX2][CX4]")):
        hints.append("ether linkage")
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[SX4](=O)(=O)")):
        hints.append("sulfone")
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[CX4][F]")):
        hints.append("fluorinated")
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3]#[NX2]")):
        hints.append("nitrile")
    if not hints:
        hints.append("novel scaffold")
    return "; ".join(hints)


def _check_al_corrosion_risk(mol: Chem.Mol) -> bool:
    """High-LUMO fluorinated motifs that risk Al current-collector corrosion."""
    n_f = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
    if n_f < 3:
        return False
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:
            neighbors = [n.GetAtomicNum() for n in atom.GetNeighbors()]
            if 9 in neighbors and 8 in neighbors:
                return True
    return False


def _check_hydrolytic_instability(mol: Chem.Mol) -> bool:
    """Flag motifs known to hydrolyse under battery-relevant conditions."""
    for pattern, _name, _severity in HYDROLYTICALLY_UNSTABLE_PATTERNS:
        if pattern is not None and mol.HasSubstructMatch(pattern):
            return True
    return False


def _confidence_interval(
    homo: float | None,
    lumo: float | None,
    conformal_conf: float,
) -> tuple[float, float]:
    """Approximate [lo, hi] confidence interval for the HOMO-LUMO gap."""
    if homo is None or lumo is None:
        return (0.0, 0.0)
    gap = lumo - homo
    uncertainty = (1.0 - conformal_conf) * 2.0
    return (max(0.0, gap - uncertainty), gap + uncertainty)


def _diversity_penalty(selected_smis: list[str], cand_smi: str) -> float:
    """Max Tanimoto to already-selected molecules (0 = diverse, 1 = duplicate)."""
    if not selected_smis or TanimotoSimilarity is None:
        return 0.0
    cand_fp = _ecfp4(cand_smi)
    if cand_fp is None:
        return 0.0
    best = 0.0
    for s in selected_smis:
        ref = _ecfp4(s)
        if ref is not None:
            best = max(best, _tanimoto(cand_fp, ref))
    return best


CANDIDATE_FIELDS = [
    "smiles", "total_score", "adjusted_score", "synthesis_feasibility",
    "conformal_confidence", "novelty_to_seed", "homo_eV", "lumo_eV",
    "gap_eV", "confidence_interval_low", "confidence_interval_high",
    "synthesis_hints", "risk_flags", "sa_score", "dielectric_proxy",
    "viscosity_proxy", "diversity_penalty", "combined_grounding_score",
    "synthesizability_complexity", "domain_penalty", "is_viable",
    "synthesis_depth", "dft_grounding_score", "dft_final_energy_eV", "dft_method",
    "synthesizability_route", "synthesizability_route_desc",
]


class ReportingEngine:
    """Generate standardized wet-lab candidate reports from discovery runs.

    Usage::

        engine = ReportingEngine()
        summary, report = engine.generate_report(
            output_dir="reports",
            n_generations=50,
            top_n=20,
        )
    """

    def __init__(self, pipeline: AureliusPipeline | None = None) -> None:
        self._pipeline = pipeline if pipeline is not None else _make_pipeline()

    # -- candidate assembly -------------------------------------------------

    def _build_candidate(
        self,
        result: object,
        known: list[str],
        skip_dft: bool = False,
        dft_cache_path: str = DFT_CACHE_PATH,
    ) -> dict[str, Any] | None:
        """Convert a single discovery-loop result into a candidate dict."""
        smiles = getattr(result, "smiles", None)
        if not smiles:
            return None
        if getattr(result, "total_score", 0.0) < 65.0:
            return None
        if (sc := getattr(result, "synthesizability_complexity", None)) is not None and sc < 0.6:
            return None

        ctx = MoleculeContext.from_smiles(smiles)
        if ctx is None:
            return None
        mol = ctx.mol

        synthesis_feas = compute_synthesis_feasibility(mol)
        if synthesis_feas < SYNTHESIS_FEASIBILITY_THRESHOLD:
            return None

        sub = getattr(result, "sub_scores", {}) or {}
        conformal_conf = sub.get("confidence", 1.0)
        novelty = getattr(result, "novelty_to_seed", None) or 0.0
        if novelty < NOVELTY_THRESHOLD:
            return None

        homo = getattr(result, "homo_eV", None)
        lumo = getattr(result, "lumo_eV", None)
        ci_low, ci_high = _confidence_interval(homo, lumo, conformal_conf)

        # DFT geometry-optimization gate (mandatory unless --skip-dft).
        # Runs xTB GFN2-xTB --opt to confirm a realistic 3-D geometry.
        if skip_dft:
            dft_grounding_score = 1.0
            dft_final_energy_eV = float("nan")
            dft_method = "skipped (--skip-dft)"
        else:
            dft_result: dict[str, Any] = dft_geometry_optimize(mol, cache_path=dft_cache_path)
            dft_grounding_score = float(dft_result.get("dft_grounding_score", 1.0))
            dft_final_energy_eV = dft_result.get("dft_final_energy_eV", float("nan"))
            dft_method = dft_result.get("dft_method", "unknown")

        risk_flags: list[str] = []
        if _check_al_corrosion_risk(mol):
            risk_flags.append("Al_corrosion_risk")
        if _check_hydrolytic_instability(mol):
            risk_flags.append("hydrolytic_instability")

        # Prefer explicit dataclass fields for these scores, falling back to
        # sub_scores so the engine works with both live and cached results.
        combined_gs = getattr(result, "combined_grounding_score", None)
        if combined_gs is None:
            combined_gs = sub.get("grounding", 0.0)
        domain_penalty = getattr(result, "domain_penalty", None)
        if domain_penalty is None:
            domain_penalty = sub.get("domain", 1.0)

        # Check retrosynthetic route (hard gate for synthesizability)
        has_route, route_desc = has_plausible_route(mol)

        return {
            "smiles": smiles,
            "total_score": float(getattr(result, "total_score", 0.0)),
            "synthesis_feasibility": round(synthesis_feas, 4),
            "synthesizability_complexity": round(float(synthesis_feas), 4),
            "conformal_confidence": round(float(conformal_conf), 4),
            "novelty_to_seed": round(float(novelty), 4),
            "homo_eV": homo,
            "lumo_eV": lumo,
            "gap_eV": round(lumo - homo, 4) if homo is not None and lumo is not None else None,
            "confidence_interval_low": round(ci_low, 4),
            "confidence_interval_high": round(ci_high, 4),
            "synthesis_hints": _generate_synthesis_hints(mol),
            "risk_flags": "; ".join(risk_flags) if risk_flags else "none",
            "sa_score": getattr(result, "sa_score", None),
            "dielectric_proxy": getattr(result, "dielectric_proxy", None),
            "viscosity_proxy": getattr(result, "viscosity_proxy", None),
            "combined_grounding_score": round(float(combined_gs), 4),
            "domain_penalty": round(float(domain_penalty), 4),
            "is_viable": bool(getattr(result, "is_viable", False)),
            "synthesis_depth": getattr(result, "synthesis_depth", None),
            "adjusted_score": round(float(getattr(result, "total_score", 0.0)), 4),
            "dft_grounding_score": round(float(dft_grounding_score), 4),
            "dft_final_energy_eV": dft_final_energy_eV,
            "dft_method": dft_method,
            "synthesizability_route": has_route,
            "synthesizability_route_desc": route_desc,
            "nearest_known_electrolytes": _find_nearest_known(smiles, known, top_n=3),
        }

    # -- discovery ----------------------------------------------------------

    def run_discovery(
        self,
        n_generations: int = DEFAULT_N_GENERATIONS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> dict[str, Any]:
        """Execute a discovery loop and return its raw results."""
        cfg = AgentConfig(
            max_generations=n_generations,
            batch_size=batch_size,
            use_nsga2=True,
            active_learning_threshold=0.7,
        )
        return run_screening(cfg)

    # -- reporting ---------------------------------------------------------

    def _apply_cascade(
        self,
        candidates: list[dict[str, Any]],
        skip_dft: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Apply the wet-lab cascade filter.

        When ``skip_dft`` is False (the default), an additional DFT geometry-
        optimization stage is appended requiring ``dft_grounding_score >= 0.80``.
        
        ADR-2026-08-16: Added synthesizability_route stage as a hard gate.
        Candidates without a plausible retrosynthetic route to commercial
        precursors are rejected.
        """
        cascade_stages = list(CANDIDATE_CASCADE)
        if not skip_dft:
            cascade_stages.append(
                (
                    "dft_grounding_score",
                    DFT_GEOMETRY_OPT_THRESHOLD,
                    "DFT geometry optimization must converge (dft_grounding_score >= 0.80)",
                ),
            )
        # Synthesizability route hard gate (WS-4: G4)
        cascade_stages.append(
            (
                "synthesizability_route",
                True,
                "Must have at least one plausible retrosynthetic route to commercial precursors",
            ),
        )
        rejection_log: dict[str, int] = {}
        selected: list[dict[str, Any]] = []

        remaining = list(candidates)
        for key, threshold, _desc in cascade_stages:
            passed = [c for c in remaining if _passes_stage(c, key, threshold)]
            rejected = len(remaining) - len(passed)
            rejection_log[key] = rejected
            remaining = passed

        selected = remaining
        return selected, rejection_log

    def _render_markdown(
        self,
        candidates: list[dict[str, Any]],
        selected: list[dict[str, Any]],
        rejection_log: dict[str, int],
    ) -> str:
        """Render the standardized markdown handoff report."""
        _load_known_electrolytes()
        sorted(candidates, key=lambda c: -float(c.get("total_score", 0.0)))[:DEFAULT_TOP_N]

        lines: list[str] = []
        lines.append("# Prospective Candidates Report for Wet-Lab Validation\n")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append("## Selection Summary\n")
        total = len(candidates) + sum(rejection_log.values())
        lines.append(f"- **Total candidates evaluated**: {total}\n")
        lines.append(f"- **Final selected**: {len(selected)}\n")
        for stage, rej in rejection_log.items():
            lines.append(f"- **Rejected at {stage}**: {rej}\n")

        lines.append("\n## Cascade Funnel Visualization\n")
        lines.append("| Stage | Passed | Rejected | Total | Success Rate |\n")
        lines.append("|-------|--------|----------|-------|--------------|\n")
        cumulative = total
        running = total
        for stage, rej in rejection_log.items():
            passed = running - rej
            running = passed
            rate = (passed / cumulative * 100) if cumulative > 0 else 0.0
            label = stage.replace("_", " ").title()
            lines.append(f"| {label} | {passed} | {rej} | {cumulative} | {rate:.1f}% |\n")

        lines.append("\n## Final Candidate Table\n")
        lines.append("| Rank | SMILES | Score | HOMO | LUMO | Gap | SA Feas | Conf | Novelty | Grounding | Domain | Nearest | Hints | Risk |\n")
        lines.append("|------|--------|-------|------|------|-----|---------|------|---------|-----------|--------|---------|-------|------|\n")
        for i, c in enumerate(selected, 1):
            smi = str(c.get("smiles", ""))[:40]
            smi_disp = f"`{smi}{'...' if len(str(c.get('smiles',''))) > 40 else ''}`"
            homo = c.get("homo_eV", "N/A")
            lumo = c.get("lumo_eV", "N/A")
            gap = c.get("gap_eV", "N/A")
            nearest = c.get("nearest_known_electrolytes", [])
            nearest_str = ""
            if nearest:
                nearest_str = f"{str(nearest[0].get('smiles',''))[:20]} (T={nearest[0].get('similarity',0):.2f})"
            lines.append(
                f"| {i} | {smi_disp} | {c.get('total_score', 0.0):.1f} | {homo} | {lumo} | {gap} | "
                f"{c.get('synthesis_feasibility', 0):.3f} | {c.get('conformal_confidence', 0):.3f} | "
                f"{c.get('novelty_to_seed', 0):.3f} | {c.get('combined_grounding_score', 0):.3f} | "
                f"{c.get('domain_penalty', 1):.3f} | `{nearest_str}` | {c.get('synthesis_hints','')} | {c.get('risk_flags','none')} |\n"
            )

        lines.append("\n## Selection Rationale\n")
        for i, c in enumerate(selected, 1):
            lines.append(f"### Candidate {i}: {c.get('smiles', '')}\n")
            reasons = []
            if not c.get("is_viable"):
                reasons.append("Not viable")
            if float(c.get("combined_grounding_score", 0.0)) < 0.75:
                reasons.append(f"Insufficient grounding ({c.get('combined_grounding_score', 0):.3f} < 0.75)")
            if (sc := c.get("synthesizability_complexity")) is not None and sc < 0.6:
                reasons.append(f"Complexity {sc:.3f} < 0.6")
            if float(c.get("domain_penalty", 1.0)) < 0.95:
                reasons.append(f"Domain {c.get('domain_penalty', 1):.3f} < 0.95")
            if float(c.get("novelty_to_seed", 0.0)) < 0.3:
                reasons.append(f"Novelty {c.get('novelty_to_seed', 0):.3f} < 0.3")
            if reasons:
                lines.append(f"**Would have been rejected for:** {', '.join(reasons)}\n")
            else:
                lines.append("**Passed all cascade filters**\n")
            lines.append(f"- Total Score: {c.get('total_score', 0.0):.2f}\n")
            lines.append(f"- HOMO/LUMO/Gap: {c.get('homo_eV')} / {c.get('lumo_eV')} / {c.get('gap_eV')} eV\n")
            lines.append(f"- Synthesis Feasibility: {c.get('synthesis_feasibility', 0):.3f}\n")
            lines.append(f"- Confidence / Novelty: {c.get('conformal_confidence', 0):.3f} / {c.get('novelty_to_seed', 0):.3f}\n")
            lines.append(f"- Hints: {c.get('synthesis_hints', 'N/A')}\n")
            lines.append(f"- Risk: {c.get('risk_flags', 'none')}\n")

        if not selected:
            lines.append("### No candidates pass the cascade — DFT re-ranking of top-20 is mandatory.\n")

        return "".join(lines)

    def _render_csv(
        self,
        candidates: list[dict[str, Any]],
        output_path: str,
    ) -> None:
        """Write candidates to CSV in legacy-compatible format."""
        selected_smis: list[str] = []
        if not candidates:
            return
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CANDIDATE_FIELDS + ["nearest_known_electrolytes"])
            writer.writeheader()
            for c in candidates:
                selected_smis = [str(x.get("smiles", "")) for x in candidates if x is not c]
                dp = _diversity_penalty(selected_smis, str(c.get("smiles", "")))
                base = float(c.get("adjusted_score", c.get("total_score", 0.0)))
                row = dict(c)
                row["adjusted_score"] = round(base * (1.0 - 0.3 * dp), 4)
                row["diversity_penalty"] = round(dp, 4)
                for k in CANDIDATE_FIELDS + ["nearest_known_electrolytes"]:
                    if k not in row:
                        row[k] = ""
                writer.writerow(row)

    def _run_dft_validation(
        self,
        candidates: list[dict[str, Any]],
        cache_path: str = "dft_cache.json",
    ) -> dict[str, Any] | None:
        """Optional ORCA wB97X-D3/def2-SVP re-ranking of the top-N."""
        try:
            from aurelius.scoring.oracle.dft_validator import DFTValidator, has_orca
        except Exception:
            return None
        if len(candidates) < 3 or not has_orca():
            return None

        validator = DFTValidator(cache_path=cache_path)
        top_n = sorted(candidates, key=lambda c: -float(c.get("total_score", 0.0)))[:DEFAULT_TOP_N]
        for cand in top_n:
            ctx = MoleculeContext.from_smiles(str(cand["smiles"]))
            if ctx is None:
                continue
            dft = validator.compute(ctx.mol)
            if dft is not None:
                cand["dft_homo_eV"] = round(dft["homo_eV"], 4)
                cand["dft_lumo_eV"] = round(dft["lumo_eV"], 4)
                cand["dft_composite"] = round(-(dft["homo_eV"] + dft["lumo_eV"]) / 2.0, 4)
            else:
                cand["dft_homo_eV"] = None
                cand["dft_lumo_eV"] = None
                cand["dft_composite"] = None
        return validator.validate_ranking(
            [float(c.get("adjusted_score", c.get("total_score", 0.0))) for c in top_n],
            [
                ctx.mol
                for c in top_n
                if (ctx := MoleculeContext.from_smiles(str(c["smiles"]))) is not None
            ],
        )

    def generate_report(
        self,
        *,
        n_generations: int = DEFAULT_N_GENERATIONS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        top_n: int = DEFAULT_TOP_N,
        output_dir: str = ".",
        dft: bool = False,
        dft_cache: str = DFT_CACHE_PATH,
        skip_dft: bool = False,
    ) -> tuple[list[dict[str, Any]], str]:
        """Run a discovery loop and emit standardized wet-lab handoff artifacts.

        Returns ``(selected_candidates, report_markdown)`` and writes:
          - ``<output_dir>/prospective_candidates.csv``
          - ``<output_dir>/prospective_candidates_report.md``

        Args:
            skip_dft: If True, the DFT geometry-optimization cascade gate is
                skipped (all candidates get ``dft_grounding_score=1.0``).
            dft_cache: Path to the JSON cache file for DFT geometry optimization
                results, keyed by canonical SMILES.
        """
        print(f"Running {n_generations}-generation discovery loop...")
        results = self.run_discovery(n_generations=n_generations, batch_size=batch_size)
        all_results: list[dict[str, Any]] = cast(
            list[dict[str, Any]], results.get("all_results", [])
        )
        getattr(results, "discoveries", results.get("discoveries", [])) if isinstance(results, dict) else getattr(results, "discoveries", [])

        known = _load_known_electrolytes()
        print(f"Loaded {len(known)} known commercial electrolytes for similarity lookup")

        candidates = []
        for r in all_results:
            cand = self._build_candidate(r, known, skip_dft=skip_dft, dft_cache_path=dft_cache)
            if cand is not None:
                candidates.append(cand)

        print(f"Discovery complete: {len(all_results)} evaluated, {len(candidates)} passed pre-filters")

        selected, rejection_log = self._apply_cascade(candidates, skip_dft=skip_dft)
        print(f"Cascade: {len(selected)}/{len(candidates)} selected")

        report = self._render_markdown(candidates, selected, rejection_log)

        csv_candidates = selected or sorted(candidates, key=lambda c: -float(c.get("total_score", 0.0)))[:top_n]

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._render_csv(csv_candidates, str(out / "prospective_candidates.csv"))
        report_path = out / "prospective_candidates_report.md"
        with open(report_path, "w") as f:
            f.write(report)

        if dft:
            dft_metrics = self._run_dft_validation(csv_candidates, dft_cache)
            if dft_metrics:
                rho = dft_metrics.get("rho_composite", 0.0)
                report += f"\n## DFT Re-Ranking (wB97X-D3/def2-SVP)\n\nSpearman rho = {rho:.4f}.\n\n"
                with open(report_path, "w") as f:
                    f.write(report)

        print(f"Selected {len(csv_candidates)} candidates -> {out / 'prospective_candidates.csv'}")
        print(f"Report written to {report_path}")
        return csv_candidates, report


def _passes_stage(candidate: dict[str, Any], key: str, threshold: object) -> bool:
    value = candidate.get(key)
    if key == "is_viable":
        return bool(value)
    if key == "synthesizability_route":
        # Hard gate: must have a plausible retrosynthetic route
        return bool(value)
    if isinstance(threshold, (int, float)) and isinstance(value, (int, float)):
        if key == "synthesizability_complexity":
            return value >= threshold  # higher complexity = more makeable
        if key == "synthesis_depth":
            # Retained for backward compatibility: treat depth > 4 as too deep
            return value <= threshold
        return value >= threshold
    return bool(value == threshold)


def _make_pipeline() -> AureliusPipeline | None:
    """Create the optional pipeline wrapper (kept private to the reporter)."""
    from aurelius.pipeline import AureliusPipeline

    pipeline = AureliusPipeline()
    pipeline.initialize()
    return pipeline


def generate_report(
    n_generations: int = DEFAULT_N_GENERATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    top_n: int = DEFAULT_TOP_N,
    output_dir: str = ".",
    dft: bool = False,
    skip_dft: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    """Module-level convenience entry point for ``aurelius report``."""
    engine = ReportingEngine()
    return engine.generate_report(
        n_generations=n_generations,
        batch_size=batch_size,
        top_n=top_n,
        output_dir=output_dir,
        dft=dft,
        skip_dft=skip_dft,
    )
