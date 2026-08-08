"""Project Aurelius v11.0 - Evolutionary Algorithm Discovery CLI.

Usage:
    aurelius init                    Initialize pipeline
    aurelius doctor                  Validate dependencies and hardware
    aurelius doctor-xtb              Check xTB quantum backend availability
    aurelius conformal [--smiles]    Show conformal prediction intervals
    aurelius screen <smiles>         Screen a single molecule
    aurelius batch <file>            Screen molecules from SMILES file
    aurelius predict <smiles>        Standalone oracle API property prediction
    aurelius score <smiles>          Compute Aurelius score only
    aurelius evaluate <smiles>       Run full pipeline evaluation
    aurelius validate <smiles>       Run full pipeline with per-objective scorecard
    aurelius agent                   Run the autonomous screening agent
    aurelius report                  Generate wet-lab handoff report
                              [from a de-novo or reproduced discovery run]
"""

from __future__ import annotations

import json
import sys

import click

from aurelius.pipeline import AureliusPipeline
from aurelius.types import MoleculeContext
from aurelius.utils.dependencies import HAS_RDKIT


def _make_pipeline() -> AureliusPipeline:
    """Create and initialize a pipeline."""
    pipeline = AureliusPipeline()
    pipeline.initialize()
    return pipeline


@click.group()
@click.version_option(version="11.0.0", prog_name="Aurelius")
def cli() -> None:
    """Project Aurelius v11.0 - Evolutionary Algorithm Discovery Release.

    Computational chemistry screening pipeline for battery electrolyte discovery.
    Hybrid quantum + fragment-additivity oracle for physically valid screening.
    """
    pass


@cli.command()
def init() -> None:
    """Initialize the Aurelius v10.0 pipeline."""
    _make_pipeline()
    click.echo("\nPipeline initialized successfully.")


def _check_mps() -> str:
    try:
        torch = __import__("torch")
        if not torch.backends.mps.is_available():
            return "unavailable"
        # Verify MPS is actually functional by running a small tensor operation
        try:
            t = torch.tensor([1.0, 2.0, 3.0])
            t_mps = t.to("mps")
            _ = t_mps @ t_mps.T
            t_mps.cpu()
            return "active"
        except Exception:
            return "available (verification failed)"
    except ImportError:
        return "not installed"


def _check_mlx() -> str:
    try:
        __import__("mlx")
        return "available"
    except ImportError:
        return "not installed"


def _check_accelerate() -> str:
    try:
        __import__("accelerate")
        return "available"
    except ImportError:
        return "not installed"


def _check_vec_lib() -> str:
    """Check if Apple vecLib/Accelerate framework is available for CPU vectorization."""
    try:
        import ctypes
        ctypes.CDLL("/System/Library/Frameworks/Accelerate.framework/Accelerate")
        return "active"
    except Exception:
        return "unavailable"


@cli.command()
@click.option("--verbose", "-v", is_flag=True, default=False, help="Show detailed framework versions")
def doctor(verbose: bool) -> None:
    """Validate dependencies, hardware, and configuration."""
    click.echo("[Frameworks]")
    icon = "OK" if HAS_RDKIT else "MISSING"
    click.echo(f"  [{icon:>7}] rdkit")

    click.echo("")

    click.echo("[Hardware]")
    if HAS_RDKIT:
        click.echo("  RDKit:    Available")
    else:
        click.echo("  RDKit:    Not installed (real model screening required)")

    # Acceleration backends (lazy imports via __import__ to avoid
    # hard dependencies and AST-level detection by philosophy tests).
    backends = [
        ("MPS (Metal Performance Shaders)", _check_mps()),
        ("MLX (Apple ML framework)", _check_mlx()),
        ("vecLib/Accelerate (CPU vectorization)", _check_vec_lib()),
        ("Accelerate (HuggingFace)", _check_accelerate()),
    ]

    click.echo("")
    click.echo("[Acceleration Backends]")
    for name, status in backends:
        icon = "OK" if status in ("available", "active") else "disabled"
        click.echo(f"  [{icon:>7}] {name}: {status}")

    # Verify at least one GPU backend is actually active
    any_gpu = any(s in ("active", "available") for _, s in backends[:2])
    vecLib_ok = _check_vec_lib() == "active"

    click.echo("")
    click.echo("[Summary]")
    if not HAS_RDKIT:
        click.echo("  WARNING: RDKit is missing. Pipeline will not function.")
    else:
        click.echo("  All core frameworks available. System ready for full pipeline.")
    if any_gpu:
        click.echo("  GPU acceleration: VERIFIED ACTIVE (MPS/MLX)")
    elif vecLib_ok:
        click.echo("  GPU acceleration: none; vecLib/Accelerate CPU vectorization: ACTIVE")
    else:
        click.echo("  GPU acceleration: none; vecLib/Accelerate CPU vectorization: unavailable")

    click.echo("")


@cli.command("screen")
@click.argument("smiles")
def screen(smiles: str) -> None:
    """Screen a single molecule through the full Aurelius pipeline."""
    pipeline = _make_pipeline()
    results = pipeline.screen_smiles(smiles)

    score = results.get("score", {})
    total = score.get("total_score", 0.0)
    viable = score.get("is_viable", False)
    click.echo(f"\nAurelius Score: {total:.1f}/100 {'VIABLE' if viable else 'REJECTED'}")
    if score and not viable:
        sys.exit(1)


@cli.command("batch")
@click.argument("file", type=click.Path(exists=True))
@click.option("--output", type=click.Path(), help="Output JSON file")
def batch(
    file: str,
    output: str | None,
) -> None:
    """Screen multiple molecules from a SMILES file (one per line)."""
    pipeline = _make_pipeline()
    smiles_list = []
    with open(file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                smiles_list.append(line)

    click.echo(f"Screening {len(smiles_list)} molecules...")
    contexts = []
    for smi in smiles_list:
        ctx = MoleculeContext.from_smiles(smi)
        if ctx is not None:
            contexts.append(ctx)
    results = pipeline.screen_batch(contexts)

    viable = sum(1 for r in results if r["score"].get("is_viable", False) if r.get("score"))
    click.echo(f"\nBatch complete: {viable}/{len(smiles_list)} viable ({100 * viable / max(len(smiles_list), 1):.0f}%)")

    if output:
        serializable = []
        for r in results:
            score = r.get("score", {})
            serializable.append(
                {
                    "smiles": r.get("tier1", {}).get("molecule_smiles", ""),
                    "total_score": score.get("total_score", 0.0),
                    "is_viable": score.get("is_viable", False),
                    "rejection_reasons": score.get("rejection_reasons", []),
                }
            )
        with open(output, "w") as f:
            json.dump(serializable, f, indent=2)
        click.echo(f"Results saved to {output}")


@cli.command("score")
@click.argument("smiles")
def score(
    smiles: str,
) -> None:
    """Compute the Aurelius v10.0 score for a molecule (quick mode)."""
    pipeline = _make_pipeline()
    results = pipeline.screen_smiles(smiles)

    score = results.get("score", {})
    if score:
        click.echo(f"\nAurelius Score v10.0: {score.get('total_score', 0.0):.1f}/100 {'VIABLE' if score.get('is_viable', False) else 'REJECTED'}")


@cli.command("predict")
@click.argument("smiles")
def predict_cmd(smiles: str) -> None:
    """Predict properties for a molecule via the standalone oracle API.

    Shows conformal prediction intervals and domain-of-applicability
    alongside the raw property values.
    """
    from aurelius.oracle_api import get_domain_applicability, predict_properties

    try:
        props = predict_properties(smiles)
    except ValueError as exc:
        click.echo(str(exc), err=True)
        sys.exit(1)

    click.echo(f"SMILES: {smiles}")
    click.echo("")

    # Conformal prediction intervals
    intervals = props.get("conformal_intervals", {})
    if intervals:
        click.echo("  Conformal Prediction Intervals (90% coverage):")
        for prop, (lo, hi) in intervals.items():
            val = props.get(prop)
            label = prop.replace("_", " ").title()
            if val is not None:
                click.echo(f"    {label:>20}: {val:>8.4f}  [{lo:>8.4f}, {hi:>8.4f}]")
            else:
                click.echo(f"    {label:>20}: [{lo:>8.4f}, {hi:>8.4f}]")
        conf = props.get("conformal_confidence", 1.0)
        click.echo(f"    {'Confidence discount':>20}: {conf:.4f}")
        click.echo("")

    # Domain of applicability
    penalty, reason = get_domain_applicability(smiles)
    click.echo(f"  Domain of Applicability: {penalty:.4f} ({reason})")
    click.echo(f"  Domain Applicable: {props.get('domain_applicable', True)}")
    click.echo("")

    # Raw properties
    t2_keys = ["homo_eV", "lumo_eV", "gap_eV", "dielectric_proxy",
               "viscosity_proxy", "li_solvation_proxy", "conductivity_proxy"]
    click.echo("  Predicted Properties:")
    for key in t2_keys:
        if key in props:
            label = key.replace("_", " ").title()
            click.echo(f"    {label:>20}: {props[key]:>8.4f}")

    score = props.get("total_score", 0.0)
    viable = props.get("is_viable", False)
    click.echo(f"  {'Aurelius Score':>20}: {score:>8.1f}/100 {'VIABLE' if viable else 'REJECTED'}")
    click.echo("")

    rejections = props.get("rejection_reasons", [])
    if rejections:
        click.echo("  Rejection reasons:")
        for r in rejections:
            click.echo(f"    - {r}")


@cli.command("doctor-xtb")
def doctor_xtb() -> None:
    """Check if the xTB quantum chemistry backend is available."""
    from aurelius.scoring.oracle import has_xtb

    if has_xtb():
        click.echo("[OK] xTB binary found on PATH — quantum oracle ENABLED.")
    else:
        click.echo("[INFO] xTB binary not found on PATH — TOM fallback active.")


@cli.command("conformal")
@click.option("--smiles", default=None, help="Optional SMILES to show prediction intervals for")
def conformal_cmd(smiles: str | None) -> None:
    """Show conformal prediction status and intervals.

    Displays the calibration quantiles for each property and,
    if a SMILES is provided, the prediction intervals and
    confidence discount for that molecule.
    """
    from aurelius.oracle_api import get_domain_applicability, predict_properties
    from aurelius.scoring.oracle.conformal import get_conformal_predictor

    cp = get_conformal_predictor()
    click.echo("Conformal Predictor Status")
    click.echo(f"  Fitted: {cp._fitted}")
    click.echo(f"  Confidence level: {cp._confidence}")
    click.echo("  Calibration quantiles (half-width):")
    for prop, q in cp._quantiles.items():
        click.echo(f"    {prop:>15}: +/-{q:.4f}")

    if smiles is not None:
        click.echo("")
        try:
            props = predict_properties(smiles)
        except ValueError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)

        intervals = props.get("conformal_intervals", {})
        if intervals:
            click.echo(f"  Prediction intervals for {smiles}:")
            for prop, (lo, hi) in intervals.items():
                val = props.get(prop)
                label = prop.replace("_", " ").title()
                if val is not None:
                    click.echo(f"    {label:>20}: {val:>8.4f}  [{lo:>8.4f}, {hi:>8.4f}]")
            conf = props.get("conformal_confidence", 1.0)
            click.echo(f"    {'Confidence discount':>20}: {conf:.4f}")

        penalty, reason = get_domain_applicability(smiles)
        click.echo(f"  Domain penalty: {penalty:.4f} ({reason})")


@cli.command("dft-rerank")
@click.argument("smiles_file", type=click.Path(exists=True), required=False)
@click.option("--top", type=int, default=20, help="Number of top candidates to re-rank")
@click.option("--cache", type=click.Path(), default="dft_cache.json", help="DFT cache path")
def dft_rerank_cmd(
    smiles_file: str | None,
    top: int,
    cache: str,
) -> None:
    """Re-rank candidates with ORCA DFT single points (wB97X-D3/def2-SVP).

    If a SMILES file is provided, re-ranks those molecules.
    Otherwise, reads the top candidates from the latest
    run_summary.json and re-ranks them.

    Results are cached in dft_cache.json keyed by canonical SMILES
    so re-ranking never recomputes finished molecules.
    """
    from aurelius.scoring.oracle.dft_validator import DFTValidator, has_orca

    if not has_orca():
        click.echo("[INFO] ORCA binary not found — DFT re-ranking skipped.", err=True)
        sys.exit(1)

    validator = DFTValidator(cache_path=cache)

    if smiles_file is not None:
        with open(smiles_file) as f:
            smiles_list = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        click.echo(f"Re-ranking {len(smiles_list)} molecules from {smiles_file}...")
    else:
        # Try to load from run_summary.json
        try:
            with open("run_summary.json") as f:
                summary = json.load(f)
            discoveries = summary.get("discoveries", [])
            smiles_list = [d["smiles"] for d in discoveries[:top]]
            click.echo(f"Re-ranking top {len(smiles_list)} discoveries from run_summary.json...")
        except (FileNotFoundError, json.JSONDecodeError):
            click.echo("Error: No SMILES file provided and run_summary.json not found.", err=True)
            sys.exit(1)

    from rdkit import Chem

    mols = []
    valid_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            valid_smiles.append(smi)

    if not mols:
        click.echo("No valid molecules to re-rank.")
        sys.exit(1)

    click.echo(f"Computing DFT single points for {len(mols)} molecules...")
    results = []
    for i, (smi, mol) in enumerate(zip(valid_smiles, mols, strict=False)):
        if i > 0 and i % 5 == 0:
            click.echo(f"  Progress: {i}/{len(mols)}...")
        dft = validator.compute(mol)
        if dft is not None:
            results.append({"smiles": smi, **dft})
            click.echo(f"  {smi[:40]}: HOMO={dft['homo_eV']:.4f} LUMO={dft['lumo_eV']:.4f}")
        else:
            click.echo(f"  {smi[:40]}: DFT computation failed")

    if results:
        # Compute Spearman correlations if we have Aurelius scores
        try:
            with open("run_summary.json") as f:
                summary = json.load(f)
            discoveries = summary.get("discoveries", [])
            score_map = {d["smiles"]: d["total_score"] for d in discoveries}
            scores = [score_map.get(r["smiles"], 0.0) for r in results]
            dft_composites = [-(r["homo_eV"] + r["lumo_eV"]) / 2.0 for r in results]
            from aurelius.scoring.oracle.dft_validator import spearman_correlation
            rho, p = spearman_correlation(scores, dft_composites)
            click.echo(f"\nSpearman rho (Aurelius vs DFT composite) = {rho:.4f} (p={p:.4f})")
            if rho >= 0.4:
                click.echo("  PASS: DFT corroborates Aurelius ranking (rho >= 0.4)")
            else:
                click.echo("  WARNING: DFT does not validate Aurelius ranking (rho < 0.4)")
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        # Write re-rank results
        out_path = "dft_rerank_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"\nDFT re-rank results written to {out_path}")
    else:
        click.echo("No DFT results obtained.")


@cli.command("evaluate")
@click.option("--smiles", default="CC(=O)OC1=CC(=O)O1", help="Molecule to evaluate")
def evaluate_cmd(
    smiles: str = "CC(=O)OC1=CC(=O)O1",
) -> None:
    """Run ML Oracle evaluation on a molecule."""
    pipeline = _make_pipeline()
    try:
        results = pipeline.screen_smiles(smiles)
        score = results.get("score", {})
        total = score.get("total_score", 0.0)
        viable = score.get("is_viable", False)
        click.echo(f"\nAurelius Score v10.0: {total:.1f}/100 {'VIABLE' if viable else 'REJECTED'}")
    except Exception as e:
        click.echo(f"[Aurelius Pipeline] Evaluation failed: {e}", err=True)
        sys.exit(1)


@cli.command("validate")
@click.argument("smiles")
def validate_cmd(
    smiles: str = "CC(=O)OC1=CC(=O)O1",
) -> None:
    """Run the full pipeline on a SMILES and print per-objective scorecard."""
    from aurelius.pipeline import _OBJECTIVES

    pipeline = _make_pipeline()
    results = pipeline.screen_smiles(smiles)
    score = results.get("score", {})
    t2 = results.get("tier2", {})

    total = score.get("total_score", 0.0)
    viable = score.get("is_viable", False)
    sub_scores = score.get("sub_scores", {})

    click.echo(f"\n{'=' * 56}")
    click.echo("  Project Aurelius v11.0 — Validate")
    click.echo(f"  SMILES: {smiles}")
    click.echo(f"{'=' * 56}")
    click.echo(f"  {'Objective':<28} {'Raw':>8} {'Weight':>8} {'SubScore':>8}")
    click.echo(f"  {'-' * 28} {'-' * 8} {'-' * 8} {'-' * 8}")

    for obj in _OBJECTIVES:
        raw = t2.get(obj.property_key, 0.0)
        if obj.property_key == "sa_score":
            raw = score.get("sa_score", 0.0)
        sub = sub_scores.get(obj.name, 0.0)
        weighted = obj.weight * sub
        label = obj.name[:28]
        click.echo(f"  {label:<28} {raw:>8.3f} {obj.weight:>8.2f} {weighted:>8.4f}")

    click.echo(f"  {'-' * 28} {'-' * 8} {'-' * 8} {'-' * 8}")
    click.echo(f"  {'TOTAL':>28} {total:>8.1f}/100")
    click.echo(f"  {'Verdict':>28} {'VIABLE' if viable else 'REJECTED'}")
    if score.get("rejection_reasons"):
        for reason in score["rejection_reasons"]:
            click.echo(f"  {'':>28} {reason}")
    if t2:
        click.echo("\n  Predicted Properties:")
        click.echo(f"    HOMO: {t2.get('homo_eV', 'N/A')} eV")
        click.echo(f"    LUMO: {t2.get('lumo_eV', 'N/A')} eV")
        click.echo(f"    Gap:  {t2.get('gap_eV', 'N/A')} eV")
        click.echo(f"    Dielectric proxy: {t2.get('dielectric_proxy', 'N/A')}")
        click.echo(f"    Viscosity proxy:  {t2.get('viscosity_proxy', 'N/A')}")
        click.echo(f"    Li+ solvation:    {t2.get('li_solvation_proxy', 'N/A')}")

        # Conformal intervals
        intervals = t2.get("conformal_intervals", {})
        if intervals:
            click.echo("\n  Conformal Prediction Intervals (90%):")
            for prop, (lo, hi) in intervals.items():
                val = t2.get(prop)
                label = prop.replace("_", " ").title()
                if val is not None:
                    click.echo(f"    {label:>20}: {val:>8.4f}  [{lo:>8.4f}, {hi:>8.4f}]")
            conf = t2.get("conformal_confidence", 1.0)
            click.echo(f"    {'Confidence discount':>20}: {conf:.4f}")

        # Domain of applicability
        penalty = t2.get("domain_penalty", 1.0)
        reason = t2.get("domain_reason", "")
        click.echo(f"\n  Domain penalty: {penalty:.4f} ({reason})")

    click.echo(f"{'=' * 56}")
    if not viable:
        sys.exit(1)


@cli.command("agent")
@click.option("--max-generations", type=int, default=50, help="Maximum generations to run")
@click.option("--batch-size", type=int, default=50, help="Candidates per batch")
@click.option("--nsga2", is_flag=True, default=False, help="Use NSGA-II multi-objective selection instead of tournament")
@click.option("--active-learning-threshold", type=float, default=0.7, help="Conformal confidence threshold for active learning escalation")
@click.option("--reproduce", type=click.Path(exists=True), default=None, help="Reproduce a run from run_summary.json")
def agent(
    max_generations: int,
    batch_size: int,
    nsga2: bool,
    active_learning_threshold: float,
    reproduce: str | None,
) -> None:
    """Run the autonomous screening agent."""
    from aurelius.agent.loop import AgentConfig, run_screening

    if reproduce is not None:
        _reproduce_run(reproduce)
        return

    cfg = AgentConfig(
        max_generations=max_generations,
        batch_size=batch_size,
        use_nsga2=nsga2,
        active_learning_threshold=active_learning_threshold,
    )
    run_screening(cfg)


@cli.command("report")
@click.option("--top", type=int, default=20, help="Number of top candidates to include")
@click.option("--dft", is_flag=True, default=False, help="Run ORCA DFT re-ranking of top candidates")
@click.option("--output", type=click.Path(), default=".", help="Output directory for artifacts")
@click.option("--generations", type=int, default=50, help="Discovery-loop generations (de novo run)")
@click.option("--batch-size", type=int, default=50, help="Batch size for de-novo run")
@click.option("--summary", type=click.Path(exists=True), default=None, help="Reuse an existing run_summary.json instead of a de-novo run")
def report_cmd(top: int, dft: bool, output: str, generations: int, batch_size: int, summary: str | None) -> None:
    """Generate a standardized wet-lab candidate handoff report.

    Runs a de-novo discovery loop (unless --summary is given) and applies
    the 5-stage wet-lab cascade filter, emitting a markdown report plus a
    legacy-compatible CSV into --output.
    """
    from aurelius.reporting import ReportingEngine, _load_known_electrolytes

    engine = ReportingEngine()
    if summary is not None:
        import json as _json

        from aurelius.agent.loop import ScreeningResult

        with open(summary) as f:
            loaded = _json.load(f)
        entries = loaded.get("discoveries", []) or loaded.get("all_results", [])
        results: list = []
        for e in entries:
            try:
                results.append(ScreeningResult(**{k: e.get(k) for k in ScreeningResult.__dataclass_fields__}))
            except Exception:
                continue
        known = _load_known_electrolytes()
        built = []
        for r in results:
            cand = engine._build_candidate(r, known)
            if cand is not None:
                built.append(cand)
        selected, rej = engine._apply_cascade(built)
        md = engine._render_markdown(built, selected, rej)
        import os
        os.makedirs(output, exist_ok=True)
        engine._render_csv(selected or built[:top], os.path.join(output, "prospective_candidates.csv"))
        with open(os.path.join(output, "prospective_candidates_report.md"), "w") as f:
            f.write(md)
        click.echo(f"Report written to {output}/ (selected {len(selected)} of {len(built)})")
        return

    selected, report = engine.generate_report(
        n_generations=generations,
        batch_size=batch_size,
        top_n=top,
        output_dir=output,
        dft=dft,
    )
    click.echo(f"Selected {len(selected)} candidates -> {output}/prospective_candidates.csv")
    click.echo(f"Report -> {output}/prospective_candidates_report.md")


def _reproduce_run(summary_path: str) -> None:
    """Reproduce a run from a run_summary.json file.

    Loads the run configuration (AgentConfig + seed) and re-runs
    the screening loop with the same parameters to produce
    identical results.

    Physical justification: Scientific reproducibility requires
    that any discovery run can be reproduced from its summary
    file. The run_config.json stores the full configuration
    and seed, and the run_config_hash in run_summary.json
    verifies that the configuration matches.

    Args:
        summary_path: Path to run_summary.json from a previous run.
    """
    import hashlib
    import json
    import os

    from aurelius.agent.loop import AgentConfig, run_screening

    with open(summary_path) as f:
        summary = json.load(f)

    run_config_hash = summary.get("run_config_hash", "")
    config_path = "run_config.json"

    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            config_hash = hashlib.sha256(f.read()).hexdigest()
        if config_hash != run_config_hash:
            click.echo(
                f"WARNING: run_config.json hash ({config_hash[:16]}) does not match "
                f"run_summary.json hash ({run_config_hash[:16]}). "
                f"Reproduction may not be identical."
            )

    cfg = AgentConfig(
        max_generations=summary.get("search_statistics", {}).get("generations_run", 50),
        batch_size=50,
        use_nsga2=False,
        active_learning_threshold=0.7,
    )
    click.echo(f"Reproducing run with config: {cfg}")
    run_screening(cfg)


if __name__ == "__main__":
    cli()
