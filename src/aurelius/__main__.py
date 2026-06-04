"""Project Aurelius v10.0 - Evolutionary Algorithm Discovery CLI.

Usage:
    aurelius init                    Initialize pipeline
    aurelius doctor                  Validate dependencies and hardware
    aurelius doctor-xtb              Check xTB quantum backend availability
    aurelius screen <smiles>         Screen a single molecule
    aurelius batch <file>            Screen molecules from SMILES file
    aurelius score <smiles>          Compute Aurelius score only
    aurelius evaluate <smiles>       Run full pipeline evaluation
    aurelius agent                   Run the autonomous screening agent
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
@click.version_option(version="10.0.0", prog_name="Aurelius")
def cli() -> None:
    """Project Aurelius v10.0 - Evolutionary Algorithm Discovery Release.

    Computational chemistry screening pipeline for battery electrolyte discovery.
    Hybrid quantum + fragment-additivity oracle for physically valid screening.
    """
    pass


@cli.command()
def init() -> None:
    """Initialize the Aurelius v10.0 pipeline."""
    _make_pipeline()
    click.echo("\nPipeline initialized successfully.")


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

    click.echo("")

    click.echo("[Summary]")
    if not HAS_RDKIT:
        click.echo("  WARNING: RDKit is missing. Pipeline will not function.")
    else:
        click.echo("  All core frameworks available. System ready for full pipeline.")

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


@cli.command("doctor-xtb")
def doctor_xtb() -> None:
    """Check if the xTB quantum chemistry backend is available."""
    from aurelius.scoring.oracle import has_xtb

    if has_xtb():
        click.echo("[OK] xTB binary found on PATH — quantum oracle ENABLED.")
    else:
        click.echo("[INFO] xTB binary not found on PATH — TOM fallback active.")


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
def validate_cmd(smiles: str) -> None:
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
    click.echo("  Project Aurelius v10.0 — Validate")
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
    click.echo(f"{'=' * 56}")
    if not viable:
        sys.exit(1)


@cli.command("agent")
@click.option("--max-generations", type=int, default=50, help="Maximum generations to run")
@click.option("--batch-size", type=int, default=50, help="Candidates per batch")
def agent(
    max_generations: int,
    batch_size: int,
) -> None:
    """Run the autonomous screening agent."""
    from aurelius.agent.loop import AgentConfig, run_screening

    cfg = AgentConfig(max_generations=max_generations, batch_size=batch_size)
    run_screening(cfg)


if __name__ == "__main__":
    cli()
