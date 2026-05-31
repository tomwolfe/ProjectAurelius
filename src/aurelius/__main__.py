"""Project Aurelius v9.0 - CLI Entry Point.

Usage:
    aurelius init                    Initialize pipeline
    aurelius doctor                  Validate dependencies and hardware
    aurelius screen <smiles>         Screen a single molecule
    aurelius batch <file>            Screen molecules from SMILES file
    aurelius score <smiles>          Compute Aurelius score only
    aurelius evaluate <smiles>       Run ML Oracle evaluation
    aurelius train                   Train the QM9 surrogate model
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
@click.version_option(version="9.0.0", prog_name="Aurelius")
def cli() -> None:
    """Project Aurelius v9.0 - The Bayesian Discovery Release.

    Computational chemistry screening pipeline for battery electrolyte discovery.
    """
    pass


@cli.command()
def init() -> None:
    """Initialize the Aurelius v9.0 pipeline."""
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
    """Compute the Aurelius v9.0 score for a molecule (quick mode)."""
    pipeline = _make_pipeline()
    results = pipeline.screen_smiles(smiles)

    score = results.get("score", {})
    if score:
        click.echo(f"\nAurelius Score v9.0: {score.get('total_score', 0.0):.1f}/100 {'VIABLE' if score.get('is_viable', False) else 'REJECTED'}")


@cli.command("train")
@click.option("--model-path", default="models/oracle_rf.joblib", show_default=True,
              help="Path to save the trained RF model (.joblib)")
def train(model_path: str) -> None:
    """Retrain the Oracle's RF model on QM9 HOMO/LUMO data.

    Featurises ~500 QM9 molecules with ECFP4 + RDKit descriptors (2053-dim)
    and trains a multi-output RandomForestRegressor for HOMO/LUMO prediction.
    """
    from aurelius.scoring.oracle import train_oracle_rf

    path = train_oracle_rf(save_path=model_path)
    click.echo(f"RF model trained and saved to {path}")


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
        click.echo(f"\nAurelius Score v9.0: {total:.1f}/100 {'VIABLE' if viable else 'REJECTED'}")
    except Exception as e:
        click.echo(f"[Aurelius Pipeline] Evaluation failed: {e}", err=True)
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
