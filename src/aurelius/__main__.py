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

import functools
import json
import os
import sys
from typing import Any

import click

from aurelius.cli_scripts import evaluate
from aurelius.config import AureliusConfig, get_config
from aurelius.pipeline import AureliusPipeline
from aurelius.utils.dependencies import HAS_RDKIT


def _init_pipeline_from_ctx(ctx: click.Context) -> None:
    """Initialise the pipeline stored in the Click context.

    Called by ``@with_pipeline`` after the command function has been
    matched but before the command body runs.
    """
    pipeline: AureliusPipeline = ctx.ensure_object(dict)["pipeline"]
    pipeline.initialize()


def with_pipeline(command: click.Command) -> click.Command:
    """Click decorator that injects ``pipeline`` and ``config`` into the command.

    The decorated command receives ``pipeline`` and ``config`` as the
    first two positional arguments.  This eliminates the repeated
    ``config = get_config()`` / ``pipeline = AureliusPipeline(config)``
    boilerplate found in every CLI command.
    """

    @functools.wraps(command)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        ctx = click.get_current_context()
        obj: dict[str, Any] = ctx.ensure_object(dict)
        if "pipeline" not in obj:
            config = get_config()
            pipeline = AureliusPipeline(config)
            obj["config"] = config
            obj["pipeline"] = pipeline
        else:
            config = obj["config"]
            pipeline = obj["pipeline"]

        _init_pipeline_from_ctx(ctx)

        return command(pipeline, config, *args, **kwargs)  # type: ignore[return-value]

    return wrapper  # type: ignore[return-value]


@click.group()
@click.version_option(version="9.0.0", prog_name="Aurelius")
def cli() -> None:
    """Project Aurelius v9.0 - The Bayesian Discovery Release.

    Computational chemistry screening pipeline for battery electrolyte discovery.
    """
    pass


@cli.command()
@with_pipeline  # type: ignore[arg-type]
def init(pipeline: AureliusPipeline, config: AureliusConfig) -> None:
    """Initialize the Aurelius v9.0 pipeline."""
    click.echo("\nPipeline initialized successfully.")


@cli.command()
@click.option("--verbose", "-v", is_flag=True, default=False, help="Show detailed framework versions")
def doctor(verbose: bool, pipeline: AureliusPipeline | None = None, config: AureliusConfig | None = None) -> None:
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
    from aurelius.config import get_config
    from aurelius.pipeline import AureliusPipeline

    config = get_config()
    pipeline = AureliusPipeline(config)
    pipeline.initialize()
    results = pipeline.screen_molecule(smiles)

    score = results.get("score", {})
    total = score.get("total_score", 0.0)
    viable = score.get("is_viable", False)
    click.echo(f"\nAurelius Score: {total:.1f}/100 {'VIABLE' if viable else 'REJECTED'}")
    if score and not viable:
        sys.exit(1)


@cli.command("batch")
@click.argument("file", type=click.Path(exists=True))
@click.option("--output", type=click.Path(), help="Output JSON file")
@with_pipeline  # type: ignore[arg-type]
def batch(
    file: str,
    output: str | None,
    pipeline: AureliusPipeline,
    config: AureliusConfig,
) -> None:
    """Screen multiple molecules from a SMILES file (one per line)."""
    smiles_list = []
    with open(file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                smiles_list.append(line)

    click.echo(f"Screening {len(smiles_list)} molecules...")
    results = pipeline.screen_batch(smiles_list)

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
@with_pipeline  # type: ignore[arg-type]
def score(
    smiles: str,
    pipeline: AureliusPipeline,
    config: AureliusConfig,
) -> None:
    """Compute the Aurelius v9.0 score for a molecule (quick mode)."""
    results = pipeline.screen_molecule(smiles)

    score = results.get("score", {})
    if score:
        click.echo(f"\nAurelius Score v9.0: {score.get('total_score', 0.0):.1f}/100 {'VIABLE' if score.get('is_viable', False) else 'REJECTED'}")


@cli.command("train")
def train() -> None:
    """Retrain the Oracle's RF model on QM9 HOMO/LUMO data."""
    from aurelius.scoring.oracle import PropertyOracle

    oracle = PropertyOracle()
    oracle.save("oracle_cache.joblib")
    click.echo("Oracle RF model trained and saved to oracle_cache.joblib")


@cli.command("evaluate")
@click.option("--smiles", default="CC(=O)OC1=CC(=O)O1", help="Molecule to evaluate")
def evaluate_cmd(smiles: str = "CC(=O)OC1=CC(=O)O1", pipeline: Any = None, config: Any = None) -> None:
    """Run ML Oracle evaluation on a molecule."""
    sys.argv = ["evaluate", "--smiles", smiles]
    evaluate.main()


@cli.command("agent")
@click.option("--max-generations", type=int, default=50, help="Maximum generations to run")
@click.option("--batch-size", type=int, default=50, help="Candidates per batch")
def agent(
    max_generations: int,
    batch_size: int,
) -> None:
    """Run the autonomous screening agent."""
    from aurelius.agent.state import CheckpointManager
    from aurelius.cli_scripts.agent import AgentConfig, run_screening

    output_dir = os.environ.get("AURELIUS_OUTPUT_DIR")
    checkpoint = CheckpointManager(output_dir=output_dir)
    try:
        agent_cfg = AgentConfig(
            max_generations=max_generations,
            batch_size=batch_size,
        )
        run_screening(agent_cfg, checkpoint)
    except Exception as e:
        click.echo(f"[ERROR] {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
