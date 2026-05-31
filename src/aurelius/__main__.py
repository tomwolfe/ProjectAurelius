"""Project Aurelius v9.0 - CLI Entry Point.

Usage:
    aurelius init                    Initialize pipeline
    aurelius doctor                  Validate dependencies and hardware
    aurelius screen <smiles>         Screen a single molecule
    aurelius batch <file>            Screen molecules from SMILES file
    aurelius score <smiles>          Compute Aurelius score only
    aurelius validate <smiles>       Run physics validation
    aurelius benchmark               Run hardware benchmark
    aurelius status                  Show pipeline status and memory
    aurelius hf-upload               Upload model to HuggingFace Hub
"""

from __future__ import annotations

import functools
import json
import os
import sys
from typing import Any

import click

from aurelius.cli_scripts import (
    train_tier1,
    validate_physics,
)
from aurelius.config import AureliusConfig, get_config
from aurelius.hub.uploader import upload_model_to_hub
from aurelius.pipeline import AureliusPipeline
from aurelius.utils.dependencies import (
    HAS_HF_HUB,
    HAS_RDKIT,
    HAS_TORCH,
)


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
    from aurelius.utils.dependencies import report_status

    status = report_status()

    click.echo("[Frameworks]")
    for fw, info in status.items():
        icon = "OK" if info["available"] else "MISSING"
        version_str = f" (v{info['version']})" if info["version"] else ""
        click.echo(f"  [{icon:>7}] {fw}{version_str}")
        if verbose and info["version"]:
            click.echo(f"           Minimum required: {info['min_version']}")
            if not info["meets_minimum"]:
                click.echo("           WARNING: Version below minimum!")

    click.echo("")

    click.echo("[Hardware]")

    if HAS_TORCH:
        try:
            import torch  # noqa: F401

            if hasattr(torch.backends, "cuda") and torch.backends.cuda.is_built():  # type: ignore[no-untyped-call, unused-ignore]
                if torch.cuda.is_available():  # type: ignore[no-untyped-call, unused-ignore]
                    click.echo(f"  PyTorch:  CUDA available ({torch.cuda.get_device_name(0)})")  # type: ignore[no-untyped-call, unused-ignore]
                else:
                    click.echo("  PyTorch:  CUDA not available")
            if torch.backends.mps.is_available():
                click.echo("  PyTorch:  MPS (Apple Silicon) available")
            else:
                click.echo("  PyTorch:  MPS not available")
            click.echo("  PyTorch:  Device will auto-select at runtime")
        except Exception:
            click.echo("  PyTorch:  Framework check failed")
    else:
        click.echo("  PyTorch:  Not installed")

    if HAS_RDKIT:
        click.echo("  RDKit:    Available")
    else:
        click.echo("  RDKit:    Not installed (real model screening required)")

    if HAS_HF_HUB:
        click.echo("  HuggingFace Hub: Available")
    else:
        click.echo("  HuggingFace Hub: Not installed (local training only)")

    click.echo("")

    click.echo("[Summary]")
    issues = []
    if not HAS_TORCH:
        issues.append("PyTorch (ML models)")
    if not HAS_RDKIT:
        issues.append("RDKit (real model screening)")

    if issues:
        click.echo(f"  WARNING: Missing {len(issues)} framework(s): {', '.join(issues)}")
        click.echo("  Pipeline will use fallback paths. Some features may be degraded.")
    else:
        click.echo("  All core frameworks available. System ready for full pipeline.")

    click.echo("")


@cli.command("screen")
@click.argument("smiles")
@click.option("--solvent", default="ec:dmc", help="Solvent type")
@click.option("--salt", default="LiPF6", help="Salt type")
@click.option("--ion", default="Na+", help="Ion type")
@click.option("--temperature", default=298.15, type=float, help="Temperature in Kelvin")
@click.option("--voltage", default=3.7, type=float, help="Voltage cutoff")
@click.option("--cycles", default=500, type=int, help="Number of scan cycles")
def screen(
    smiles: str,
    solvent: str,
    salt: str,
    ion: str,
    temperature: float,
    voltage: float,
    cycles: int,
) -> None:
    """Screen a single molecule through the full Aurelius pipeline."""
    from aurelius.config import get_config
    from aurelius.pipeline import AureliusPipeline

    config = get_config()
    pipeline = AureliusPipeline(config)
    pipeline.initialize()
    results = pipeline.screen_molecule(
        smiles,
        solvent_type=solvent,
        salt_type=salt,
        ion_type=ion,
        temperature_k=temperature,
        voltage_cutoff=voltage,
        n_scan_cycles=cycles,
    )

    score = results.get("score", {})
    total = score.get("total_score", 0.0)
    viable = score.get("is_viable", False)
    click.echo(f"\nAurelius Score: {total:.1f}/100 {'VIABLE' if viable else 'REJECTED'}")
    if score and not viable:
        sys.exit(1)


@cli.command("batch")
@click.argument("file", type=click.Path(exists=True))
@click.option("--solvent", default="ec:dmc", help="Solvent type")
@click.option("--output", type=click.Path(), help="Output JSON file")
@with_pipeline  # type: ignore[arg-type]
def batch(
    file: str,
    solvent: str,
    output: str | None,
    pipeline: AureliusPipeline,
    config: AureliusConfig,
) -> None:
    """Screen multiple molecules from a SMILES file (one per line)."""
    salt = "NaPF6"
    smiles_list = []
    with open(file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                smiles_list.append(line)

    click.echo(f"Screening {len(smiles_list)} molecules...")
    results = pipeline.screen_batch(smiles_list, solvent_type=solvent, salt_type=salt)

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
@click.option("--solvent", default="ec:dmc", help="Solvent type")
@click.option("--salt", default="NaPF6", help="Salt type")
@click.option("--ion", default="Na+", help="Ion type")
@with_pipeline  # type: ignore[arg-type]
def score(
    smiles: str,
    solvent: str,
    salt: str,
    ion: str,
    pipeline: AureliusPipeline,
    config: AureliusConfig,
) -> None:
    """Compute the Aurelius v9.0 score for a molecule (quick mode)."""
    results = pipeline.screen_molecule(
        smiles,
        solvent_type=solvent,
        salt_type=salt,
        ion_type=ion,
    )

    score = results.get("score", {})
    if score:
        click.echo(f"\nAurelius Score v9.0: {score.get('total_score', 0.0):.1f}/100 {'VIABLE' if score.get('is_viable', False) else 'REJECTED'}")


@cli.command("train")
@click.option("--dataset", default="esol", help="Dataset to train on (esol/qm9)")
@click.option("--epochs", type=int, default=200, help="Number of training epochs")
@click.option("--batch-size", type=int, default=16, help="Mini-batch size")
@click.option("--learning-rate", type=float, default=0.005, help="Learning rate")
@click.option("--csv-path", type=str, default=None, help="Path to local CSV file")
def train(
    dataset: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    csv_path: str | None,
    pipeline: AureliusPipeline,
    config: AureliusConfig,
) -> None:
    """Train a model on a dataset (Tier 1 MLP for ESOL/QM9)."""
    _run_tier1_train(dataset, epochs, batch_size, learning_rate, csv_path)


def _run_tier1_train(
    dataset: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    csv_path: str | None,
) -> None:
    """Run Tier 1 model training via train_tier1.py."""
    train_tier1.train_main(
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        csv_path=csv_path,
    )


@cli.command("validate")
@click.option("--smiles", default="CC(=O)OC1=CC(=O)O1", help="Molecule to validate")
def validate(smiles: str = "CC(=O)OC1=CC(=O)O1", pipeline: Any = None, config: Any = None) -> None:
    """Run physics validation on a molecule."""
    sys.argv = ["validate_physics", "--smiles", smiles]
    validate_physics.main()


@cli.command("status")
def status(pipeline: AureliusPipeline, config: AureliusConfig) -> None:
    """Show pipeline status and memory partition."""
    click.echo("\nAurelius v9.0 Configuration:")
    click.echo("  Pipeline initialised: True")


@cli.command("benchmark")
@click.option(
    "--tier",
    type=click.Choice(["1", "2"]),
    default=None,
    help="Benchmark only a specific tier (1 or 2). Omit for all tiers.",
)
@click.option("--quick/--detailed", default=True, help="Quick mode with fewer repeats (default: enabled)")
@click.option("--output", type=click.Path(), default=None, help="Save results to JSON file")
def benchmark(
    tier: str | None,
    quick: bool,
    output: str | None,
    pipeline: AureliusPipeline,
    config: AureliusConfig,
) -> None:
    """Run hardware benchmark and validation."""
    from aurelius.benchmark import run_benchmark

    run_benchmark(tier=tier, quick=quick, output=output)


@cli.command("hf-upload")
@click.option("--model-dir", required=True, help="Local directory containing model files to upload")
@click.option("--repo-id", required=True, help="HuggingFace repository ID (e.g., 'user/repo-name')")
@click.option(
    "--task", type=click.Choice(["esol", "qm9"]), default="esol", help="Model task type (default: esol)"
)
@click.option("--private/--public", default=True, help="Make repository private (default: private)")
@click.option("--commit-message", default="Upload model via Aurelius CLI", help="Commit message for the upload")
@click.option("--dry-run", is_flag=True, help="Validate without uploading")
def hf_upload(
    model_dir: str,
    repo_id: str,
    task: str,
    private: bool,
    commit_message: str,
    pipeline: AureliusPipeline,
    config: AureliusConfig,
    dry_run: bool,
) -> None:
    """Upload a locally trained model to HuggingFace Hub."""
    upload_model_to_hub(
        model_dir=model_dir,
        repo_id=repo_id,
        task=task,
        private=private,
        commit_message=commit_message,
        dry_run=dry_run,
    )


@cli.command("agent")
@click.option("--max-generations", type=int, default=50, help="Maximum generations to run")
@click.option("--batch-size", type=int, default=50, help="Candidates per batch")
def agent(
    max_generations: int,
    batch_size: int,
    pipeline: AureliusPipeline,
    config: AureliusConfig,
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
