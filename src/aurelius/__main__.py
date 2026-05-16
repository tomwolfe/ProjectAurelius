"""Project Aurelius v5.2 - CLI Entry Point.

Usage:
    aurelius init                    Initialize pipeline
    aurelius screen <smiles>         Screen a single molecule
    aurelius batch <file>            Screen molecules from SMILES file
    aurelius score <smiles>          Compute Aurelius score only
    aurelius train <smiles>          Train Tier 1 model on dataset
    aurelius validate <smiles>       Run physics validation
    aurelius benchmark               Run hardware benchmark
    aurelius status                  Show pipeline status and memory
"""

import json
import os
import sys

import click

from aurelius.config import get_config
from aurelius.pipeline import AureliusPipeline


def _apply_env_thread_safe(env_vars: dict[str, str]) -> None:
    """Apply environment variables in a thread-safe manner.

    Only sets variables that are not already set by the user,
    preventing race conditions during concurrent pipeline init.
    """
    for k, v in env_vars.items():
        if k not in os.environ:
            os.environ[k] = v


@click.group()
@click.version_option(version="5.2.0", prog_name="Aurelius")
def cli() -> None:
    """Project Aurelius v5.2 - The Hardened Release.

    Accelerated computational chemistry screening pipeline optimized
    for Apple M-series Neural Accelerators.
    """
    pass


@cli.command()
def init() -> None:
    """Initialize the Aurelius v5.2 pipeline."""
    config = get_config()
    # Thread-safe environment variable application
    env_vars = config.apply_environment()
    _apply_env_thread_safe(env_vars)
    pipeline = AureliusPipeline(config)
    pipeline.initialize()
    click.echo("\nPipeline initialized successfully.")


@cli.command("screen")
@click.argument("smiles")
@click.option("--solvent", default="ec:dmc", help="Solvent type (default: ec:dmc)")
@click.option("--salt", default="NaPF6", help="Salt type (default: NaPF6)")
@click.option("--ion", default="Na+", help="Ion type (default: Na+)")
@click.option("--temperature", default=298.15, help="Temperature in Kelvin")
@click.option("--voltage", default=0.05, help="Voltage cutoff")
@click.option("--cycles", default=500, help="MD simulation cycles")
@click.option("--gwp", default=1.0, help="Global Warming Potential value")
@click.option(
    "--use-real-models",
    is_flag=True,
    default=True,
    help="Use real pre-trained models trained on experimental data (default: enabled)",
)
@click.option(
    "--demo",
    is_flag=True,
    default=False,
    help="Use synthetic training data for demonstration purposes",
)
def screen(
    smiles: str,
    solvent: str,
    salt: str,
    ion: str,
    temperature: float,
    voltage: float,
    cycles: int,
    gwp: float,
    use_real_models: bool,
    demo: bool,
) -> None:
    """Screen a single molecule through the full Aurelius v5.2 pipeline.

    By default, Tier 1 loads or trains on real experimental data (ESOL/QM9).
    Use --demo to switch to synthetic training data for demonstration.
    """
    # --demo overrides --use-real-models
    if demo:
        use_real_models = False

    config = get_config()
    env_vars = config.apply_environment()
    _apply_env_thread_safe(env_vars)
    pipeline = AureliusPipeline(config, use_real_models=use_real_models)
    pipeline.initialize()

    results = pipeline.screen_molecule(
        smiles,
        solvent_type=solvent,
        salt_type=salt,
        ion_type=ion,
        temperature_k=temperature,
        voltage_cutoff=voltage,
        n_md_cycles=cycles,
        gwp_value=gwp,
    )

    score = results.get("score")
    if score and not score.is_viable:
        sys.exit(1)


@cli.command("batch")
@click.argument("file", type=click.Path(exists=True))
@click.option("--solvent", default="ec:dmc", help="Solvent type")
@click.option("--salt", default="NaPF6", help="Salt type")
@click.option("--output", type=click.Path(), help="Output JSON file")
def batch(file: str, solvent: str, salt: str, output: str | None) -> None:
    """Screen multiple molecules from a SMILES file (one per line)."""
    config = get_config()
    env_vars = config.apply_environment()
    _apply_env_thread_safe(env_vars)
    pipeline = AureliusPipeline(config)
    pipeline.initialize()

    smiles_list = []
    with open(file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                smiles_list.append(line)

    click.echo(f"Screening {len(smiles_list)} molecules...")
    results = pipeline.screen_batch(smiles_list, solvent_type=solvent, salt_type=salt)

    # Summary
    viable = sum(1 for r in results if r["score"].is_viable)
    click.echo(f"\nBatch complete: {viable}/{len(smiles_list)} viable ({100*viable/len(smiles_list):.0f}%)")

    if output:
        # Serialize results
        serializable = []
        for r in results:
            score = r["score"]
            serializable.append({
                "smiles": score.molecule_smiles,
                "total_score": score.total_score,
                "is_viable": score.is_viable,
                "sigma": score.sigma_score,
                "desolvation": score.desolvation_score,
                "sei_homogeneity": score.sei_homogeneity_score,
                "mx_synthesis": score.mx_synthesis_score,
                "gwp_penalty": score.gwp_penalty,
                "rejection_reasons": score.rejection_reasons,
            })
        with open(output, "w") as f:
            json.dump(serializable, f, indent=2)
        click.echo(f"Results saved to {output}")


@cli.command("score")
@click.argument("smiles")
@click.option("--solvent", default="ec:dmc", help="Solvent type")
@click.option("--salt", default="NaPF6", help="Salt type")
@click.option("--ion", default="Na+", help="Ion type")
@click.option("--gwp", default=1.0, help="Global Warming Potential")
def score(smiles: str, solvent: str, salt: str, ion: str, gwp: float) -> None:
    """Compute the Aurelius v5.2 score for a molecule (quick mode)."""
    config = get_config()
    env_vars = config.apply_environment()
    _apply_env_thread_safe(env_vars)
    pipeline = AureliusPipeline(config)
    pipeline.initialize()

    results = pipeline.screen_molecule(
        smiles,
        solvent_type=solvent,
        salt_type=salt,
        ion_type=ion,
        gwp_value=gwp,
    )

    score = results.get("score")
    if score:
        click.echo(f"\nAurelius Score v5.2: {score.total_score:.1f}/100 "
                   f"{'VIABLE' if score.is_viable else 'REJECTED'}")


@cli.command("train")
@click.option("--dataset", default="esol", help="Dataset to train on (esol/qm9)")
@click.option("--epochs", type=int, default=200, help="Number of training epochs")
@click.option("--batch-size", type=int, default=16, help="Mini-batch size")
@click.option("--learning-rate", type=float, default=0.005, help="Learning rate")
@click.option("--csv-path", type=str, default=None, help="Path to local CSV file")
def train(dataset: str, epochs: int, batch_size: int, learning_rate: float, csv_path: str | None) -> None:
    """Train Tier 1 model on a dataset (esol or qm9).

    Wraps scripts/train_tier1.py as a native CLI subcommand.
    """

    import os
    import sys
    _scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts")
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from train_tier1 import train_main

    train_main(
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        csv_path=csv_path,
    )


@cli.command("validate")
@click.option("--smiles", default="CC(=O)OC1=CC(=O)O1", help="Molecule to validate")
def validate(smiles: str) -> None:
    """Run physics validation on a molecule.

    Wraps scripts/validate_physics.py as a native CLI subcommand.
    """

    import os
    import sys
    _scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts")
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from validate_physics import main as validate_main

    sys.argv = ["validate_physics.py"]
    validate_main()


@cli.command("status")
def status() -> None:
    """Show pipeline status and memory partition."""
    config = get_config()
    env_vars = config.apply_environment()
    _apply_env_thread_safe(env_vars)
    click.echo("\nAurelius v5.2 Configuration:")
    click.echo(f"  MLX Max Memory:    {config.mlx_max_mem_gb}GB")
    click.echo(f"  Shader Cache:      {config.metal_shader_cache_gb}GB")
    click.echo(f"  GCMD kMC Steps:    {config.turquant_max_context:,} steps")
    click.echo(f"  Desolvation Cutoff: {config.desolvation_barrier_threshold_eV} eV")
    click.echo(f"  Memory Valid:      {config.validate_memory_budget()}")


@cli.command("benchmark")
@click.option("--tier", type=click.Choice(["1", "2"]), default=None, help="Benchmark only a specific tier (1 or 2). Omit for all tiers.")
@click.option("--quick/--detailed", default=True, help="Quick mode with fewer repeats (default: enabled)")
@click.option("--output", type=click.Path(), default=None, help="Save results to JSON file")
def benchmark(tier: str | None, quick: bool, output: str | None) -> None:
    """Run hardware benchmark and validation.

    Verifies that the user's Apple Silicon hardware is properly
    configured and provides performance baselines for Tier 1
    (MLX inference) and Tier 2 (vectorized physics) computation.
    """
    from aurelius.benchmark import run_benchmark

    run_benchmark(tier=tier, quick=quick, output=output)


if __name__ == "__main__":
    cli()
