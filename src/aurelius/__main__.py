"""Project Aurelius v5.1 - CLI Entry Point.

Usage:
    aurelius init                    Initialize pipeline
    aurelius screen <smiles>         Screen a single molecule
    aurelius batch <file>            Screen molecules from SMILES file
    aurelius score <smiles>          Compute Aurelius score only
    aurelius status                  Show pipeline status and memory
"""

import json
import sys
from pathlib import Path

import click

from aurelius.config import apply_global_config
from aurelius.pipeline import AureliusPipeline


@click.group()
@click.version_option(version="5.1.0", prog_name="Aurelius")
def cli():
    """Project Aurelius v5.1 - The 2nm Fusion Edition.

    Accelerated computational chemistry screening pipeline optimized
    for Apple M-series Neural Accelerators.
    """
    pass


@cli.command()
def init():
    """Initialize the Aurelius v5.1 pipeline."""
    config = apply_global_config()
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
    smiles,
    solvent,
    salt,
    ion,
    temperature,
    voltage,
    cycles,
    gwp,
    use_real_models,
    demo,
):
    """Screen a single molecule through the full Aurelius v5.1 pipeline.

    By default, Tier 1 loads or trains on real experimental data (ESOL/QM9).
    Use --demo to switch to synthetic training data for demonstration.
    """
    # --demo overrides --use-real-models
    if demo:
        use_real_models = False

    config = apply_global_config()
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
def batch(file, solvent, salt, output):
    """Screen multiple molecules from a SMILES file (one per line)."""
    config = apply_global_config()
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
def score(smiles, solvent, salt, ion, gwp):
    """Compute the Aurelius v5.1 score for a molecule (quick mode)."""
    config = apply_global_config()
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
        click.echo(f"\nAurelius Score v5.1: {score.total_score:.1f}/100 "
                   f"{'VIABLE' if score.is_viable else 'REJECTED'}")


@cli.command("status")
def status():
    """Show pipeline status and memory partition."""
    config = apply_global_config()
    click.echo(f"\nAurelius v5.1 Configuration:")
    click.echo(f"  MLX Max Memory:    {config.mlx_max_mem_gb}GB")
    click.echo(f"  Shader Cache:      {config.metal_shader_cache_gb}GB")
    click.echo(f"  TurboQuant Context: {config.turquant_max_context:,} tokens")
    click.echo(f"  Desolvation Cutoff: {config.desolvation_barrier_threshold_eV} eV")
    click.echo(f"  Memory Valid:      {config.validate_memory_budget()}")


if __name__ == "__main__":
    cli()
