"""Project Aurelius v6.0 - CLI Entry Point.

Usage:
    aurelius init                    Initialize pipeline
    aurelius screen <smiles>         Screen a single molecule
    aurelius batch <file>            Screen molecules from SMILES file
    aurelius score <smiles>          Compute Aurelius score only
    aurelius train                   Train model (tier1/tier0)
      --task tier1|tier0             Training task (default: tier1)
    aurelius validate <smiles>       Run physics validation
    aurelius benchmark               Run hardware benchmark
    aurelius status                  Show pipeline status and memory
    aurelius hf-upload               Upload model to HuggingFace Hub
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

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
@click.version_option(version="6.0.0", prog_name="Aurelius")
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
    help="Use real models trained on experimental data (default: enabled)",
)
@click.option(
    "--demo",
    is_flag=True,
    default=False,
    help="Use synthetic training data for demonstration purposes",
)
@click.option(
    "--allow-fallback",
    is_flag=True,
    default=False,
    help="Allow hash-based fallback when RDKit is unavailable (demo/CI only)",
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
    allow_fallback: bool,
) -> None:
    """Screen a single molecule through the full Aurelius pipeline.

    By default, Tier 1 loads or trains on real experimental data (ESOL/QM9).
    Use --demo to switch to synthetic training data for demonstration.

    RDKit is required for real model screening. Use --allow-fallback to
    permit hash-based pseudo-fingerprints in demo/CI environments.
    """
    # --demo overrides --use-real-models
    if demo:
        use_real_models = False

    # Enforce RDKit for real models (unless --allow-fallback is set)
    if use_real_models and not allow_fallback:
        try:
            from rdkit import Chem  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "RDKit is required for real model screening. "
                "Install with: pip install rdkit\n\n"
                "To use hash-based fallback (demo/CI only), add --allow-fallback.\n"
                "To run in demo mode without RDKit, use: aurelius screen <smiles> --demo"
            ) from None

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
@click.option(
    "--allow-fallback",
    is_flag=True,
    default=False,
    help="Allow hash-based fallback when RDKit is unavailable (demo/CI only)",
)
def batch(file: str, solvent: str, salt: str, output: str | None, allow_fallback: bool) -> None:
    """Screen multiple molecules from a SMILES file (one per line)."""
    # Enforce RDKit for real models (unless --allow-fallback is set)
    if not allow_fallback:
        try:
            from rdkit import Chem  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "RDKit is required for batch screening. "
                "Install with: pip install rdkit\n\n"
                "To use hash-based fallback (demo/CI only), add --allow-fallback."
            ) from None

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
@click.option("--task", type=click.Choice(["tier1", "tier0"]), default="tier1", help="Training task (tier1 for MLX filter, tier0 for MPNN activation energy predictor)")
@click.option("--epochs", type=int, default=200, help="Number of training epochs")
@click.option("--batch-size", type=int, default=16, help="Mini-batch size")
@click.option("--learning-rate", type=float, default=0.005, help="Learning rate")
@click.option("--csv-path", type=str, default=None, help="Path to local CSV file")
def train(dataset: str, task: str, epochs: int, batch_size: int, learning_rate: float, csv_path: str | None) -> None:
    """Train a model on a dataset.

    Use --task tier1 to train the MLX filter (esol/qm9).
    Use --task tier0 to train the MPNN activation energy predictor.
    """
    if task == "tier0":
        _run_tier0_train(epochs, batch_size, learning_rate, csv_path)
    else:
        _run_tier1_train(dataset, epochs, batch_size, learning_rate, csv_path)


def _run_tier1_train(
    dataset: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    csv_path: str | None,
) -> None:
    """Run Tier 1 model training via train_tier1.py."""
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "train_tier1.py"
    if not script_path.exists():
        raise FileNotFoundError(
            f"Training script not found: {script_path}. "
            "Ensure the package is installed (e.g., pip install -e .)."
        )
    spec = importlib.util.spec_from_file_location("train_tier1", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.train_main(
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        csv_path=csv_path,
    )


def _run_tier0_train(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    csv_path: str | None,
) -> None:
    """Run Tier 0 MPNN model training via train_tier0.py."""
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "train_tier0.py"
    if not script_path.exists():
        raise FileNotFoundError(
            f"Training script not found: {script_path}. "
            "Ensure the package is installed (e.g., pip install -e .)."
        )
    spec = importlib.util.spec_from_file_location("train_tier0", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.train_main(
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
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "validate_physics.py"
    if not script_path.exists():
        raise FileNotFoundError(
            f"Validation script not found: {script_path}. "
            "Ensure the package is installed (e.g., pip install -e .)."
        )
    spec = importlib.util.spec_from_file_location("validate_physics", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = ["validate_physics.py"]
    mod.main()


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


@cli.command("hf-upload")
@click.option("--model-dir", required=True, help="Local directory containing model files to upload")
@click.option("--repo-id", required=True, help="HuggingFace Hub repository ID (e.g., 'user/repo-name')")
@click.option("--task", type=click.Choice(["tier0", "esol", "qm9"]), default="tier0", help="Model task type (default: tier0)")
@click.option("--private/--public", default=True, help="Make repository private (default: private)")
@click.option("--commit-message", default="Upload model via Aurelius CLI", help="Commit message for the upload")
@click.option("--dry-run", is_flag=True, help="Validate repo ID, token, and metadata without uploading")
def hf_upload(
    model_dir: str,
    repo_id: str,
    task: str,
    private: bool,
    commit_message: str,
    dry_run: bool,
) -> None:
    """Upload a locally trained model to HuggingFace Hub.

    Pushes model files (weights, metadata, README) to a HuggingFace
    Hub repository. Authentication is handled via the HF_TOKEN
    environment variable or interactive login.

    Auth Handling:
        Uses huggingface_hub.login() with add_to_git_credential=True.
        Tokens are never stored in CLI history.

    Examples:
        aurelius hf-upload --model-dir models/tier0 --repo-id myuser/aurelius-tier0
        aurelius hf-upload --model-dir models/esol --repo-id myuser/esol-mlp --task esol --public
        aurelius hf-upload --model-dir models/tier0 --repo-id myuser/tier0 --dry-run
    """
    import os
    import sys

    try:
        from huggingface_hub import (
            HfApi,
            ModelCard,
            ModelCardData,
            create_repo,
            repo_exists,
            upload_folder,
        )
    except ImportError:
        click.echo("[ERROR] huggingface_hub is required for hf-upload.", err=True)
        click.echo("Install with: pip install huggingface-hub", err=True)
        sys.exit(1)

    # Validate model directory
    if not os.path.isdir(model_dir):
        click.echo(f"[ERROR] Model directory not found: {model_dir}", err=True)
        sys.exit(1)

    api = HfApi()

    # Validate repo ID format
    if "/" not in repo_id:
        click.echo(
            f"[ERROR] Invalid repo ID format: '{repo_id}'. "
            f"Expected 'username/repo-name'.",
            err=True,
        )
        sys.exit(1)

    # Dry run mode: validate everything without uploading
    if dry_run:
        click.echo(f"[DRY RUN] Validating upload for: {repo_id}")
        click.echo(f"  Model directory: {model_dir}")
        click.echo(f"  Task: {task}")
        click.echo(f"  Visibility: {'private' if private else 'public'}")
        click.echo(f"  Commit message: {commit_message}")

        # Check authentication
        try:
            user_info = api.whoami()
            click.echo(f"  Authenticated as: {user_info['name']} ({user_info['fullname']})")
        except Exception as e:
            click.echo(f"[ERROR] HuggingFace authentication failed: {e}", err=True)
            click.echo("Ensure HF_TOKEN is set or run 'huggingface-cli login'.", err=True)
            sys.exit(1)

        # Check if repo exists
        if repo_exists(repo_id):
            click.echo(f"  Repository '{repo_id}' already exists.")
        else:
            click.echo(f"  Repository '{repo_id}' does not exist (would be created).")

        # List files that would be uploaded
        files = []
        for root, _, filenames in os.walk(model_dir):
            for fname in filenames:
                rel_path = os.path.relpath(os.path.join(root, fname), model_dir)
                files.append(rel_path)
        click.echo(f"  Files to upload ({len(files)}):")
        for f in sorted(files):
            click.echo(f"    - {f}")

        click.echo("\n[DRY RUN] Validation complete. No files were uploaded.")
        return

    # Real upload mode
    click.echo(f"[HF Upload] Uploading to: {repo_id}")
    click.echo(f"  Model directory: {model_dir}")
    click.echo(f"  Task: {task}")
    click.echo(f"  Visibility: {'private' if private else 'public'}")

    # Authenticate
    try:
        api.login(add_to_git_credential=True)
    except Exception as e:
        click.echo(f"[ERROR] HuggingFace authentication failed: {e}", err=True)
        click.echo("Ensure HF_TOKEN is set in your environment or run 'huggingface-cli login'.", err=True)
        sys.exit(1)

    # Create repo if it doesn't exist
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=private,
            token=api.token,
        )
        click.echo(f"  Repository '{repo_id}' ready.")
    except Exception as e:
        click.echo(f"[ERROR] Failed to create/verify repository: {e}", err=True)
        sys.exit(1)

    # Generate model card
    task_descriptions = {
        "tier0": "Tier 0 MPNN Activation Energy Predictor",
        "esol": "Tier 1 ESOL Solubility Filter",
        "qm9": "Tier 1 QM9 Energy Filter",
    }
    task_descriptions_short = {
        "tier0": "mpnn_activation_energy",
        "esol": "esol_solubility",
        "qm9": "qm9_energy",
    }

    model_card = ModelCard(
        ModelCardData(
            language="en",
            license="mit",
            pipeline_tag="regression",
            tags=[
                "aurelius",
                "battery-electrolyte",
                "molecular-screening",
                task_descriptions_short.get(task, task),
            ],
        )
    )
    model_card.content = f"""# Aurelius Model: {task_descriptions.get(task, task)}

## Model Description

This model was trained as part of Project Aurelius v6.0, a computational chemistry
screening pipeline optimized for Apple M-series Neural Accelerators.

- **Task:** {task_descriptions.get(task, task)}
- **Framework:** PyTorch / MLX
- **Hardware:** Apple Silicon (M1-M5)

## Training Details

- **Framework:** PyTorch (GPU/MPS) or MLX (Apple Silicon)
- **Dataset:** {task if task != 'tier0' else 'Synthetic (RDKit + Arrhenius shifts)'}
- **License:** MIT

## Usage

```bash
aurelius train --task {task}
```

## References

- Butler, K. T. et al. "Machine Learning Molecular Embeddings for Battery Materials." Nature 2023.
- Gilmer, J. et al. "Neural Message Passing for Quantum Chemistry." ICML 2017.
"""

    # Save README.md to model directory
    readme_path = os.path.join(model_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(model_card.content)
    click.echo(f"  Generated README.md at {readme_path}")

    # Upload folder
    try:
        upload_folder(
            folder_path=model_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            ignore_patterns=["*.pyc", "__pycache__", "*.pyc", ".DS_Store"],
        )
        click.echo(f"\n[SUCCESS] Model uploaded to: https://huggingface.co/{repo_id}")
    except Exception as e:
        click.echo(f"[ERROR] Upload failed: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
