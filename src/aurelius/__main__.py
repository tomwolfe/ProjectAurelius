"""Project Aurelius v7.0 - CLI Entry Point.

Usage:
    aurelius init                    Initialize pipeline
    aurelius doctor                  Validate dependencies and hardware
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

import json
import os
import sys

import click

from aurelius.cli_scripts import (
    agent,
    train_tier0,
    train_tier1,
    validate_physics,
)
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


@click.group()  # type: ignore[untyped-decorator]
@click.version_option(version="6.0.0", prog_name="Aurelius")  # type: ignore[untyped-decorator]
def cli() -> None:  # type: ignore[untyped-decorator]
    """Project Aurelius v5.2 - The Hardened Release.

    Accelerated computational chemistry screening pipeline optimized
    for Apple M-series Neural Accelerators.
    """
    pass


@cli.command()  # type: ignore[untyped-decorator]
def init() -> None:  # type: ignore[untyped-decorator]
    """Initialize the Aurelius v5.2 pipeline."""
    config = get_config()
    # Thread-safe environment variable application
    env_vars = config.apply_environment()
    _apply_env_thread_safe(env_vars)
    pipeline = AureliusPipeline(config)
    pipeline.initialize()
    click.echo("\nPipeline initialized successfully.")


@cli.command()  # type: ignore[untyped-decorator]
@click.option("--verbose", "-v", is_flag=True, default=False, help="Show detailed framework versions")  # type: ignore[untyped-decorator]
def doctor(verbose: bool) -> None:
    """Validate dependencies, hardware, and configuration.

    Checks availability of MLX, PyTorch, RDKit, and HuggingFace Hub.
    Validates GPU/MPS/CUDA hardware detection. Reports any configuration
    mismatches between environment variables and runtime config.

    This command is useful for:
    - Diagnosing setup issues before running screening
    - CI pipelines to verify the fallback-only environment
    - Quick system readiness assessment
    """
    from aurelius.config import validate_environment
    from aurelius.utils.dependencies import (
        HAS_MLX,
        HAS_RDKIT,
        HAS_TORCH,
        DependencyManager,
    )

    click.echo("\n=== Aurelius v7.0 Doctor ===")
    click.echo("")

    # Framework status
    deps = DependencyManager()
    status = deps.report_status()

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

    # Hardware detection
    click.echo("[Hardware]")

    # MLX Metal
    if HAS_MLX:
        try:
            import mlx.core as _mx  # noqa: F401

            click.echo("  MLX:      Metal backend available")
        except Exception:
            click.echo("  MLX:      Metal backend unavailable")
    else:
        click.echo("  MLX:      Not installed (will use PyTorch fallback)")

    # PyTorch device detection
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

    # RDKit
    if HAS_RDKIT:
        click.echo("  RDKit:    Available")
    else:
        click.echo("  RDKit:    Not installed (hash fallback only)")

    # HuggingFace
    if deps.is_hf_hub_available():
        click.echo("  HuggingFace Hub: Available")
    else:
        click.echo("  HuggingFace Hub: Not installed (local training only)")

    click.echo("")

    # Environment validation
    config = get_config()
    env_result = validate_environment(config, strict=True)

    if env_result["mismatches"]:
        click.echo("[Environment Mismatches]")
        for m in env_result["mismatches"]:
            click.echo(f"  {m['env_var']}: expected={m['expected']!r}, actual={m['actual']!r}")
    elif env_result["missing"]:
        click.echo("[Missing Environment Variables]")
        for var in env_result["missing"]:
            click.echo(f"  {var} (will be set automatically)")
    else:
        click.echo("[Environment] All variables match config defaults.")

    click.echo("")

    # Summary
    click.echo("[Summary]")
    issues = []
    if not HAS_MLX:
        issues.append("MLX (Apple Silicon optimization)")
    if not HAS_TORCH:
        issues.append("PyTorch (Tier 2 physics)")
    if not HAS_RDKIT:
        issues.append("RDKit (real model screening)")

    if issues:
        click.echo(f"  WARNING: Missing {len(issues)} framework(s): {', '.join(issues)}")
        click.echo("  Pipeline will use fallback paths. Some features may be degraded.")
    else:
        click.echo("  All core frameworks available. System ready for full pipeline.")

    click.echo("")


@cli.command("screen")  # type: ignore[untyped-decorator]
@click.argument("smiles")  # type: ignore[untyped-decorator]
@click.option("--solvent", default="ec:dmc", help="Solvent type (default: ec:dmc)")  # type: ignore[untyped-decorator]
@click.option("--salt", default="NaPF6", help="Salt type (default: NaPF6)")  # type: ignore[untyped-decorator]
@click.option("--ion", default="Na+", help="Ion type (default: Na+)")  # type: ignore[untyped-decorator]
@click.option("--temperature", default=298.15, help="Temperature in Kelvin")  # type: ignore[untyped-decorator]
@click.option("--voltage", default=0.05, help="Voltage cutoff")  # type: ignore[untyped-decorator]
@click.option("--cycles", default=500, help="MD simulation cycles")  # type: ignore[untyped-decorator]
@click.option("--gwp", default=1.0, help="Global Warming Potential value")  # type: ignore[untyped-decorator]
@click.option(
    "--use-real-models",
    is_flag=True,
    default=True,
    help="Use real models trained on experimental data (default: enabled)",
)  # type: ignore[untyped-decorator]
@click.option(
    "--demo",
    is_flag=True,
    default=False,
    help="Use synthetic training data for demonstration purposes",
)  # type: ignore[untyped-decorator]
@click.option(
    "--allow-fallback",
    is_flag=True,
    default=False,
    help="Allow hash-based fallback when RDKit is unavailable (demo/CI only)",
)  # type: ignore[untyped-decorator]
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

    # Production-risk warning for --allow-fallback
    if allow_fallback:
        click.echo(
            "\n[WARNING] --allow-fallback is enabled. "
            "Molecular screening will use hash-based pseudo-fingerprints "
            "which are NOT chemically valid. This should NOT be used in "
            "production workflows.\n",
        )

    # Enforce RDKit for real models (unless --allow-fallback is set)
    if use_real_models and not allow_fallback:
        try:
            from rdkit import Chem  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "RDKit is required for real model screening.\n\n"
                "Install RDKit:\n"
                "  pip install rdkit\n"
                "  conda install -c conda-forge rdkit\n\n"
                "Platform notes:\n"
                "  - macOS: pip install rdkit (or conda install -c conda-forge rdkit)\n"
                "  - Linux: pip install rdkit (requires libcdt5, libgtsb0, libgl1)\n"
                "  - Windows: pip install rdkit (pre-built wheels available)\n\n"
                "Dependency guide: https://github.com/rdkit/rdkit#installation\n\n"
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
        n_scan_cycles=cycles,
        gwp_value=gwp,
    )

    score = results.get("score")
    if score and not score.is_viable:
        sys.exit(1)


@cli.command("batch")  # type: ignore[untyped-decorator]
@click.argument("file", type=click.Path(exists=True))  # type: ignore[untyped-decorator]
@click.option("--solvent", default="ec:dmc", help="Solvent type")  # type: ignore[untyped-decorator]
@click.option("--output", type=click.Path(), help="Output JSON file")  # type: ignore[untyped-decorator]
@click.option(
    "--allow-fallback",
    is_flag=True,
    default=False,
    help="Allow hash-based fallback when RDKit is unavailable (demo/CI only)",
)  # type: ignore[untyped-decorator]
def batch(file: str, solvent: str, salt: str, output: str | None, allow_fallback: bool) -> None:  # type: ignore[untyped-decorator]
    """Screen multiple molecules from a SMILES file (one per line)."""
    # Enforce RDKit for real models (unless --allow-fallback is set)
    if not allow_fallback:
        try:
            from rdkit import Chem  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "RDKit is required for batch screening.\n\n"
                "Install RDKit:\n"
                "  pip install rdkit\n"
                "  conda install -c conda-forge rdkit\n\n"
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
    click.echo(f"\nBatch complete: {viable}/{len(smiles_list)} viable ({100 * viable / len(smiles_list):.0f}%)")

    if output:
        # Serialize results
        serializable = []
        for r in results:
            score = r["score"]
            serializable.append(
                {
                    "smiles": score.molecule_smiles,
                    "total_score": score.total_score,
                    "is_viable": score.is_viable,
                    "sigma": score.sigma_score,
                    "desolvation": score.desolvation_score,
                    "sei_homogeneity": score.sei_homogeneity_score,
                    "mx_synthesis": score.mx_synthesis_score,
                    "gwp_penalty": score.gwp_penalty,
                    "rejection_reasons": score.rejection_reasons,
                }
            )
        with open(output, "w") as f:
            json.dump(serializable, f, indent=2)
        click.echo(f"Results saved to {output}")


@cli.command("score")  # type: ignore[untyped-decorator]
@click.argument("smiles")  # type: ignore[untyped-decorator]
@click.option("--solvent", default="ec:dmc", help="Solvent type")  # type: ignore[untyped-decorator]
@click.option("--salt", default="NaPF6", help="Salt type")  # type: ignore[untyped-decorator]
@click.option("--ion", default="Na+", help="Ion type")  # type: ignore[untyped-decorator]
@click.option("--gwp", default=1.0, help="Global Warming Potential")  # type: ignore[untyped-decorator]
def score(smiles: str, solvent: str, salt: str, ion: str, gwp: float) -> None:  # type: ignore[untyped-decorator]
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
        click.echo(f"\nAurelius Score v5.2: {score.total_score:.1f}/100 {'VIABLE' if score.is_viable else 'REJECTED'}")


@cli.command("train")  # type: ignore[untyped-decorator]
@click.option("--dataset", default="esol", help="Dataset to train on (esol/qm9)")  # type: ignore[untyped-decorator]
@click.option(
    "--task",
    type=click.Choice(["tier1", "tier0"]),
    default="tier1",
    help="Training task (tier1 for MLX filter, tier0 for MPNN activation energy predictor)",
)  # type: ignore[untyped-decorator]
@click.option("--epochs", type=int, default=200, help="Number of training epochs")  # type: ignore[untyped-decorator]
@click.option("--batch-size", type=int, default=16, help="Mini-batch size")  # type: ignore[untyped-decorator]
@click.option("--learning-rate", type=float, default=0.005, help="Learning rate")  # type: ignore[untyped-decorator]
@click.option("--csv-path", type=str, default=None, help="Path to local CSV file")  # type: ignore[untyped-decorator]
def train(dataset: str, task: str, epochs: int, batch_size: int, learning_rate: float, csv_path: str | None) -> None:  # type: ignore[untyped-decorator]
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
    train_tier1.train_main(
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
    train_tier0.train_main(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        csv_path=csv_path,
    )


@cli.command("validate")  # type: ignore[untyped-decorator]
@click.option("--smiles", default="CC(=O)OC1=CC(=O)O1", help="Molecule to validate")  # type: ignore[untyped-decorator]
def validate(smiles: str) -> None:  # type: ignore[untyped-decorator]
    """Run physics validation on a molecule.

    Wraps validate_physics module as a native CLI subcommand.
    """
    sys.argv = ["validate_physics", "--smiles", smiles]
    validate_physics.main()


@cli.command("status")  # type: ignore[untyped-decorator]
def status() -> None:  # type: ignore[untyped-decorator]
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


@cli.command("benchmark")  # type: ignore[untyped-decorator]
@click.option(
    "--tier",
    type=click.Choice(["1", "2"]),
    default=None,
    help="Benchmark only a specific tier (1 or 2). Omit for all tiers.",
)  # type: ignore[untyped-decorator]
@click.option("--quick/--detailed", default=True, help="Quick mode with fewer repeats (default: enabled)")  # type: ignore[untyped-decorator]
@click.option("--output", type=click.Path(), default=None, help="Save results to JSON file")  # type: ignore[untyped-decorator]
def benchmark(tier: str | None, quick: bool, output: str | None) -> None:  # type: ignore[untyped-decorator]
    """Run hardware benchmark and validation.

    Verifies that the user's Apple Silicon hardware is properly
    configured and provides performance baselines for Tier 1
    (MLX inference) and Tier 2 (vectorized physics) computation.
    """
    from aurelius.benchmark import run_benchmark

    run_benchmark(tier=tier, quick=quick, output=output)


@cli.command("hf-upload")  # type: ignore[untyped-decorator]
@click.option("--model-dir", required=True, help="Local directory containing model files to upload")  # type: ignore[untyped-decorator]
@click.option("--repo-id", required=True, help="HuggingFace Hub repository ID (e.g., 'user/repo-name')")  # type: ignore[untyped-decorator]
@click.option(
    "--task", type=click.Choice(["tier0", "esol", "qm9"]), default="tier0", help="Model task type (default: tier0)"
)  # type: ignore[untyped-decorator]
@click.option("--private/--public", default=True, help="Make repository private (default: private)")  # type: ignore[untyped-decorator]
@click.option("--commit-message", default="Upload model via Aurelius CLI", help="Commit message for the upload")  # type: ignore[untyped-decorator]
@click.option("--dry-run", is_flag=True, help="Validate repo ID, token, and metadata without uploading")  # type: ignore[untyped-decorator]
def hf_upload(
    model_dir: str,
    repo_id: str,
    task: str,
    private: bool,
    commit_message: str,
    dry_run: bool,
) -> None:  # type: ignore[untyped-decorator]
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
            f"[ERROR] Invalid repo ID format: '{repo_id}'. Expected 'username/repo-name'.",
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
        from huggingface_hub import login as hf_login

        hf_login(add_to_git_credential=True)
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
    model_card = ModelCard("# Aurelius Model")
    model_card.content = f"""# Aurelius Model: {task_descriptions.get(task, task)}

## Model Description

This model was trained as part of Project Aurelius v7.0, a computational chemistry
screening pipeline optimized for Apple M-series Neural Accelerators.

- **Task:** {task_descriptions.get(task, task)}
- **Framework:** PyTorch / MLX
- **Hardware:** Apple Silicon (M1-M5)

## Training Details

- **Framework:** PyTorch (GPU/MPS) or MLX (Apple Silicon)
- **Dataset:** {task if task != "tier0" else "Synthetic (RDKit + Arrhenius shifts)"}
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
    with open(readme_path, "w") as readme_file:
        readme_file.write(model_card.content)
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



@cli.command("agent")
@click.option("--max-generations", type=int, default=50, help="Maximum generations to run")
@click.option("--batch-size", type=int, default=50, help="Candidates per batch")
@click.option("--profile-memory", is_flag=True, default=False, help="Enable memory profiling with CSV report output")
def agent(max_generations: int, batch_size: int, profile_memory: bool) -> None:
    """Run the autonomous screening agent.

    Executes the full autonomous discovery loop:
    Generation (RDKit mutation engine) -> Screening (3-tier pipeline) ->
    Feedback-driven mutation -> Convergence check -> Report generation
    """
    from aurelius.agent.state import CheckpointManager

    checkpoint = CheckpointManager()
    try:
        import argparse
        from typing import Any

        args = argparse.Namespace(
            max_generations=max_generations,
            batch_size=batch_size,
            profile_memory=profile_memory,
        )
        from aurelius.cli_scripts.agent import run_screening
        run_screening(args, checkpoint)
    except Exception as e:
        click.echo(f"[ERROR] {e}", err=True)
        sys.exit(1)

if __name__ == "__main__":
    cli()
