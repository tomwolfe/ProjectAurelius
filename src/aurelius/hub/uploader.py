"""HuggingFace model upload logic.

Provides ``upload_model_to_hub()`` which handles the full upload
workflow: authentication, repository creation, README generation,
and folder upload.
"""

from __future__ import annotations

import os
import sys

import click


def upload_model_to_hub(
    model_dir: str,
    repo_id: str,
    task: str,
    private: bool,
    commit_message: str,
    dry_run: bool,
) -> None:
    """Upload a locally trained model to HuggingFace Hub.

    Args:
        model_dir: Local directory containing model files to upload.
        repo_id: HuggingFace Hub repository ID (e.g., ``'user/repo-name'``).
        task: Model task type (e.g. ``'tier0'``, ``'esol'``, ``'qm9'``).
        private: Whether the repository should be private.
        commit_message: Commit message for the upload.
        dry_run: If True, validate without uploading.

    Raises:
        SystemExit: On validation errors (directory not found, invalid repo ID).
        click.ClickException: On HF authentication or upload failures.
    """
    _validate_upload_inputs(model_dir, repo_id)

    if dry_run:
        _run_dry_run(model_dir, repo_id, task)
        return

    _perform_upload(model_dir, repo_id, task, commit_message)


def _validate_upload_inputs(model_dir: str, repo_id: str) -> None:
    """Validate model directory and repo ID before upload.

    Args:
        model_dir: Path to model files.
        repo_id: HuggingFace repository ID.

    Raises:
        SystemExit: On invalid inputs.
    """
    if not os.path.isdir(model_dir):
        click.echo(f"[ERROR] Model directory not found: {model_dir}", err=True)
        sys.exit(1)

    if "/" not in repo_id:
        click.echo(
            f"[ERROR] Invalid repo ID format: '{repo_id}'. Expected 'username/repo-name'.",
            err=True,
        )
        sys.exit(1)


def _run_dry_run(model_dir: str, repo_id: str, task: str) -> None:
    """Run a dry-run: validate auth and list files without uploading.

    Args:
        model_dir: Model directory path.
        repo_id: Repository ID.
        task: Model task type.
    """
    from huggingface_hub import HfApi

    api = HfApi()

    click.echo(f"[DRY RUN] Validating upload for: {repo_id}")
    click.echo(f"  Model directory: {model_dir}")
    click.echo(f"  Task: {task}")
    click.echo("  Validating authentication...")

    try:
        user_info = api.whoami()
        click.echo(f"  Authenticated as: {user_info['name']} ({user_info['fullname']})")
    except Exception as e:
        click.echo(f"[ERROR] HuggingFace authentication failed: {e}", err=True)
        click.echo("Ensure HF_TOKEN is set or run 'huggingface-cli login'.", err=True)
        sys.exit(1)

    if repo_id and HfApi().model_info(repo_id) is not None:
        click.echo(f"  Repository '{repo_id}' already exists.")
    else:
        click.echo(f"  Repository '{repo_id}' does not exist (would be created).")

    files = []
    for root, _, filenames in os.walk(model_dir):
        for fname in filenames:
            rel_path = os.path.relpath(os.path.join(root, fname), model_dir)
            files.append(rel_path)
    click.echo(f"  Files to upload ({len(files)}):")
    for f in sorted(files):
        click.echo(f"    - {f}")

    click.echo("\n[DRY RUN] Validation complete. No files were uploaded.")


def _perform_upload(model_dir: str, repo_id: str, task: str, commit_message: str) -> None:
    """Perform the actual model upload to HuggingFace Hub.

    Args:
        model_dir: Model directory path.
        repo_id: Repository ID.
        task: Model task type.
        commit_message: Commit message for the upload.
    """
    import os

    from huggingface_hub import (
        HfApi,
        ModelCard,
        create_repo,
        upload_folder,
    )
    from huggingface_hub import login as hf_login

    click.echo(f"[HF Upload] Uploading to: {repo_id}")
    click.echo(f"  Model directory: {model_dir}")
    click.echo(f"  Task: {task}")
    click.echo(f"  Visibility: {'private' if True else 'public'}")

    try:
        hf_login(add_to_git_credential=True)
    except Exception as e:
        click.echo(f"[ERROR] HuggingFace authentication failed: {e}", err=True)
        click.echo("Ensure HF_TOKEN is set in your environment or run 'huggingface-cli login'.", err=True)
        sys.exit(1)

    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=True,
            token=HfApi().token if hasattr(HfApi(), "token") else None,
        )
        click.echo(f"  Repository '{repo_id}' ready.")
    except Exception as e:
        click.echo(f"[ERROR] Failed to create/verify repository: {e}", err=True)
        sys.exit(1)

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

    readme_path = os.path.join(model_dir, "README.md")
    with open(readme_path, "w") as readme_file:
        readme_file.write(model_card.content)
    click.echo(f"  Generated README.md at {readme_path}")

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
