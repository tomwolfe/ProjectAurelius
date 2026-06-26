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
    aurelius mixture <smi_a> <smi_b> [--frac]  Screen a binary electrolyte mixture
"""

from __future__ import annotations

import json
import sys

import click

from aurelius.pipeline import AureliusPipeline
from aurelius.types import MoleculeContext
from aurelius.utils.dependencies import HAS_RDKIT


def _make_pipeline(pack: str = "electrolyte") -> AureliusPipeline:
    """Create and initialize a pipeline."""
    from aurelius.scoring.oracle.gc import ElectrolytePack
    from aurelius.scoring.oracle.packs import OrganicElectronicsPack
    pack_map = {
        "electrolyte": ElectrolytePack(),
        "organic_electronics": OrganicElectronicsPack(),
    }
    pipeline = AureliusPipeline(property_pack=pack_map[pack])
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
    import platform
    from aurelius.scoring.oracle import has_xtb

    click.echo("[Frameworks]")
    icon = "OK" if HAS_RDKIT else "MISSING"
    click.echo(f"  [{icon:>7}] rdkit")
    if not HAS_RDKIT:
        _ = platform.system()
        click.echo("         → conda install -c conda-forge rdkit")

    xtb_ok = has_xtb()
    click.echo(f"  [{'OK' if xtb_ok else 'MISSING':>7}] xtb")
    if not xtb_ok:
        click.echo("         → https://github.com/grimme-lab/xtb/releases")
        click.echo("         → Add xtb directory to PATH")

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
    elif not xtb_ok:
        click.echo("  WARNING: xTB not on PATH — TOM fallback active (reduced accuracy).")
    else:
        click.echo("  All core frameworks available. System ready for full pipeline.")

    click.echo("")


@cli.command("screen")
@click.argument("smiles")
@click.option("--pack", type=click.Choice(["electrolyte", "organic_electronics"]), default="electrolyte", help="Property pack (default: electrolyte)")
def screen(smiles: str, pack: str) -> None:
    """Screen a single molecule through the full Aurelius pipeline."""
    pipeline = _make_pipeline(pack=pack)
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
@click.option("--pack", type=click.Choice(["electrolyte", "organic_electronics"]), default="electrolyte", help="Property pack (default: electrolyte)")
def batch(
    file: str,
    output: str | None,
    pack: str,
) -> None:
    """Screen multiple molecules from a SMILES file (one per line)."""
    pipeline = _make_pipeline(pack=pack)
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
@click.option("--pretty", is_flag=True, default=False, help="Print ASCII report card summary")
def validate_cmd(smiles: str, pretty: bool) -> None:
    """Run the full pipeline on a SMILES and print a report card."""
    from aurelius.pipeline import _OBJECTIVES

    pipeline = _make_pipeline()
    results = pipeline.screen_smiles(smiles)
    score = results.get("score", {})
    t2 = results.get("tier2", {})

    total = score.get("total_score", 0.0)
    viable = score.get("is_viable", False)
    sub_scores = score.get("sub_scores", {})
    reason_keywords: dict[str, tuple[str, str]] = {
        "lumo": ("LUMO outside SEI window", "LUMO"),
        "homo": ("HOMO too high — oxidative instability", "HOMO"),
        "dielectric": ("Low dielectric — poor salt dissolution", "Dielectric"),
        "viscosity": ("High viscosity — poor ion mobility", "Viscosity"),
        "li_solvation": ("Li+ binding outside ideal range", "Li Solvation"),
        "ced": ("Low CED — weak SEI cohesion", "CED"),
        "sa": ("Hard to synthesise", "SA Score"),
        "gas_evolution": ("High gas evolution risk", "Gas Evo"),
        "hydrolysis": ("High hydrolysis risk", "Hydrolysis"),
    }

    click.echo(f"\n{'=' * 56}")
    click.echo("  Project Aurelius v10.0 — Validate")
    click.echo(f"  SMILES: {smiles}")
    click.echo(f"{'=' * 56}")

    all_passed = True
    for obj in _OBJECTIVES:
        raw = t2.get(obj.property_key, 0.0)
        if obj.property_key == "sa_score":
            raw = score.get("sa_score", 0.0)
        sub = sub_scores.get(obj.name, 0.0)
        weighted = obj.weight * sub
        label = obj.name[:28]
        if sub >= 0.7:
            icon = "✅"
        elif sub >= 0.4:
            icon = "⚠️"
        else:
            icon = "❌"
            all_passed = False
        eng = reason_keywords.get(obj.name.split("_")[0], ("", ""))[0] if sub < 0.4 else ""
        eng_note = f" — {eng}" if eng else ""
        click.echo(f"  {icon} {label:<26} raw={raw:>7.3f}  w={obj.weight:.2f}  sub={weighted:.4f}{eng_note}")

    click.echo(f"  {'-' * 56}")
    verdict_icon = "✅" if viable else "❌"
    click.echo(f"  {verdict_icon} TOTAL: {total:>7.1f}/100  {'VIABLE' if viable else 'REJECTED'}")
    if score.get("rejection_reasons"):
        for reason in score["rejection_reasons"]:
            click.echo(f"     ❌ {reason}")
    if t2:
        click.echo(f"\n  {'=' * 56}")
        click.echo("  Predicted Properties:")
        click.echo(f"    HOMO:               {t2.get('homo_eV', 'N/A')} eV")
        click.echo(f"    LUMO:               {t2.get('lumo_eV', 'N/A')} eV")
        click.echo(f"    Gap:                {t2.get('gap_eV', 'N/A')} eV")
        click.echo(f"    Dielectric proxy:   {t2.get('dielectric_proxy', 'N/A')}")
        click.echo(f"    Viscosity proxy:    {t2.get('viscosity_proxy', 'N/A')}")
        click.echo(f"    Li+ solvation:      {t2.get('li_solvation_proxy', 'N/A')}")
    click.echo(f"{'=' * 56}")

    if pretty:
        bar_len = 20
        click.echo(f"\n  {'─' * 40}")
        click.echo(f"  Score: {total:5.1f}/100 {'✓' if viable else '✗'}")
        if sub_scores:
            best = max(sub_scores.values())
            best_name = max(sub_scores, key=sub_scores.get)
            worst = min(sub_scores.values())
            worst_name = min(sub_scores, key=sub_scores.get)
            click.echo(f"  Best:  {best_name:<22} {best:.3f}")
            click.echo(f"  Worst: {worst_name:<22} {worst:.3f}")
            for name, val in sorted(sub_scores.items(), key=lambda x: x[1], reverse=True):
                filled = int(val * bar_len)
                bar = "█" * filled + "░" * (bar_len - filled)
                click.echo(f"    {name:<22} {bar} {val:.2f}")
        click.echo(f"  {'─' * 40}")

    if not viable:
        sys.exit(1)


@cli.command("agent")
@click.option("--max-generations", type=int, default=50, help="Maximum generations to run")
@click.option("--batch-size", type=int, default=50, help="Candidates per batch")
@click.option("--pack", type=click.Choice(["electrolyte", "organic_electronics"]), default="electrolyte", help="Property pack (default: electrolyte)")
def agent(
    max_generations: int,
    batch_size: int,
    pack: str,
) -> None:
    """Run the autonomous screening agent."""
    from aurelius.agent.loop import AgentConfig, run_screening

    cfg = AgentConfig(max_generations=max_generations, batch_size=batch_size, pack=pack)
    run_screening(cfg)


def _validate_component_smiles(smiles_a: str, smiles_b: str) -> tuple[MoleculeContext, MoleculeContext]:
    ctx_a = MoleculeContext.from_smiles(smiles_a)
    ctx_b = MoleculeContext.from_smiles(smiles_b)
    if ctx_a is None or ctx_b is None:
        click.echo("Error: Invalid SMILES provided.", err=True)
        sys.exit(1)
    return ctx_a, ctx_b


def _screen_ternary(pipeline: AureliusPipeline, ctx_a: MoleculeContext, ctx_b: MoleculeContext, smiles_c: str, frac_a: float | None, frac_b: float | None) -> tuple[dict, str]:
    if frac_a is None or frac_b is None:
        click.echo("Error: --frac-a and --frac-b required for ternary mixtures", err=True)
        sys.exit(1)
    if not (0.0 < frac_a < 1.0 and 0.0 < frac_b < 1.0 and frac_a + frac_b < 1.0):
        click.echo("Error: frac_a and frac_b must be in (0,1) and sum < 1.0", err=True)
        sys.exit(1)
    ctx_c = MoleculeContext.from_smiles(smiles_c)
    if ctx_c is None:
        click.echo("Error: Invalid SMILES for third component.", err=True)
        sys.exit(1)
    result = pipeline.screen_mixture(ctx_a, ctx_b, frac_a, ctx3=ctx_c, frac2=frac_b)
    return result, "Ternary Mixture"


def _screen_binary(pipeline: AureliusPipeline, ctx_a: MoleculeContext, ctx_b: MoleculeContext, frac: float) -> tuple[dict, str]:
    if not (0.0 <= frac <= 1.0):
        click.echo("Error: --frac must be between 0.0 and 1.0", err=True)
        sys.exit(1)
    result = pipeline.screen_mixture(ctx_a, ctx_b, frac)
    return result, "Binary Mixture"


def _report_mixture_result(result: dict, label: str) -> None:
    score = result.get("score", {})
    mix_props = result.get("mixture_properties", {})
    click.echo(f"\n{label} Aurelius Score: {score.get('total_score', 0.0):.1f}/100 {'VIABLE' if score.get('is_viable', False) else 'REJECTED'}")
    click.echo(f"  Synergy Bonus: {mix_props.get('synergy_bonus', 0.0):.4f}")
    click.echo(f"  Dielectric Proxy: {mix_props.get('dielectric_proxy', 0.0):.2f}")
    click.echo(f"  Viscosity Proxy:  {mix_props.get('viscosity_proxy', 0.0):.2f}")
    if not score.get("is_viable", False):
        sys.exit(1)


@cli.command("mixture")
@click.argument("smiles_a")
@click.argument("smiles_b")
@click.option("--frac", type=float, default=0.5, help="Volume fraction of component A (0.0 to 1.0)")
@click.option("--smiles-c", type=str, default=None, help="Third component SMILES (ternary mixture)")
@click.option("--frac-a", type=float, default=None, help="Volume fraction of component A (ternary)")
@click.option("--frac-b", type=float, default=None, help="Volume fraction of component B (ternary)")
def mixture_cmd(smiles_a: str, smiles_b: str, frac: float, smiles_c: str | None, frac_a: float | None, frac_b: float | None) -> None:
    """Screen a binary or ternary electrolyte mixture.

    Binary: SMILES_A SMILES_B [--frac FRAC]
    Ternary: SMILES_A SMILES_B --smiles-c SMILES_C --frac-a FRAC_A --frac-b FRAC_B
    """
    pipeline = _make_pipeline()
    ctx_a, ctx_b = _validate_component_smiles(smiles_a, smiles_b)

    if smiles_c is not None:
        result, label = _screen_ternary(pipeline, ctx_a, ctx_b, smiles_c, frac_a, frac_b)
    else:
        result, label = _screen_binary(pipeline, ctx_a, ctx_b, frac)

    _report_mixture_result(result, label)


# ---------------------------------------------------------------------------
# Local Lightweight Kernel Tuner (no Stripe, JWT, or Postgres)
# ---------------------------------------------------------------------------


def _spearman_rho(x: list[float], y: list[float]) -> float:
    try:
        import numpy as np
    except ImportError:
        return 0.0
    n = len(x)
    if n < 3:
        return 0.0
    x_arr, y_arr = np.array(x), np.array(y)
    x_rank = np.argsort(np.argsort(x_arr)).astype(np.float64)
    y_rank = np.argsort(np.argsort(y_arr)).astype(np.float64)
    d = x_rank - y_rank
    rho = 1.0 - (6.0 * np.sum(d ** 2)) / (n * (n ** 2 - 1.0))
    return 0.0 if np.isnan(rho) else float(rho)


def _adjust_prediction(
    oracle_result: dict,
    property_name: str,
    homo_offset: float,
    lumo_offset: float,
    gc_scale: float,
) -> float:
    raw = oracle_result.get(property_name, 0.0)
    if not raw:
        raw = oracle_result.get("homo_eV", 0.0)
    if property_name in ("homo", "homo_eV"):
        return raw + homo_offset
    if property_name in ("lumo", "lumo_eV"):
        return raw + lumo_offset
    return raw * gc_scale


def _tune_objective(
    params: list[float],
    smiles_list: list[str],
    property_names: list[str],
    exp_values: list[float],
) -> float:
    import numpy as np
    from aurelius.scoring.oracle import PropertyOracle

    oracle = PropertyOracle(use_xtb=False, use_surrogate=False, use_gc_uq=False)
    predictions: list[float] = []
    for smi, prop in zip(smiles_list, property_names):
        try:
            result = oracle.evaluate_smiles(smi)
            pred = _adjust_prediction(
                result, prop,
                homo_offset=params[0],
                lumo_offset=params[1],
                gc_scale=params[2],
            )
            predictions.append(pred)
        except Exception as exc:
            click.echo(f"  Skipping {smi}: {exc}", err=True)
            continue
    if len(predictions) < 2:
        return 999.0
    pred_arr = np.array(predictions, dtype=np.float64)
    exp_arr = np.array(exp_values[:len(predictions)], dtype=np.float64)
    return float(np.sqrt(np.mean((pred_arr - exp_arr) ** 2)))


def _nelder_mead(
    f,
    x0: list[float],
    args: tuple = (),
    bounds: list[tuple[float, float]] | None = None,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> list[float]:
    import numpy as np

    n = len(x0)

    def clamp(x: np.ndarray) -> np.ndarray:
        if bounds is None:
            return x
        return np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])

    simplex = [clamp(np.array(x0, dtype=np.float64))]
    for i in range(n):
        p = np.array(x0, dtype=np.float64)
        p[i] = x0[i] + 0.5 if abs(x0[i]) < 0.01 else x0[i] * 1.05
        simplex.append(clamp(p))

    f_vals = [float(f(simplex[i], *args)) for i in range(n + 1)]

    for _ in range(max_iter):
        indices = np.argsort(f_vals)
        simplex = [simplex[i] for i in indices]
        f_vals = [f_vals[i] for i in indices]

        if np.std(f_vals) < tol:
            break

        centroid = np.mean(simplex[:n], axis=0)
        xr = clamp(centroid + 1.0 * (centroid - simplex[-1]))
        fr = float(f(xr, *args))

        if f_vals[0] <= fr < f_vals[-2]:
            simplex[-1] = xr
            f_vals[-1] = fr
        elif fr < f_vals[0]:
            xe = clamp(centroid + 2.0 * (xr - centroid))
            fe = float(f(xe, *args))
            if fe < fr:
                simplex[-1] = xe
                f_vals[-1] = fe
            else:
                simplex[-1] = xr
                f_vals[-1] = fr
        else:
            if fr < f_vals[-1]:
                xc = clamp(centroid + 0.5 * (xr - centroid))
                fc = float(f(xc, *args))
                if fc < fr:
                    simplex[-1] = xc
                    f_vals[-1] = fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = clamp(simplex[0] + 0.5 * (simplex[i] - simplex[0]))
                        f_vals[i] = float(f(simplex[i], *args))
            else:
                xc = clamp(centroid + 0.5 * (centroid - simplex[-1]))
                fc = float(f(xc, *args))
                if fc < f_vals[-1]:
                    simplex[-1] = xc
                    f_vals[-1] = fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = clamp(simplex[0] + 0.5 * (simplex[i] - simplex[0]))
                        f_vals[i] = float(f(simplex[i], *args))

    best_idx = int(np.argmin(f_vals))
    return simplex[best_idx].tolist()


@cli.command("tune")
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("--output", default="aurelius_kernel.json", help="Output kernel file path")
@click.option("--max-iter", default=200, help="Maximum Nelder-Mead iterations")
def tune_cmd(csv_path: str, output: str, max_iter: int) -> None:
    """Tune kernel parameters from a CSV of experimental data.

    CSV must have columns: smiles, property, value

    Runs a local Nelder-Mead optimizer (no Stripe/JWT/Postgres required)
    and writes the tuned kernel to --output (default: aurelius_kernel.json).
    """
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        click.echo("Error: 'aurelius tune' requires numpy. Install with: pip install -e '.[ml]'", err=True)
        sys.exit(1)

    click.echo(f"Loading training data from {csv_path}...")
    import csv
    smiles_list: list[str] = []
    property_names: list[str] = []
    values: list[float] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles_list.append(row["smiles"].strip())
            prop = row.get("property", "homo").strip()
            property_names.append(prop)
            values.append(float(row["value"]))

    n = len(smiles_list)
    if n < 3:
        click.echo(f"Error: need at least 3 data points, got {n}", err=True)
        sys.exit(1)
    click.echo(f"Loaded {n} data points. Running Nelder-Mead optimization...")

    x0 = [0.0, 0.0, 1.0]
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (0.1, 10.0)]

    result_x = _nelder_mead(
        _tune_objective,
        x0,
        args=(smiles_list, property_names, values),
        bounds=bounds,
        max_iter=max_iter,
    )

    tom_parameters = {
        "homo_offset": float(result_x[0]),
        "lumo_offset": float(result_x[1]),
        "gc_scale": float(result_x[2]),
    }

    # Compute validation metrics
    from aurelius.scoring.oracle import PropertyOracle
    oracle = PropertyOracle(use_xtb=False, use_surrogate=False, use_gc_uq=False)
    predictions: list[float] = []
    valid_smiles: list[str] = []
    for smi, prop in zip(smiles_list, property_names):
        try:
            result = oracle.evaluate_smiles(smi)
            pred = _adjust_prediction(
                result, prop,
                homo_offset=tom_parameters["homo_offset"],
                lumo_offset=tom_parameters["lumo_offset"],
                gc_scale=tom_parameters["gc_scale"],
            )
            predictions.append(pred)
            valid_smiles.append(smi)
        except Exception:
            continue

    if len(predictions) >= 2:
        residuals = [p - v for p, v in zip(predictions, values[:len(predictions)])]
        mae = sum(abs(r) for r in residuals) / len(residuals)
        rmse = (sum(r * r for r in residuals) / len(residuals)) ** 0.5
        spearman = _spearman_rho(predictions, values[:len(predictions)])
    else:
        mae, rmse, spearman = 0.0, 0.0, 0.0

    kernel = {
        "version": "1.0.0",
        "domain_boundary": {"domain": "electrolyte"},
        "tom_parameters": tom_parameters,
        "gc_fragments": [
            "ester", "carboxylic_acid", "amide", "ketone", "aldehyde",
            "carbonate", "ether", "alcohol", "primary_amine", "secondary_amine",
            "tertiary_amine", "nitrile", "alkene", "alkyne", "aromatic_carbon",
            "fluorine", "chlorine", "bromine", "sulfone", "sulfonate",
            "sulfonyl_fluoride", "cyclic_sulfone_5", "cyclic_sulfone_6",
            "sultone_5", "sultone_6", "phosphate", "trifluoromethyl",
            "difluoromethylene", "boronate", "borate", "thioether",
            "fluorinated_ether", "phosphazene", "glyme_chelating", "sulfonimide",
            "fluorinated_carbonate", "sulfoxide", "aromatic_nitrogen",
            "phosphonate", "hf_scavenger", "cyclic_carbonate",
        ],
        "uq_weights": {"ensemble_weight": 0.5},
        "validation_metrics": {
            "spearman_rho": round(spearman, 4),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "n_training": len(valid_smiles),
        },
    }

    with open(output, "w") as f:
        json.dump(kernel, f, indent=2)

    click.echo(f"\nTuning complete. Results written to {output}")
    click.echo(f"  HOMO offset:  {tom_parameters['homo_offset']:+.4f}")
    click.echo(f"  LUMO offset:  {tom_parameters['lumo_offset']:+.4f}")
    click.echo(f"  GC scale:     {tom_parameters['gc_scale']:.4f}")
    click.echo(f"  Spearman ρ:   {spearman:.4f}")
    click.echo(f"  MAE:          {mae:.4f}")
    click.echo(f"  RMSE:         {rmse:.4f}")
    click.echo(f"  Training pts: {len(valid_smiles)}")


if __name__ == "__main__":
    cli()
