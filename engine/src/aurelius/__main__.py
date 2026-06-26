"""Project Aurelius v10.0 - Evolutionary Algorithm Discovery CLI.

Examples:
    aurelius init
    aurelius screen "C1COC(=O)O1"
    aurelius screen "C1COC(=O)O1" --pack organic_electronics
    aurelius batch molecules.smi --output results.json
    aurelius score "CC#N"
    aurelius view "C1COC(=O)O1"
    aurelius validate "C1COC(=O)O1"
    aurelius mixture "C1COC(=O)O1" "COCCOC" --frac 0.3
    aurelius mixture "C1COC(=O)O1" "COCCOC" --smiles-c "CC#N" --frac-a 0.4 --frac-b 0.4
    aurelius doctor --verbose
    aurelius agent --max-generations 100 --batch-size 25
    aurelius tune experiments.csv --output my_kernel.json
    aurelius verify-kernel my_kernel.json
"""

from __future__ import annotations

import json
import sys

import click

from aurelius.pipeline import AureliusPipeline, _load_demo_kernel
from aurelius.types import MoleculeContext
from aurelius.utils.dependencies import HAS_RDKIT


def _make_pipeline(pack: str = "electrolyte", demo: bool = False) -> AureliusPipeline:
    """Create and initialize a pipeline."""
    from aurelius.scoring.oracle.gc import ElectrolytePack
    from aurelius.scoring.oracle.packs import OrganicElectronicsPack
    pack_map = {
        "electrolyte": ElectrolytePack(),
        "organic_electronics": OrganicElectronicsPack(),
    }
    pipeline = AureliusPipeline(property_pack=pack_map[pack])
    pipeline.initialize()
    if demo:
        kernel = _load_demo_kernel()
        if kernel:
            _echo_colored("[bold green]✓ Demo kernel loaded.[/bold green]", style="green")
        else:
            _echo_colored("[yellow]Demo kernel not found — using default parameters.[/yellow]", style="yellow")
    return pipeline


def _get_console():
    """Return a rich Console if available, otherwise fall back to click.echo."""
    try:
        from rich.console import Console
        return Console()
    except ImportError:
        return None


_console = _get_console()


def _echo(message: str = "", **kwargs: Any) -> None:
    if _console is not None:
        _console.print(message, **kwargs)
    else:
        click.echo(message, **kwargs)


def _echo_colored(message: str, style: str = "", **kwargs: Any) -> None:
    if _console is not None:
        _console.print(message, style=style, **kwargs)
    else:
        # ANSI fallback when rich is unavailable
        _ANSI_COLORS = {
            "green": "\033[32m",
            "red": "\033[31m",
            "yellow": "\033[33m",
            "bold": "\033[1m",
            "bold green": "\033[1;32m",
            "bold red": "\033[1;31m",
            "bold yellow": "\033[1;33m",
            "cyan": "\033[36m",
        }
        ansi_reset = "\033[0m"
        clean = message
        # Strip rich markup tags
        import re as _re
        clean = _re.sub(r'\[/?\w+\]', '', clean)
        color_code = _ANSI_COLORS.get(style, "")
        if color_code:
            click.echo(f"{color_code}{clean}{ansi_reset}", **kwargs)
        else:
            click.echo(clean, **kwargs)


@click.group()
@click.version_option(version="10.0.0", prog_name="Aurelius")
def cli() -> None:
    """Project Aurelius v10.0 - Evolutionary Algorithm Discovery Release.

    Computational chemistry screening pipeline for battery electrolyte discovery.

    Hybrid quantum (xTB / TOM) + fragment-additivity (GC) oracle for
    physically valid screening of electrolyte molecules and mixtures.

    \b
    Examples:
      aurelius screen "C1COC(=O)O1"
      aurelius batch molecules.smi
      aurelius agent --max-generations 100
      aurelius doctor --verbose
    """
    pass


@cli.command()
@click.option("--pack", type=click.Choice(["electrolyte", "organic_electronics"]), default="electrolyte", help="Property pack (default: electrolyte)")
def init(pack: str) -> None:
    """Initialize the Aurelius v10.0 pipeline.

    \b
    Examples:
      aurelius init
      aurelius init --pack organic_electronics
    """
    _make_pipeline(pack=pack, demo=False)
    _echo_colored("\n[bold green]Pipeline initialized successfully.[/bold green]", style="green")


@cli.command()
@click.option("--verbose", "-v", is_flag=True, default=False, help="Show detailed framework versions")
def doctor(verbose: bool) -> None:
    """Validate dependencies, hardware, and configuration.

    \b
    Examples:
      aurelius doctor
      aurelius doctor --verbose
    """
    import platform
    from aurelius.scoring.oracle import has_xtb

    _echo_colored("[bold]Frameworks[/bold]")
    if HAS_RDKIT:
        _echo_colored("  [green]OK[/green]      rdkit")
    else:
        _echo_colored("  [red]MISSING[/red]  rdkit")
        _echo("         → conda install -c conda-forge rdkit")
        _echo("         → pip install rdkit-pypi")
        if platform.system() == "Linux":
            _echo("         → sudo apt install python3-rdkit")
        elif platform.system() == "Darwin":
            _echo("         → brew install rdkit")

    xtb_ok = has_xtb()
    if xtb_ok:
        _echo_colored("  [green]OK[/green]      xtb")
    else:
        _echo_colored("  [yellow]MISSING[/yellow]  xtb")
        _echo("         → xTB not found. Install it via: https://github.com/grimme-lab/xtb/releases or run 'brew install xtb'.")
        _echo("         → Add xtb directory to PATH after installation.")

    _echo("")

    _echo_colored("[bold]Hardware[/bold]")
    if HAS_RDKIT:
        _echo("  RDKit:    Available")
    else:
        _echo("  RDKit:    Not installed (real model screening required)")

    _echo("")

    _echo_colored("[bold]Summary[/bold]")
    if not HAS_RDKIT:
        _echo_colored("  [red]ERROR:[/red] RDKit is missing. Pipeline will not function.", style="bold red")
        _echo("  Install with:")
        _echo("    conda install -c conda-forge rdkit")
        _echo("    pip install rdkit-pypi")
    elif not xtb_ok:
        _echo_colored("  [yellow]WARNING:[/yellow] xTB not on PATH — TOM fallback active (reduced accuracy).", style="bold yellow")
    else:
        _echo_colored("  [green]All core frameworks available. System ready for full pipeline.[/green]")

    _echo("")


@cli.command("screen")
@click.argument("smiles")
@click.option("--pack", type=click.Choice(["electrolyte", "organic_electronics"]), default="electrolyte", help="Property pack (default: electrolyte)")
@click.option("--demo", is_flag=True, default=False, help="Load pre-certified demo kernel (carbonate high-voltage)")
def screen(smiles: str, pack: str, demo: bool) -> None:
    """Screen a single molecule through the full Aurelius pipeline.

    \b
    Examples:
      aurelius screen "C1COC(=O)O1"
      aurelius screen "CC#N" --pack organic_electronics
      aurelius screen "COC(=O)OC"
      aurelius screen "C1COC(=O)O1" --demo
    """
    pipeline = _make_pipeline(pack=pack, demo=demo)
    try:
        results = pipeline.screen_smiles(smiles)
    except ValueError as e:
        _echo_colored(f"[red]Invalid SMILES:[/red] {e}", style="bold red", err=True)
        sys.exit(1)

    score = results.get("score", {})
    total = score.get("total_score", 0.0)
    viable = score.get("is_viable", False)
    style = "bold green" if viable else "bold red"
    label = "DISCOVERY" if viable else "REJECTED"
    _echo_colored(f"\n[bold]Aurelius Score:[/bold] {total:.1f}/100 [{'green' if viable else 'red'}]{label}[/]", style=style)
    if score and not viable:
        sys.exit(1)


@cli.command("view")
@click.argument("smiles")
@click.option("--pack", type=click.Choice(["electrolyte", "organic_electronics"]), default="electrolyte", help="Property pack (default: electrolyte)")
@click.option("--output", type=click.Path(), help="Save HTML report to file instead of opening browser")
def view_cmd(smiles: str, pack: str, output: str | None) -> None:
    """Generate and open an HTML report for a molecule.

    Displays the molecular structure, predicted properties, and a
    Viability/REJECTED badge in the default web browser.

    \b
    Examples:
      aurelius view "C1COC(=O)O1"
      aurelius view "C1COC(=O)O1" --output report.html
    """
    from rdkit.Chem import Draw
    import tempfile
    import webbrowser
    import base64
    from io import BytesIO

    pipeline = _make_pipeline(pack=pack)
    try:
        results = pipeline.screen_smiles(smiles)
    except ValueError as e:
        _echo_colored(f"[red]Invalid SMILES:[/red] {e}", style="bold red", err=True)
        sys.exit(1)

    score = results.get("score", {})
    t2 = results.get("tier2", {})
    total = score.get("total_score", 0.0)
    viable = score.get("is_viable", False)

    # Generate molecule image
    ctx = MoleculeContext.from_smiles(smiles)
    if ctx is not None:
        img = Draw.MolToImage(ctx.mol, size=(300, 300))
        buf = BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    else:
        img_b64 = ""

    badge_color = "green" if viable else "red"
    badge_text = "DISCOVERY" if viable else "REJECTED"

    sub_scores = score.get("sub_scores", {})
    rejection_reasons = score.get("rejection_reasons", [])

    props_html = ""
    if t2:
        prop_rows = [
            ("HOMO", f'{t2.get("homo_eV", "N/A")} eV'),
            ("LUMO", f'{t2.get("lumo_eV", "N/A")} eV'),
            ("Gap", f'{t2.get("gap_eV", "N/A")} eV'),
            ("Dielectric Proxy", f'{t2.get("dielectric_proxy", "N/A")}'),
            ("Viscosity Proxy", f'{t2.get("viscosity_proxy", "N/A")}'),
            ("Li+ Solvation", f'{t2.get("li_solvation_proxy", "N/A")}'),
            ("CED Proxy", f'{t2.get("ced_proxy", "N/A")}'),
            ("SEI Fracture Toughness", f'{t2.get("sei_fracture_toughness_proxy", "N/A")}'),
            ("Quantum Confidence", t2.get("quantum_confidence", "N/A")),
            ("Domain Penalty", f'{t2.get("domain_penalty", "N/A")}'),
        ]
        for name, val in prop_rows:
            props_html += f"<tr><td style='padding:4px 12px;font-weight:600'>{name}</td><td style='padding:4px 12px'>{val}</td></tr>\n"

    subs_html = ""
    if sub_scores:
        for name, val in sorted(sub_scores.items(), key=lambda x: x[1], reverse=True):
            bar = int(val * 20)
            bar_str = "&#9608;" * bar + "&#9617;" * (20 - bar)
            subs_html += f"<tr><td style='padding:4px 12px;font-weight:600'>{name}</td><td style='padding:4px 12px'>{val:.3f}</td><td style='padding:4px 12px;font-family:monospace'>{bar_str}</td></tr>\n"

    reasons_html = ""
    if rejection_reasons:
        for r in rejection_reasons:
            reasons_html += f"<li style='color:#d32f2f;margin-bottom:4px'>{r}</li>\n"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Aurelius Report — {smiles}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 2em auto; padding: 0 1em; background: #f5f5f5; }}
        .card {{ background: #fff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); padding: 2em; margin-bottom: 1.5em; }}
        h1 {{ font-size: 1.4em; margin-top: 0; }}
        h2 {{ font-size: 1.1em; margin-top: 0; }}
        .badge {{ display: inline-block; padding: 4px 14px; border-radius: 12px; color: #fff; font-weight: 700; font-size: 0.9em; background: {badge_color}; }}
        .molecule-img {{ text-align: center; margin: 1em 0; }}
        table {{ width: 100%; border-collapse: collapse; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        td {{ padding: 4px 12px; }}
        th {{ padding: 4px 12px; text-align: left; border-bottom: 2px solid #ddd; }}
        .score {{ font-size: 2em; font-weight: 700; text-align: center; margin: 0.5em 0; }}
        ul {{ padding-left: 1.2em; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>Aurelius v10 — Molecular Report</h1>
        <p><strong>SMILES:</strong> <code>{smiles}</code></p>
        <div class="molecule-img"><img src="data:image/png;base64,{img_b64}" alt="Molecule structure"></div>
        <div class="score">
            <span>{total:.1f}/100</span>
            <span class="badge">{badge_text}</span>
        </div>
    </div>
    <div class="card">
        <h2>Predicted Properties</h2>
        <table>
            {props_html}
        </table>
    </div>"""
    if subs_html:
        html += f"""    <div class="card">
        <h2>Sub-Scores</h2>
        <table>
            <tr><th>Objective</th><th>Score</th><th>Bar</th></tr>
            {subs_html}
        </table>
    </div>"""
    if reasons_html:
        html += f"""    <div class="card">
        <h2>Rejection Reasons</h2>
        <ul>
            {reasons_html}
        </ul>
    </div>"""
    html += """</body>
</html>"""

    if output:
        with open(output, "w") as f:
            f.write(html)
        _echo_colored(f"[green]Report saved to[/green] {output}", style="green")
    else:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            f.write(html)
            html_path = f.name
        webbrowser.open(f"file://{html_path}")
        _echo_colored("[green]HTML report opened in browser.[/green]", style="green")


@cli.command("batch")
@click.argument("file", type=click.Path(exists=True))
@click.option("--output", type=click.Path(), help="Output JSON file")
@click.option("--pack", type=click.Choice(["electrolyte", "organic_electronics"]), default="electrolyte", help="Property pack (default: electrolyte)")
def batch(
    file: str,
    output: str | None,
    pack: str,
) -> None:
    """Screen multiple molecules from a SMILES file (one per line).

    \b
    Examples:
      aurelius batch molecules.smi
      aurelius batch molecules.smi --output results.json
      aurelius batch my_set.smi --pack organic_electronics
    """
    pipeline = _make_pipeline(pack=pack)
    smiles_list = []
    with open(file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                smiles_list.append(line)

    _echo(f"Screening {len(smiles_list)} molecules...")
    contexts = []
    for smi in smiles_list:
        ctx = MoleculeContext.from_smiles(smi)
        if ctx is not None:
            contexts.append(ctx)
    results = pipeline.screen_batch(contexts)

    viable = sum(1 for r in results if r["score"].get("is_viable", False) if r.get("score"))
    pct = 100 * viable / max(len(smiles_list), 1)
    color = "green" if pct > 50 else "yellow"
    _echo_colored(f"\n[bold]Batch complete:[/bold] {viable}/{len(smiles_list)} viable ({pct:.0f}%)", style=color)

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
        _echo(f"[green]Results saved to[/green] {output}")


@cli.command("score")
@click.argument("smiles")
def score(
    smiles: str,
) -> None:
    """Compute the Aurelius v10.0 score for a molecule (quick mode).

    \b
    Examples:
      aurelius score "C1COC(=O)O1"
      aurelius score "CC#N"
    """
    pipeline = _make_pipeline()
    try:
        results = pipeline.screen_smiles(smiles)
    except ValueError as e:
        _echo_colored(f"[red]Invalid SMILES:[/red] {e}", style="bold red", err=True)
        sys.exit(1)

    score = results.get("score", {})
    if score:
        total = score.get("total_score", 0.0)
        viable = score.get("is_viable", False)
        style = "bold green" if viable else "bold red"
        label = "DISCOVERY" if viable else "REJECTED"
        _echo_colored(f"\n[bold]Aurelius Score:[/bold] {total:.1f}/100 [{'green' if viable else 'red'}]{label}[/]", style=style)


@cli.command("evaluate")
@click.option("--smiles", default="CC(=O)OC1=CC(=O)O1", help="Molecule to evaluate")
def evaluate_cmd(
    smiles: str = "CC(=O)OC1=CC(=O)O1",
) -> None:
    """Run ML Oracle evaluation on a molecule.

    \b
    Examples:
      aurelius evaluate --smiles "C1COC(=O)O1"
      aurelius evaluate
    """
    pipeline = _make_pipeline()
    try:
        results = pipeline.screen_smiles(smiles)
        score = results.get("score", {})
        total = score.get("total_score", 0.0)
        viable = score.get("is_viable", False)
        style = "bold green" if viable else "bold red"
        label = "DISCOVERY" if viable else "REJECTED"
        _echo_colored(f"\n[bold]Aurelius Score:[/bold] {total:.1f}/100 [{'green' if viable else 'red'}]{label}[/]", style=style)
    except ValueError as e:
        _echo_colored(f"[red]Invalid SMILES:[/red] {e}", style="bold red", err=True)
        sys.exit(1)
    except Exception as e:
        _echo_colored(f"[red]Evaluation failed:[/red] {e}", style="bold red", err=True)
        sys.exit(1)


@cli.command("validate")
@click.argument("smiles")
@click.option("--pretty", is_flag=True, default=False, help="Print ASCII report card summary")
def validate_cmd(smiles: str, pretty: bool) -> None:
    """Run the full pipeline on a SMILES and print a report card.

    \b
    Examples:
      aurelius validate "C1COC(=O)O1"
      aurelius validate "C1COC(=O)O1" --pretty
      aurelius validate "CC#N"
    """
    from aurelius.pipeline import _OBJECTIVES

    pipeline = _make_pipeline()
    try:
        results = pipeline.screen_smiles(smiles)
    except ValueError as e:
        _echo_colored(f"[red]Invalid SMILES:[/red] {e}", style="bold red", err=True)
        sys.exit(1)
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

    _echo(f"\n{'=' * 56}")
    _echo_colored("  [bold]Project Aurelius v10.0 — Validate[/bold]")
    _echo(f"  SMILES: {smiles}")
    _echo(f"{'=' * 56}")

    all_passed = True
    for obj in _OBJECTIVES:
        raw = t2.get(obj.property_key, 0.0)
        if obj.property_key == "sa_score":
            raw = score.get("sa_score", 0.0)
        sub = sub_scores.get(obj.name, 0.0)
        weighted = obj.weight * sub
        label = obj.name[:28]
        if sub >= 0.7:
            icon = "[green]✓[/green]"
        elif sub >= 0.4:
            icon = "[yellow]~[/yellow]"
        else:
            icon = "[red]✗[/red]"
            all_passed = False
        eng = reason_keywords.get(obj.name.split("_")[0], ("", ""))[0] if sub < 0.4 else ""
        eng_note = f" — {eng}" if eng else ""
        _echo(f"  {icon} {label:<26} raw={raw:>7.3f}  w={obj.weight:.2f}  sub={weighted:.4f}{eng_note}")

    _echo(f"  {'-' * 56}")
    verdict = "[green]✓[/green]" if viable else "[red]✗[/red]"
    style = "bold green" if viable else "bold red"
    label = "DISCOVERY" if viable else "REJECTED"
    _echo_colored(f"  {verdict} TOTAL: {total:>7.1f}/100  {label}", style=style)
    if score.get("rejection_reasons"):
        for reason in score["rejection_reasons"]:
            _echo_colored(f"     [red]✗[/red] {reason}")
    if t2:
        _echo(f"\n  {'=' * 56}")
        _echo_colored("  [bold]Predicted Properties:[/bold]")
        _echo(f"    HOMO:               {t2.get('homo_eV', 'N/A')} eV")
        _echo(f"    LUMO:               {t2.get('lumo_eV', 'N/A')} eV")
        _echo(f"    Gap:                {t2.get('gap_eV', 'N/A')} eV")
        _echo(f"    Dielectric proxy:   {t2.get('dielectric_proxy', 'N/A')}")
        _echo(f"    Viscosity proxy:    {t2.get('viscosity_proxy', 'N/A')}")
        _echo(f"    Li+ solvation:      {t2.get('li_solvation_proxy', 'N/A')}")
    _echo(f"{'=' * 56}")

    if pretty:
        if _console is not None:
            from rich.table import Table
            from rich import box

            label = "DISCOVERY" if viable else "REJECTED"
            table = Table(title=f"Score: {total:.1f}/100 — {label}", box=box.SIMPLE)
            table.add_column("Objective", style="cyan")
            table.add_column("Raw", justify="right")
            table.add_column("Score", justify="right")
            table.add_column("Bar", style="green", no_wrap=True)
            for name, val in sorted(sub_scores.items(), key=lambda x: x[1], reverse=True):
                bar_len = 20
                filled = int(val * bar_len)
                bar = "█" * filled + "░" * (bar_len - filled)
                raw_val = t2.get(name[:name.rfind("_")], "")
                table.add_row(name[:26], f"{raw_val if raw_val else '':>7}", f"{val:.3f}", bar)
            _console.print(table)
        else:
            bar_len = 20
            label = "DISCOVERY" if viable else "REJECTED"
            _echo(f"\n  {'─' * 40}")
            _echo(f"  Score: {total:5.1f}/100 {label}")
            if sub_scores:
                best = max(sub_scores.values())
                best_name = max(sub_scores, key=sub_scores.get)
                worst = min(sub_scores.values())
                worst_name = min(sub_scores, key=sub_scores.get)
                _echo(f"  Best:  {best_name:<22} {best:.3f}")
                _echo(f"  Worst: {worst_name:<22} {worst:.3f}")
                for name, val in sorted(sub_scores.items(), key=lambda x: x[1], reverse=True):
                    filled = int(val * bar_len)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    _echo(f"    {name:<22} {bar} {val:.2f}")
            _echo(f"  {'─' * 40}")

    if not viable:
        sys.exit(1)


@cli.command("agent")
@click.option("--max-generations", type=int, default=50, help="Maximum generations to run")
@click.option("--batch-size", type=int, default=50, help="Candidates per batch")
@click.option("--pack", type=click.Choice(["electrolyte", "organic_electronics"]), default="electrolyte", help="Property pack (default: electrolyte)")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Show per-generation score histogram")
@click.option("--strict", is_flag=True, default=False, help="Raise on SMARTS/BRICS failures instead of logging warnings")
def agent(
    max_generations: int,
    batch_size: int,
    pack: str,
    verbose: bool,
    strict: bool,
) -> None:
    """Evolve electrolyte candidates via BRICS mutation + multi-objective screening.

    \b
    Examples:
      aurelius agent
      aurelius agent --max-generations 100 --batch-size 25

    \b
    Tip: Run 'aurelius doctor' first to verify dependencies.
    """
    from aurelius.agent.loop import AgentConfig, run_screening

    cfg = AgentConfig(max_generations=max_generations, batch_size=batch_size, pack=pack, verbose=verbose, strict=strict)
    run_screening(cfg)


def _validate_component_smiles(smiles_a: str, smiles_b: str) -> tuple[MoleculeContext, MoleculeContext]:
    ctx_a = MoleculeContext.from_smiles(smiles_a)
    ctx_b = MoleculeContext.from_smiles(smiles_b)
    if ctx_a is None:
        _echo_colored(f"[red]Invalid SMILES:[/red] could not parse '{smiles_a}'", style="bold red", err=True)
        sys.exit(1)
    if ctx_b is None:
        _echo_colored(f"[red]Invalid SMILES:[/red] could not parse '{smiles_b}'", style="bold red", err=True)
        sys.exit(1)
    return ctx_a, ctx_b


def _screen_ternary(pipeline: AureliusPipeline, ctx_a: MoleculeContext, ctx_b: MoleculeContext, smiles_c: str, frac_a: float | None, frac_b: float | None) -> tuple[dict, str]:
    if frac_a is None or frac_b is None:
        _echo_colored("[red]Error:[/red] --frac-a and --frac-b required for ternary mixtures", style="bold red", err=True)
        sys.exit(1)
    if not (0.0 < frac_a < 1.0 and 0.0 < frac_b < 1.0 and frac_a + frac_b < 1.0):
        _echo_colored("[red]Error:[/red] frac_a and frac_b must be in (0,1) and sum < 1.0", style="bold red", err=True)
        sys.exit(1)
    ctx_c = MoleculeContext.from_smiles(smiles_c)
    if ctx_c is None:
        _echo_colored("[red]Error:[/red] Invalid SMILES for third component.", style="bold red", err=True)
        sys.exit(1)
    result = pipeline.screen_mixture(ctx_a, ctx_b, frac_a, ctx3=ctx_c, frac2=frac_b)
    return result, "Ternary Mixture"


def _screen_binary(pipeline: AureliusPipeline, ctx_a: MoleculeContext, ctx_b: MoleculeContext, frac: float) -> tuple[dict, str]:
    if not (0.0 <= frac <= 1.0):
        _echo_colored("[red]Error:[/red] --frac must be between 0.0 and 1.0", style="bold red", err=True)
        sys.exit(1)
    result = pipeline.screen_mixture(ctx_a, ctx_b, frac)
    return result, "Binary Mixture"


def _report_mixture_result(result: dict, label: str) -> None:
    score = result.get("score", {})
    mix_props = result.get("mixture_properties", {})
    total = score.get("total_score", 0.0)
    viable = score.get("is_viable", False)
    style = "bold green" if viable else "bold red"
    verdict_label = "DISCOVERY" if viable else "REJECTED"
    _echo_colored(f"\n[bold]{label} Aurelius Score:[/bold] {total:.1f}/100 [{'green' if viable else 'red'}]{verdict_label}[/]", style=style)
    _echo(f"  Synergy Bonus: {mix_props.get('synergy_bonus', 0.0):.4f}")
    _echo(f"  Dielectric Proxy: {mix_props.get('dielectric_proxy', 0.0):.2f}")
    _echo(f"  Viscosity Proxy:  {mix_props.get('viscosity_proxy', 0.0):.2f}")
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

    Thermodynamic mixing rules with synergy bonus for complementary pairs
    (high-dielectric + low-viscosity).

    \b
    Binary:
      aurelius mixture "C1COC(=O)O1" "COCCOC"
      aurelius mixture "C1COC(=O)O1" "COCCOC" --frac 0.3

    \b
    Ternary:
      aurelius mixture "C1COC(=O)O1" "COCCOC" --smiles-c "CC#N" --frac-a 0.4 --frac-b 0.4
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
            _echo(f"  Skipping {smi}: {exc}", err=True)
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

    \b
    Examples:
      aurelius tune experiments.csv
      aurelius tune experiments.csv --output my_kernel.json --max-iter 500
    """
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        _echo_colored("[red]Error:[/red] 'aurelius tune' requires numpy. Install with: pip install -e '.[ml]'", style="bold red", err=True)
        sys.exit(1)

    _echo(f"Loading training data from {csv_path}...")
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
    _echo(f"Loaded {n} data points. Running Nelder-Mead optimization...")

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

    _echo_colored(f"\n[bold green]Tuning complete.[/bold green] Results written to [cyan]{output}[/cyan]", style="green")
    _echo(f"  HOMO offset:  {tom_parameters['homo_offset']:+.4f}")
    _echo(f"  LUMO offset:  {tom_parameters['lumo_offset']:+.4f}")
    _echo(f"  GC scale:     {tom_parameters['gc_scale']:.4f}")
    _echo(f"  Spearman ρ:   {spearman:.4f}")
    _echo(f"  MAE:          {mae:.4f}")
    _echo(f"  RMSE:         {rmse:.4f}")
    _echo(f"  Training pts: {len(valid_smiles)}")


@cli.command("verify-kernel")
@click.argument("kernel_path", type=click.Path(exists=True))
def verify_kernel_cmd(kernel_path: str) -> None:
    """Verify a kernel's Ed25519 signature using the public key in constants.

    KERNEL_PATH is the path to a signed aurelius_kernel.json file.

    \b
    Examples:
      aurelius verify-kernel aurelius_kernel.json
      aurelius verify-kernel ~/lab/signed_kernel.json
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        from aurelius.constants import KERNEL_PUBLIC_KEY

        with open(kernel_path) as f:
            kernel = json.load(f)

        stored = kernel.get("signature", "")
        if not stored:
            _echo_colored("[red]FAIL:[/red] No signature field found in kernel.", style="bold red", err=True)
            sys.exit(1)

        payload = {k: v for k, v in kernel.items() if k != "signature"}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))

        pub = Ed25519PublicKey.from_public_bytes(KERNEL_PUBLIC_KEY)
        pub.verify(bytes.fromhex(stored), canonical.encode("utf-8"))
        _echo_colored("[bold green]OK:[/bold green] Kernel signature verified successfully.", style="green")
    except json.JSONDecodeError:
        _echo_colored(f"[red]FAIL:[/red] Could not parse {kernel_path} as JSON.", style="bold red", err=True)
        sys.exit(1)
    except Exception as exc:
        _echo_colored(f"[red]FAIL:[/red] Signature verification failed: {exc}", style="bold red", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
