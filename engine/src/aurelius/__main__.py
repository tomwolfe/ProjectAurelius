"""Project Aurelius v11.0 — Discovery CLI.

Examples:
    aurelius screen "C1COC(=O)O1"
    aurelius screen "C1COC(=O)O1" --verbose
    aurelius screen "C1COC(=O)O1" --json
    aurelius screen "C1COC(=O)O1" --report
    aurelius screen "C1COC(=O)O1" --pack organic_electronics
    aurelius screen molecules.smi --output results.json
    aurelius batch molecules.smi --output results.json
    aurelius mixture "C1COC(=O)O1" "COCCOC" --frac 0.3
    aurelius doctor --verbose
    aurelius agent --max-generations 100 --batch-size 25
"""

from __future__ import annotations

import json
import sys
from typing import Any

import click

from aurelius.kernel_loader import _load_demo_kernel
from aurelius.pipeline import AureliusPipeline
from aurelius.types import MoleculeContext
from aurelius.utils.dependencies import HAS_RDKIT


def _make_pipeline(pack: str = "electrolyte", demo: bool = False, solvent: str | None = "ether") -> AureliusPipeline:
    """Create and initialize a pipeline."""
    from aurelius.scoring.oracle.gc import ElectrolytePack
    from aurelius.scoring.oracle.packs import OrganicElectronicsPack
    pack_map = {
        "electrolyte": ElectrolytePack(),
        "organic_electronics": OrganicElectronicsPack(),
    }
    pipeline = AureliusPipeline(property_pack=pack_map[pack], solvent=solvent)
    pipeline.initialize()
    if demo:
        kernel = _load_demo_kernel()
        if kernel:
            _echo_colored("[bold green]Demo kernel loaded.[/bold green]", style="green")
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
        kwargs.pop("err", None)
        _console.print(message, **kwargs)
    else:
        click.echo(message, **kwargs)


def _echo_colored(message: str, style: str = "", **kwargs: Any) -> None:
    if _console is not None:
        kwargs.pop("err", None)
        _console.print(message, style=style, **kwargs)
    else:
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
        import re as _re
        clean = _re.sub(r'\[/?\w+\]', '', message)
        color_code = _ANSI_COLORS.get(style, "")
        if color_code:
            click.echo(f"{color_code}{clean}{ansi_reset}", **kwargs)
        else:
            click.echo(clean, **kwargs)


def print_result_card(
    smiles: str,
    total_score: float,
    is_viable: bool,
    sub_scores: dict[str, float],
    properties: dict[str, Any] | None = None,
    rejection_reasons: list[str] | None = None,
) -> None:
    """Display a summary card of the screening result using rich if available.

    Falls back to clean ASCII formatting when rich is not installed.
    """
    label = "DISCOVERY" if is_viable else "REJECTED"
    style = "bold green" if is_viable else "bold red"

    if _console is not None:
        _print_result_card_rich(
            _console, smiles, total_score, is_viable, label, style,
            sub_scores, properties, rejection_reasons,
        )
    else:
        _print_result_card_ascii(
            smiles, total_score, is_viable, label,
            sub_scores, properties, rejection_reasons,
        )


def _print_result_card_rich(
    console: Any,
    smiles: str,
    total_score: float,
    is_viable: bool,
    label: str,
    style: str,
    sub_scores: dict[str, float],
    properties: dict[str, Any] | None = None,
    rejection_reasons: list[str] | None = None,
) -> None:
    """Rich-formatted result card."""
    from rich.table import Table
    from rich import box
    from rich.panel import Panel

    score_color = "green" if is_viable else "red"

    score_text = f"[bold {score_color}]{total_score:.1f}/100 — {label}[/bold {score_color}]"
    console.print(Panel(score_text, title="[bold]Aurelius Screen[/bold]", width=60))
    console.print(f"  SMILES: [cyan]{smiles}[/cyan]")

    if properties:
        prop_table = Table(title="Predicted Properties", box=box.SIMPLE, width=60)
        prop_table.add_column("Property", style="cyan")
        prop_table.add_column("Value", justify="right")
        for key, val in properties.items():
            if isinstance(val, float):
                prop_table.add_row(key, f"{val:.4f}")
            else:
                prop_table.add_row(key, str(val))
        console.print(prop_table)

    if sub_scores:
        sub_table = Table(title="Sub-Scores", box=box.SIMPLE, width=60)
        sub_table.add_column("Objective", style="cyan")
        sub_table.add_column("Score", justify="right")
        sub_table.add_column("Bar", style="green", no_wrap=True)
        for name, val in sorted(sub_scores.items(), key=lambda x: x[1], reverse=True):
            bar_len = 20
            filled = int(val * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            sub_table.add_row(name[:28], f"{val:.4f}", bar)
        console.print(sub_table)

    if rejection_reasons:
        console.print("[bold red]Rejection Reasons:[/bold red]")
        for r in rejection_reasons:
            console.print(f"  [red]✗[/red] {r}")


def _print_result_card_ascii(
    smiles: str,
    total_score: float,
    is_viable: bool,
    label: str,
    sub_scores: dict[str, float],
    properties: dict[str, Any] | None = None,
    rejection_reasons: list[str] | None = None,
) -> None:
    """ASCII-fallback result card."""
    bar_len = 30
    _echo(f"\n{'=' * 60}")
    _echo(f"  Project Aurelius v10.0 — {label}")
    _echo(f"  SMILES: {smiles}")
    _echo(f"  Score:  {total_score:5.1f}/100")
    _echo(f"{'=' * 60}")

    if properties:
        _echo(f"\n  Predicted Properties:")
        for key, val in properties.items():
            if isinstance(val, float):
                _echo(f"    {key:<30} {val:.4f}")
            else:
                _echo(f"    {key:<30} {val}")
        _echo(f"{'-' * 60}")

    if sub_scores:
        _echo(f"\n  Sub-Scores:")
        for name, val in sorted(sub_scores.items(), key=lambda x: x[1], reverse=True):
            filled = int(val * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            _echo(f"    {name:<28} {val:.4f}  {bar}")
        _echo(f"{'-' * 60}")

    if rejection_reasons:
        _echo(f"\n  Rejection Reasons:")
        for r in rejection_reasons:
            _echo(f"    ✗ {r}")

    _echo(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version="10.0.0", prog_name="Aurelius")
def cli() -> None:
    """Project Aurelius v11.0 - Physics-Grounded Discovery Engine."""
    pass


@cli.command()
@click.option("--pack", type=click.Choice(["electrolyte", "organic_electronics"]), default="electrolyte")
@click.option("--solvent", default="ether", help="Implicit solvation model for xTB (e.g. ether, carbonate, acetonitrile)")
def init(pack: str, solvent: str) -> None:
    """Initialize the Aurelius v10.0 pipeline."""
    _make_pipeline(pack=pack, demo=False, solvent=solvent)
    _echo_colored("\n[bold green]Pipeline initialized successfully.[/bold green]", style="green")


@cli.command()
@click.option("--verbose", "-v", is_flag=True, default=False)
def doctor(verbose: bool) -> None:
    """Validate dependencies, hardware, and configuration."""
    import platform
    from aurelius.scoring.oracle import has_xtb

    _echo_colored("[bold]Frameworks[/bold]")
    if HAS_RDKIT:
        _echo_colored("  [green]OK[/green]      rdkit")
    else:
        _echo_colored("  [red]MISSING[/red]  rdkit")
        _echo("         \u2192 conda install -c conda-forge rdkit")
        _echo("         \u2192 pip install rdkit-pypi")
        if platform.system() == "Linux":
            _echo("         \u2192 sudo apt install python3-rdkit")
        elif platform.system() == "Darwin":
            _echo("         \u2192 brew install rdkit")

    xtb_ok = has_xtb()
    if xtb_ok:
        _echo_colored("  [green]OK[/green]      xtb")
        from aurelius.utils.dependencies import check_xtb_with_benchmark
        bench_msg = check_xtb_with_benchmark()
        if bench_msg:
            is_good = "Expected" in bench_msg
            _echo_colored(f"  {bench_msg}", style="green" if is_good else "yellow")
        else:
            _echo_colored("  xTB Active but slow/unstable.", style="yellow")
    else:
        _echo_colored("  [yellow]MISSING[/yellow]  xtb")
        if platform.system() == "Linux":
            _echo("         \u2192 sudo apt install xtb")
            _echo("         \u2192 pip install xtb")
        elif platform.system() == "Darwin":
            _echo("         \u2192 brew install xtb")
        else:
            _echo("         \u2192 conda install -c conda-forge xtb")
        _echo("         \u2192 Check common paths: /usr/local/bin, ~/.local/bin")
        _echo("         \u2192 Or download from: https://github.com/grimme-lab/xtb/releases")

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


def _print_chemist_summary(smiles: str, t2: dict[str, Any] | None) -> None:
    """Print a 'Chemist's Summary' with GC fragment contributions to low scores."""
    ctx = MoleculeContext.from_smiles(smiles)
    if ctx is None:
        return
    try:
        from aurelius.scoring.oracle.gc import _GC_FRAGMENTS, _count_fragments
        counts = _count_fragments(ctx.mol)
    except Exception:
        return

    _echo("")
    _echo_colored("[bold]Chemist's Summary — GC Fragment Analysis:[/bold]")

    if t2 is None:
        _echo("  [yellow]Tier 1 filter rejected — no GC properties computed.[/yellow]")
        return

    dielectric = t2.get("dielectric_proxy", 0.0) or 0.0
    viscosity = t2.get("viscosity_proxy", 0.0) or 0.0
    uncertainty = t2.get("uncertainty_score", 0.0) or 0.0

    if dielectric < 5.0:
        missing_high = []
        for _pat, name, dd, _dv, _ls, _dc in _GC_FRAGMENTS:
            if dd > 2.0 and counts.get(name, 0) == 0:
                missing_high.append(name)
        _echo(f"  [yellow]Low Dielectric ({dielectric:.2f}):[/yellow] Missing high-contribution fragments: {', '.join(missing_high[:5]) or 'none identified'}")
        if "cyclic_carbonate" in counts and counts["cyclic_carbonate"] > 0:
            _echo("    [green]Cyclic carbonate present — good for dielectric.[/green]")

    if viscosity > 2.5:
        high_visc = []
        for _pat, name, _dd, dv, _ls, _dc in _GC_FRAGMENTS:
            if dv > 0.3 and counts.get(name, 0) > 0:
                high_visc.append(f"{name} (×{counts[name]})")
        _echo(f"  [yellow]High Viscosity ({viscosity:.2f}):[/yellow] Contributing fragments: {', '.join(high_visc[:5]) or 'none identified'}")

    if uncertainty > 0.15:
        _echo(f"  [yellow]High Prediction Uncertainty ({uncertainty:.3f}):[/yellow] Molecule may be out-of-distribution for GC model.")

    _echo(f"  [cyan]GC Fragment Counts:[/cyan] {dict(sorted(counts.items()))}")


def _run_screen(
    pipeline: AureliusPipeline,
    smiles: str,
    verbose: bool = False,
    json_output: bool = False,
) -> dict[str, Any] | None:
    """Screen a single molecule and optionally print results."""
    try:
        results = pipeline.screen_smiles(smiles)
    except ValueError as e:
        _echo_colored(f"[red]Invalid SMILES:[/red] {e}", style="bold red", err=True)
        return None

    score = results.get("score", {})
    t2 = results.get("tier2", {})
    total = score.get("total_score", 0.0)
    viable = score.get("is_viable", False)
    sub_scores = score.get("sub_scores", {})
    rejection_reasons = score.get("rejection_reasons", [])

    if json_output:
        return {
            "smiles": smiles,
            "total_score": total,
            "is_viable": viable,
            "sub_scores": sub_scores,
            "rejection_reasons": rejection_reasons,
            "properties": t2,
        }

    properties = None
    if t2:
        properties = {
            "HOMO (eV)": t2.get("homo_eV"),
            "LUMO (eV)": t2.get("lumo_eV"),
            "Gap (eV)": t2.get("gap_eV"),
            "Dielectric": t2.get("dielectric_proxy"),
            "Viscosity": t2.get("viscosity_proxy"),
            "Li+ Solvation": t2.get("li_solvation_proxy"),
            "Quantum Confidence": t2.get("quantum_confidence"),
            "Domain Applicable": t2.get("domain_applicable"),
        }

    print_result_card(
        smiles=smiles,
        total_score=total,
        is_viable=viable,
        sub_scores=sub_scores,
        properties=properties,
        rejection_reasons=rejection_reasons,
    )

    if verbose and sub_scores:
        _echo("")
        _echo_colored("[bold]Detailed Sub-Scores:[/bold]")
        for name, val in sorted(sub_scores.items(), key=lambda x: x[1], reverse=True):
            _echo(f"  {name:<28} {val:.4f}")

    if verbose and rejection_reasons:
        _echo("")
        _echo_colored("[bold]Rejection Reasons:[/bold]")
        for r in rejection_reasons:
            _echo(f"  [red]✗[/red] {r}")

    if verbose:
        _print_chemist_summary(smiles, t2)

    return None


@cli.command("screen")
@click.argument("smiles", required=False)
@click.option("--pack", type=click.Choice(["electrolyte", "organic_electronics"]), default="electrolyte")
@click.option("--demo", is_flag=True, default=False)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Show sub-scores and rejection reasons")
@click.option("--json", "json_output", is_flag=True, default=False, help="Output raw JSON")
@click.option("--report", is_flag=True, default=False, help="Generate and open HTML report")
@click.option("--output", type=click.Path(), help="Save JSON/HTML output to file")
@click.option("--solvent", default="ether", help="Implicit solvation model for xTB (e.g. ether, carbonate, acetonitrile)")
def screen(
    smiles: str | None,
    pack: str,
    demo: bool,
    verbose: bool,
    json_output: bool,
    report: bool,
    output: str | None,
    solvent: str,
) -> None:
    """Screen a molecule or file of molecules through the full Aurelius pipeline.

    Provide a single SMILES string or a path to a file (one SMILES per line).
    """
    pipeline = _make_pipeline(pack=pack, demo=demo, solvent=solvent)

    if smiles is None and output is None:
        _echo_colored("[red]Error:[/red] SMILES argument or file input required.", style="bold red", err=True)
        sys.exit(1)

    # File input: one SMILES per line
    if smiles is not None and _is_file(smiles):
        _screen_file(pipeline, smiles, verbose, json_output, report, output)
        return

    # Single SMILES
    if smiles is None:
        _echo_colored("[red]Error:[/red] SMILES argument required.", style="bold red", err=True)
        sys.exit(1)

    result = _run_screen(pipeline, smiles, verbose, json_output)
    if result is None:
        sys.exit(1)

    if json_output:
        _emit_json(result, output)
        return

    if report:
        _generate_report(pipeline, smiles, output)
        return

    score = result
    if not score["is_viable"]:
        sys.exit(1)


def _is_file(path: str) -> bool:
    """Check if a path looks like a file (exists and not a SMILES string)."""
    import os
    if not os.path.exists(path):
        return False
    if len(path) > 200:
        return False
    # Check if it looks like a SMILES (contains typical chemistry chars)
    smiles_chars = set("CCOcNnOoSsPpFfClBrIHh#()[]=@1234567890\\/")
    path_chars = set(path)
    # If it's a short string with mostly SMILES characters, treat as SMILES
    if len(path) < 100 and path_chars.issubset(smiles_chars):
        return False
    return os.path.isfile(path)


def _screen_file(
    pipeline: AureliusPipeline,
    file_path: str,
    verbose: bool,
    json_output: bool,
    report: bool,
    output: str | None,
) -> None:
    """Screen multiple molecules from a SMILES file."""
    smiles_list: list[str] = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                smiles_list.append(line)

    if not smiles_list:
        _echo_colored("[red]Error:[/red] No SMILES found in file.", style="bold red", err=True)
        sys.exit(1)

    _echo(f"Screening {len(smiles_list)} molecules...")

    results: list[dict[str, Any]] = []
    for smi in smiles_list:
        result = _run_screen(pipeline, smi, verbose, json_output=True)
        if result is not None:
            results.append(result)

    if json_output or output:
        _emit_json(results, output)
        return

    viable = sum(1 for r in results if r["is_viable"])
    pct = 100 * viable / max(len(results), 1)
    color = "green" if pct > 50 else "yellow"
    _echo_colored(f"\n[bold]Batch complete:[/bold] {viable}/{len(results)} viable ({pct:.0f}%)", style=color)


def _emit_json(data: Any, output: str | None) -> None:
    """Print or save JSON output."""
    text = json.dumps(data, indent=2)
    if output:
        with open(output, "w") as f:
            f.write(text)
        _echo(f"[green]Results saved to[/green] {output}")
    else:
        click.echo(text)


def _generate_report(pipeline: AureliusPipeline, smiles: str, output: str | None) -> None:
    """Generate an HTML report for a single molecule."""
    try:
        from rdkit.Chem import Draw
    except ImportError:
        _echo_colored("[red]Error:[/red] RDKit Draw module required for HTML report generation.", style="bold red", err=True)
        sys.exit(1)
    import tempfile
    import webbrowser
    import base64
    from io import BytesIO

    try:
        results = pipeline.screen_smiles(smiles)
    except ValueError as e:
        _echo_colored(f"[red]Invalid SMILES:[/red] {e}", style="bold red", err=True)
        sys.exit(1)

    score = results.get("score", {})
    t2 = results.get("tier2", {})
    total = score.get("total_score", 0.0)
    viable = score.get("is_viable", False)

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
        try:
            webbrowser.open(f"file://{html_path}")
        except Exception:
            _echo(f"Report saved to [cyan]{html_path}[/cyan] (could not open browser)")
        else:
            _echo_colored("[green]HTML report opened in browser.[/green]", style="green")


@cli.command("batch")
@click.argument("file", type=click.Path(exists=True))
@click.option("--output", type=click.Path())
@click.option("--pack", type=click.Choice(["electrolyte", "organic_electronics"]), default="electrolyte")
@click.option("--solvent", default="ether", help="Implicit solvation model for xTB (e.g. ether, carbonate, acetonitrile)")
def batch(file: str, output: str | None, pack: str, solvent: str) -> None:
    """Screen multiple molecules from a SMILES file (one per line)."""
    pipeline = _make_pipeline(pack=pack, solvent=solvent)
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


@cli.command("agent")
@click.option("--max-generations", type=int, default=50)
@click.option("--batch-size", type=int, default=50)
@click.option("--pack", type=click.Choice(["electrolyte", "organic_electronics"]), default="electrolyte")
@click.option("--verbose", "-v", is_flag=True, default=False)
@click.option("--strict", is_flag=True, default=False)
def agent(max_generations: int, batch_size: int, pack: str, verbose: bool, strict: bool) -> None:
    """Evolve electrolyte candidates via BRICS mutation + multi-objective screening."""
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
@click.option("--frac", type=float, default=0.5)
@click.option("--smiles-c", type=str, default=None)
@click.option("--frac-a", type=float, default=None)
@click.option("--frac-b", type=float, default=None)
@click.option("--solvent", default="ether", help="Implicit solvation model for xTB (e.g. ether, carbonate, acetonitrile)")
def mixture_cmd(smiles_a: str, smiles_b: str, frac: float, smiles_c: str | None, frac_a: float | None, frac_b: float | None, solvent: str) -> None:
    """Screen a binary or ternary electrolyte mixture."""
    pipeline = _make_pipeline(solvent=solvent)
    ctx_a, ctx_b = _validate_component_smiles(smiles_a, smiles_b)

    if smiles_c is not None:
        result, label = _screen_ternary(pipeline, ctx_a, ctx_b, smiles_c, frac_a, frac_b)
    else:
        result, label = _screen_binary(pipeline, ctx_a, ctx_b, frac)

    _report_mixture_result(result, label)


@cli.command("tune")
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("--output", default="aurelius_kernel.json")
@click.option("--max-iter", default=200)
def tune_cmd(csv_path: str, output: str, max_iter: int) -> None:
    """Tune kernel parameters from a CSV of experimental data.

    The CSV must contain at least three columns: ``smiles``, ``property``, and ``value``
    (or two columns: ``smiles`` and ``value`` with ``homo`` assumed for the property).
    """
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        _echo_colored("[red]Error:[/red] 'aurelius tune' requires numpy. Install with: pip install -e '.[ml]'", style="bold red", err=True)
        sys.exit(1)

    _echo(f"Loading training data from {csv_path}...")
    import csv
    training_pairs: list[tuple[str, ...]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            training_pairs.append((
                row["smiles"].strip(),
                row.get("property", "homo").strip(),
                row["value"],
            ))

    n = len(training_pairs)
    if n < 3:
        click.echo(f"Error: need at least 3 data points, got {n}", err=True)
        sys.exit(1)
    _echo(f"Loaded {n} data points. Running Nelder-Mead optimization...")

    from aurelius.scoring.oracle import KernelOptimizer

    optimizer = KernelOptimizer(max_iter=max_iter)
    kernel = optimizer.optimize(training_pairs)

    with open(output, "w") as f:
        json.dump(kernel, f, indent=2)

    _echo_colored(f"[bold green]Tuning complete.[/bold green] Results written to [cyan]{output}[/cyan]", style="green")
    tom = kernel.get("tom_parameters", {})
    metrics = kernel.get("validation_metrics", {})
    _echo(f"  HOMO offset:  {tom.get('homo_offset', 0.0):+.4f}")
    _echo(f"  LUMO offset:  {tom.get('lumo_offset', 0.0):+.4f}")
    _echo(f"  GC scale:     {tom.get('gc_scale', 0.0):.4f}")
    _echo(f"  Spearman ρ:   {metrics.get('spearman_rho', 0.0):.4f}")
    _echo(f"  MAE:          {metrics.get('mae', 0.0):.4f}")
    _echo(f"  RMSE:         {metrics.get('rmse', 0.0):.4f}")
    _echo(f"  Training pts: {metrics.get('n_training', 0)}")


@cli.command("learn")
@click.option("--max-suggestions", type=int, default=10, help="Number of top discoveries to suggest")
@click.option("--feedback-csv", type=click.Path(exists=True), help="Path to CSV with experimental feedback")
@click.option("--feedback-sdf", type=click.Path(exists=True), help="Path to SDF with experimental feedback")
@click.option("--output", type=click.Path(), help="Output path for suggestions file")
@click.option("--verbose", "-v", is_flag=True, default=False)
def learn_cmd(max_suggestions: int, feedback_csv: str | None, feedback_sdf: str | None, output: str | None, verbose: bool) -> None:
    """Run the active-learning suggest → validate → retrain cycle.

    If --feedback-csv or --feedback-sdf is provided, the pipeline will
    retrain the GC UQ ensemble with the experimental data.  Otherwise,
    the command generates top-N synthesis suggestions from the current
    discovery set.
    """
    from aurelius.agent.learning_loop import SuggestAndValidatePipeline, AutoRetrainPipeline, ExperimentResultParser

    if feedback_csv is not None:
        try:
            feedback_data = ExperimentResultParser.parse_csv(feedback_csv)
        except (FileNotFoundError, ValueError) as e:
            _echo_colored(f"[red]Error parsing CSV:[/red] {e}", style="bold red", err=True)
            sys.exit(1)

        auto_retrain = AutoRetrainPipeline()
        auto_retrain._feedback_data = feedback_data

        _echo(f"Loaded {len(feedback_data)} feedback entries")
        summary = auto_retrain.summary()
        _echo(f"  Mean dielectric: {summary.get('mean_dielectric', 0.0):.4f}")
        _echo(f"  Mean viscosity:  {summary.get('mean_viscosity', 0.0):.4f}")

        if verbose:
            for entry in feedback_data:
                _echo(f"  SMILES: {entry.get('smiles', '')}")
                _echo(f"    dielectric: {entry.get('dielectric_constant', 0.0):.4f}")
                _echo(f"    viscosity:  {entry.get('viscosity_cP', 0.0):.4f}")
                _echo(f"    cycle_life: {entry.get('cycle_life', 0.0):.4f}")

        _echo_colored("[bold green]Feedback pipeline ready.[/bold green]")

    elif feedback_sdf is not None:
        try:
            feedback_data = ExperimentResultParser.parse_sdf(feedback_sdf)
        except (FileNotFoundError, ValueError) as e:
            _echo_colored(f"[red]Error parsing SDF:[/red] {e}", style="bold red", err=True)
            sys.exit(1)

        auto_retrain = AutoRetrainPipeline()
        auto_retrain._feedback_data = feedback_data

        _echo(f"Loaded {len(feedback_data)} feedback entries")
        summary = auto_retrain.summary()
        _echo(f"  Mean dielectric: {summary.get('mean_dielectric', 0.0):.4f}")
        _echo(f"  Mean viscosity:  {summary.get('mean_viscosity', 0.0):.4f}")

        _echo_colored("[bold green]Feedback pipeline ready.[/bold green]")

    else:
        try:
            from aurelius.agent.state import LoopState
            state = LoopState()
        except Exception as e:
            _echo_colored(f"[red]Error loading loop state:[/red] {e}", style="bold red", err=True)
            sys.exit(1)

        discoveries = state.discoveries[:max_suggestions] if state.discoveries else []
        suggestions = SuggestAndValidatePipeline(discoveries)

        output_path = output or "suggestions.sdf"
        suggestions.export(output_path)

        _echo_colored(f"[bold green]Generated {len(discoveries)} synthesis suggestions → {output_path}[/bold green]")


@cli.command("verify-kernel")
@click.argument("kernel_path", type=click.Path(exists=True))
def verify_kernel_cmd(kernel_path: str) -> None:
    """Verify a kernel JSON file contains all required fields."""
    from aurelius.kernel_loader import JSONKernelLoader

    loader = JSONKernelLoader()
    with open(kernel_path) as f:
        kernel = json.load(f)

    if loader.verify(kernel):
        _echo_colored("[bold green]OK:[/bold green] Kernel validation passed.", style="green")
    else:
        _echo_colored("[red]FAIL:[/red] Kernel validation failed — required fields missing.", style="bold red", err=True)
        sys.exit(1)


@cli.command("dashboard")
@click.option("--port", type=int, default=8501, help="Port for Streamlit server")
def dashboard_cmd(port: int) -> None:
    """Launch the Aurelius Discovery Dashboard (Streamlit app).

    Opens an interactive web-based visualization of:
    - Discovery trajectories
    - Chemical space (UMAP embeddings)
    - Pareto front (3D interactive plot)
    - Molecule viewer with property annotations
    """
    try:
        import streamlit as st
    except ImportError:
        _echo_colored("[red]Error:[/red] 'streamlit' is required for dashboard. Install with: pip install streamlit", style="bold red", err=True)
        sys.exit(1)

    _echo_colored("[bold green]Starting Aurelius Dashboard...[/bold green]")
    _echo(f"Dashboard will be available at http://localhost:{port}")

    import subprocess
    import sys
    import os

    # Get the directory of this file to find the dashboard module
    dashboard_dir = Path(__file__).parent / "dashboard"
    env = os.environ.copy()
    env["STREAMLIT_GLOBALLY_QUIT"] = "0"

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(dashboard_dir / "app.py"), "--server.port", str(port)],
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        _echo_colored(f"[red]Error launching dashboard:[/red] {e}", style="bold red", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
