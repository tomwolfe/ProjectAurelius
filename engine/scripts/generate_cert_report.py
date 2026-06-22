#!/usr/bin/env python3
"""Generate a Markdown certification report from a signed AureliusKernel
and independent test data.

Usage:
    python scripts/generate_cert_report.py \\
        --kernel aurelius_kernel.json \\
        --test-data test_data.csv
"""

from __future__ import annotations

import csv
import json
import sys

import click
import numpy as np
import scipy.stats

from aurelius.scoring.oracle.gc import (
    predict_dielectric_proxy,
    predict_viscosity_proxy,
)
from aurelius.types import MoleculeContext
from aurelius.utils.certification import verify_kernel_signature


def _load_test_data(path: str) -> tuple[list[MoleculeContext], list[float], list[float]]:
    mols: list[MoleculeContext] = []
    exp_diele: list[float] = []
    exp_visc: list[float] = []

    if path.endswith(".json"):
        with open(path) as f:
            records = json.load(f)
        for rec in records:
            ctx = MoleculeContext.from_smiles(str(rec.get("smiles", "")))
            if ctx is None:
                continue
            mols.append(ctx)
            exp_diele.append(float(rec.get("dielectric_constant", rec.get("Exp_Dielectric", 0.0))))
            exp_visc.append(float(rec.get("viscosity_cP", rec.get("Exp_Viscosity", 0.0))))
    else:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ctx = MoleculeContext.from_smiles(row.get("SMILES", ""))
                if ctx is None:
                    continue
                mols.append(ctx)
                exp_diele.append(float(row.get("Exp_Dielectric", 0.0)))
                exp_visc.append(float(row.get("Exp_Viscosity", 0.0)))

    return mols, exp_diele, exp_visc


@click.command()
@click.option("--kernel", required=True, help="Path to signed kernel JSON.")
@click.option(
    "--test-data",
    required=True,
    help="Path to CSV/JSON with SMILES, Exp_Dielectric, Exp_Viscosity.",
)
@click.option("--secret", default=None, help="Signing key for full signature verification.")
@click.option("--output", default=None, help="Output path for the Markdown report.")
def main(kernel: str, test_data: str, secret: str | None, output: str | None) -> None:
    """Generate a Markdown certification report."""
    with open(kernel) as f:
        kernel_dict = json.load(f)

    domain_name = kernel_dict.get("domain_name", "Unknown")
    gc_fragments = kernel_dict.get("gc_fragments", {})

    mols, exp_diele, exp_visc = _load_test_data(test_data)
    if len(mols) < 2:
        click.echo("Need at least 2 molecules with valid SMILES.", err=True)
        sys.exit(1)

    pred_diele: list[float] = []
    pred_visc: list[float] = []
    for ctx in mols:
        d = predict_dielectric_proxy(ctx, fragment_overrides=gc_fragments)
        v = predict_viscosity_proxy(ctx, fragment_overrides=gc_fragments)
        pred_diele.append(d)
        pred_visc.append(v)

    if len(set(pred_diele)) > 1:
        spearman_rho = scipy.stats.spearmanr(pred_diele, exp_diele)[0]
    else:
        spearman_rho = 0.0

    mae_visc = float(np.mean(np.abs(np.array(pred_visc) - np.array(exp_visc))))
    uq_coverage = kernel_dict.get("validation_metrics", {}).get("uq_coverage_pct", 100.0)

    # --- PASS / FAIL logic ---
    rho_pass = spearman_rho > 0.7
    mae_threshold = kernel_dict.get("validation_metrics", {}).get("mae_viscosity", 5.0) * 2.0
    mae_pass = mae_visc < mae_threshold if mae_threshold > 0 else True
    overall_pass = rho_pass and mae_pass

    # --- Signature verification ---
    if secret:
        try:
            sig_valid = verify_kernel_signature(kernel_dict, secret)
            sig_status = "Valid" if sig_valid else "Invalid (mismatch)"
        except ValueError:
            sig_status = "Invalid (no signature field)"
    else:
        has_sig = bool(kernel_dict.get("signature"))
        sig_status = "Present (full verification requires --secret)" if has_sig else "Missing"

    report_lines = [
        f"# Certification Report for {domain_name}",
        "",
        "## Validation Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Spearman ρ (Dielectric) | {spearman_rho:.4f} |",
        f"| MAE Viscosity | {mae_visc:.4f} |",
        f"| UQ Coverage (%) | {uq_coverage:.2f} |",
        "",
        "## Status",
        "",
        f"**{'PASS' if overall_pass else 'FAIL'}**",
        "",
        "| Condition | Threshold | Result |",
        "|---|---|---|",
        f"| Spearman ρ > 0.7 | > 0.7 | {'PASS' if rho_pass else 'FAIL'} |",
        f"| MAE < {mae_threshold:.2f} | < {mae_threshold:.2f} | {'PASS' if mae_pass else 'FAIL'} |",
        "",
        "## Signature",
        "",
        f"**{sig_status}**",
        "",
    ]

    report = "\n".join(report_lines)

    if output:
        with open(output, "w") as f:
            f.write(report)
        click.echo(f"Report written to {output}")
    else:
        click.echo(report)


if __name__ == "__main__":
    main()
