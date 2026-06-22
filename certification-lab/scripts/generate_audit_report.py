#!/usr/bin/env python3
"""Generate a Markdown audit report from a signed kernel and test data.

Usage::

    python generate_audit_report.py --kernel aurelius_kernel.json --test test_data.csv --output audit_report.md

Produces a report containing tables of Spearman ρ, MAE, coverage probability,
and a Pass/Fail status based on predefined thresholds.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from certifier.validator import UncertaintyAuditor


def load_test_data(path: str) -> list[tuple[str, float]]:
    """Load (SMILES, value) pairs from a CSV or JSON file."""
    ext = Path(path).suffix.lower()
    pairs: list[tuple[str, float]] = []

    if ext == ".csv":
        import csv

        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                smi = row.get("SMILES", "")
                val = float(row.get("value", row.get("experimental", "0")))
                if smi:
                    pairs.append((smi, val))
    elif ext == ".json":
        with open(path) as f:
            data = json.load(f)
        for entry in data:
            pairs.append((entry["SMILES"], float(entry["value"])))
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return pairs


SPEARMAN_THRESHOLD = 0.7
MAE_THRESHOLD = 0.3
COVERAGE_THRESHOLD = 0.90


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an audit report for a certified kernel."
    )
    parser.add_argument(
        "--kernel",
        required=True,
        help="Path to a signed aurelius_kernel.json file.",
    )
    parser.add_argument(
        "--test",
        required=True,
        help="Path to CSV/JSON test dataset.",
    )
    parser.add_argument(
        "--output",
        default="audit_report.md",
        help="Output path for the Markdown audit report.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.kernel):
        print(f"Error: kernel file not found: {args.kernel}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.test):
        print(f"Error: test file not found: {args.test}", file=sys.stderr)
        sys.exit(1)

    with open(args.kernel) as f:
        kernel = json.load(f)

    test_pairs = load_test_data(args.test)

    auditor = UncertaintyAuditor()
    audit = auditor.audit(kernel, test_pairs)

    metrics = kernel.get("validation_metrics", {})
    spearman = metrics.get("spearman_rho", 0.0)
    mae = metrics.get("mae", 1.0)

    spearman_pass = spearman >= SPEARMAN_THRESHOLD
    mae_pass = mae <= MAE_THRESHOLD
    coverage_pass = audit.get("pass", False)
    overall_pass = spearman_pass and mae_pass and coverage_pass

    report = f"""# Aurelius Kernel Audit Report

## Summary

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Spearman ρ | {spearman:.3f} | ≥ {SPEARMAN_THRESHOLD} | {"✅ PASS" if spearman_pass else "❌ FAIL"} |
| MAE | {mae:.3f} | ≤ {MAE_THRESHOLD} | {"✅ PASS" if mae_pass else "❌ FAIL"} |
| Coverage Probability | {audit.get('coverage_probability', 0.0):.3f} | ≥ {COVERAGE_THRESHOLD} | {"✅ PASS" if coverage_pass else "❌ FAIL"} |

**Overall: {"PASS ✅" if overall_pass else "FAIL ❌"}**

## Details

- **Kernel version**: {kernel.get('version', 'N/A')}
- **Domain**: {kernel.get('domain_boundary', {}).get('domain', 'N/A')}
- **Test samples**: {audit.get('n_samples', 0)}
- **Signature**: `{kernel.get('signature', 'N/A')[:16]}...`

## Tuned Parameters

| Parameter | Value |
|-----------|-------|
| HOMO offset | {kernel.get('tom_parameters', {}).get('homo_offset', 'N/A')} |
| LUMO offset | {kernel.get('tom_parameters', {}).get('lumo_offset', 'N/A')} |
| GC scale | {kernel.get('tom_parameters', {}).get('gc_scale', 'N/A')} |
| UQ scale | {kernel.get('tom_parameters', {}).get('uq_scale', 'N/A')} |
"""

    with open(args.output, "w") as f:
        f.write(report)

    print(f"Audit report written to {args.output}")
    print(f"  Overall status: {'PASS' if overall_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
