#!/usr/bin/env python3
"""CLI entry point for local kernel certification.

Usage::

    python certify_kernel.py --data training.csv --domain "carbonate" --output aurelius_kernel.json

Reads experimental data from a CSV or JSON file, runs the KernelOptimizer to
tune TOM/GC parameters, validates the result with the UncertaintyAuditor on a
held-out split, signs the kernel with HMAC-SHA256, and writes
``aurelius_kernel.json`` to disk.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from certifier.optimizer import KernelOptimizer
from certifier.validator import UncertaintyAuditor
from certifier.signer import KernelSigner


def load_training_data(path: str) -> list[tuple[str, float]]:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Certify an Aurelius kernel against experimental data."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to CSV/JSON file with SMILES and experimental values.",
    )
    parser.add_argument(
        "--domain",
        default="generic",
        help="Chemical domain descriptor (e.g., carbonate, ether, nitrile).",
    )
    parser.add_argument(
        "--output",
        default="aurelius_kernel.json",
        help="Output path for the signed kernel JSON.",
    )
    parser.add_argument(
        "--secret",
        default=None,
        help="HMAC secret key (default: AURELIUS_SECRET env var, or a dev-only fallback).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        print(f"Error: data file not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    pairs = load_training_data(args.data)
    if len(pairs) < 2:
        print("Error: need at least 2 data points for optimisation.", file=sys.stderr)
        sys.exit(1)

    # Split into train / validate
    split = int(len(pairs) * 0.8)
    train, validate = pairs[:split], pairs[split:]

    secret = (
        args.secret.encode("utf-8")
        if args.secret
        else os.environ.get("AURELIUS_SECRET", b"dev-only-fallback").encode("utf-8")
    )

    optimizer = KernelOptimizer()
    auditor = UncertaintyAuditor()
    signer = KernelSigner(secret)

    kernel = optimizer.optimize(train, domain_boundary={"domain": args.domain})
    audit = auditor.audit(kernel, validate)
    kernel["validation_metrics"]["audit"] = audit

    kernel = signer.sign(kernel)

    with open(args.output, "w") as f:
        json.dump(kernel, f, indent=2)

    print(f"Certified kernel written to {args.output}")
    print(f"  Domain         : {args.domain}")
    print(f"  Training pairs : {len(train)}")
    print(f"  Validation pairs: {len(validate)}")
    print(f"  Audit passed   : {audit['pass']}")


if __name__ == "__main__":
    main()
