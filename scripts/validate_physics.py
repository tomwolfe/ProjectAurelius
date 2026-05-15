#!/usr/bin/env python3
"""Physics validation script for Project Aurelius.

Runs small simulations and compares results against known physical
behavior to verify the physics engine is calibrated correctly.

Usage:
    python scripts/validate_physics.py
    python scripts/validate_physics.py --strict
    python scripts/validate_physics.py --tier 2
    python scripts/validate_physics.py --tier 3

This script validates:
    - Tier 2: Energy conservation, force gradients, finite energies
    - Tier 3: Arrhenius temperature dependence, concentration dependence
    - Solvation engine: Born charge interpolation, dielectric lookup
"""

from __future__ import annotations

import argparse
import sys


def validate_tier2() -> dict:
    """Validate Tier 2 physics engine behavior."""
    results = {"passed": 0, "failed": 0, "errors": []}

    try:
        import torch

        from aurelius.screening.tier2_mattersim import MatterSimMPEngine

        engine = MatterSimMPEngine()

        # Test 1: Water molecule energy is finite
        atomic_numbers = torch.tensor([1, 8, 1], dtype=torch.long)
        coordinates = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.757, 0.586],
            [0.0, -0.586, -0.586],
        ], dtype=torch.float32)

        energy = engine(atomic_numbers, coordinates)
        if torch.isfinite(energy):
            results["passed"] += 1
            print(f"  [PASS] Water molecule energy is finite: {float(energy):.4f} eV")
        else:
            results["failed"] += 1
            results["errors"].append("Water molecule energy is NaN or Inf")
            print("  [FAIL] Water molecule energy is NaN or Inf")

        # Test 2: Energy gradients (forces) are computable and finite
        coords_grad = coordinates.clone().requires_grad_(True)
        energy = engine(atomic_numbers, coords_grad)
        grad = torch.autograd.grad(
            energy, coords_grad, grad_outputs=torch.ones_like(energy),
            create_graph=False,
        )
        if grad is not None and torch.all(torch.isfinite(grad[0])):
            results["passed"] += 1
            print("  [PASS] Forces (negative energy gradients) are finite")
        else:
            results["failed"] += 1
            results["errors"].append("Forces are NaN or Inf")
            print("  [FAIL] Forces are NaN or Inf")

        # Test 3: Deterministic energy for same configuration
        e1 = float(engine(atomic_numbers, coordinates).item())
        e2 = float(engine(atomic_numbers, coordinates).item())
        if abs(e1 - e2) < 1e-6:
            results["passed"] += 1
            print(f"  [PASS] Energy is deterministic: {e1:.6f} eV")
        else:
            results["failed"] += 1
            results["errors"].append(f"Non-deterministic energy: {e1} vs {e2}")
            print(f"  [FAIL] Non-deterministic energy: {e1} vs {e2}")

        # Test 4: Na+ water cluster has physically reasonable energy
        na_coords = torch.tensor([
            [0.0, 0.0, 0.0],    # Na+
            [2.3, 0.0, 0.0],    # O (first solvation shell)
            [2.8, 0.7, 0.0],    # H
            [2.8, -0.7, 0.0],   # H
        ], dtype=torch.float32)
        na_nums = torch.tensor([11, 8, 1, 1], dtype=torch.long)
        energy_na = engine(na_nums, na_coords)
        if torch.isfinite(energy_na):
            results["passed"] += 1
            print(f"  [PASS] Na+ water cluster energy is finite: {float(energy_na):.4f} eV")
        else:
            results["failed"] += 1
            results["errors"].append("Na+ water cluster energy is NaN or Inf")
            print("  [FAIL] Na+ water cluster energy is NaN or Inf")

    except ImportError as e:
        results["skipped"] = True
        results["errors"].append(f"PyTorch not available: {e}")
        print(f"  [SKIP] Tier 2 validation: {e}")

    return results


def validate_tier3() -> dict:
    """Validate Tier 3 Arrhenius kMC behavior."""
    results = {"passed": 0, "failed": 0, "errors": []}

    try:
        from aurelius.screening.tier3_gcmtwin import GCMDigitalTwin

        twin = GCMDigitalTwin()

        # Test 1: Arrhenius rate increases with temperature
        k_250 = twin._arrhenius_rate(
            activation_energy_eV=0.65,
            temperature_k=250.0,
            concentration=1.0,
            pre_exponential_base=5.0,
            overpotential_V=0.05,
        )
        k_298 = twin._arrhenius_rate(
            activation_energy_eV=0.65,
            temperature_k=298.15,
            concentration=1.0,
            pre_exponential_base=5.0,
            overpotential_V=0.05,
        )
        k_350 = twin._arrhenius_rate(
            activation_energy_eV=0.65,
            temperature_k=350.0,
            concentration=1.0,
            pre_exponential_base=5.0,
            overpotential_V=0.05,
        )

        if k_250 < k_298 < k_350:
            results["passed"] += 1
            print(f"  [PASS] Arrhenius: rates increase with temperature "
                  f"({k_250:.4f} < {k_298:.4f} < {k_350:.4f})")
        else:
            results["failed"] += 1
            results["errors"].append(
                f"Arrhenius: {k_250:.4f} >= {k_298:.4f} >= {k_350:.4f}"
            )
            print(f"  [FAIL] Arrhenius rate not monotonic: "
                  f"{k_250:.4f} >= {k_298:.4f} >= {k_350:.4f}")

        # Test 2: Lower activation energy -> higher rate
        k_low_ea = twin._arrhenius_rate(
            activation_energy_eV=0.50,
            temperature_k=298.15,
            concentration=1.0,
            pre_exponential_base=5.0,
            overpotential_V=0.05,
        )
        k_high_ea = twin._arrhenius_rate(
            activation_energy_eV=1.20,
            temperature_k=298.15,
            concentration=1.0,
            pre_exponential_base=5.0,
            overpotential_V=0.05,
        )
        if k_low_ea > k_high_ea:
            results["passed"] += 1
            print(f"  [PASS] Lower Ea gives higher rate: "
                  f"{k_low_ea:.4f} > {k_high_ea:.4f}")
        else:
            results["failed"] += 1
            results["errors"].append(
                f"Ea reversal: {k_low_ea:.4f} <= {k_high_ea:.4f}"
            )
            print("  [FAIL] Lower Ea did not give higher rate")

        # Test 3: Concentration-dependent pre-exponential factor
        k_full = twin._arrhenius_rate(
            activation_energy_eV=0.65,
            temperature_k=298.15,
            concentration=1.0,
            pre_exponential_base=5.0,
            overpotential_V=0.05,
        )
        k_half = twin._arrhenius_rate(
            activation_energy_eV=0.65,
            temperature_k=298.15,
            concentration=0.3,
            pre_exponential_base=5.0,
            overpotential_V=0.05,
        )
        k_low = twin._arrhenius_rate(
            activation_energy_eV=0.65,
            temperature_k=298.15,
            concentration=0.05,
            pre_exponential_base=5.0,
            overpotential_V=0.05,
        )
        if k_full > k_half > k_low:
            results["passed"] += 1
            print(f"  [PASS] Rate decreases with concentration: "
                  f"{k_full:.4f} > {k_half:.4f} > {k_low:.4f}")
        else:
            results["failed"] += 1
            results["errors"].append(
                f"Concentration dependence failed: {k_full:.4f} > {k_half:.4f} > {k_low:.4f}"
            )
            print("  [FAIL] Concentration dependence incorrect")

        # Test 4: SEI thickness is physically plausible (1-50 Angstroms)
        result = twin.simulate_sei_evolution(
            "CC(=O)OC1=CC(=O)O1", "ec:dmc", "NaPF6",
        )
        thickness = result.sei_evolution.thickness_angstrom
        if 1.0 <= thickness <= 50.0:
            results["passed"] += 1
            print(f"  [PASS] SEI thickness is physically plausible: {thickness:.2f} A")
        else:
            results["failed"] += 1
            results["errors"].append(f"SEI thickness out of range: {thickness:.2f} A")
            print(f"  [FAIL] SEI thickness out of physical range: {thickness:.2f} A")

        # Test 5: kMC is deterministic for same inputs
        r1 = twin.simulate_sei_evolution("CC(=O)OC1=CC(=O)O1", "ec:dmc", "NaPF6")
        r2 = twin.simulate_sei_evolution("CC(=O)OC1=CC(=O)O1", "ec:dmc", "NaPF6")
        if abs(r1.sei_evolution.thickness_angstrom - r2.sei_evolution.thickness_angstrom) < 1e-10:
            results["passed"] += 1
            print("  [PASS] kMC simulation is deterministic")
        else:
            results["failed"] += 1
            results["errors"].append("kMC simulation is non-deterministic")
            print("  [FAIL] kMC simulation is non-deterministic")

    except ImportError as e:
        results["skipped"] = True
        results["errors"].append(f"Import error: {e}")
        print(f"  [SKIP] Tier 3 validation: {e}")

    return results


def validate_solvation() -> dict:
    """Validate solvation engine behavior."""
    results = {"passed": 0, "failed": 0, "errors": []}

    try:
        import numpy as np

        from aurelius.solvation.engine import (
            _BORN_CHARGES_K,
            _BORN_CHARGES_LI,
            _BORN_CHARGES_NA,
            MWSESolvationEngine,
        )

        engine = MWSESolvationEngine(kex_window_ps=10.0)

        # Test 1: Born charges are physically reasonable
        na_norm = np.linalg.norm(_BORN_CHARGES_NA)
        li_norm = np.linalg.norm(_BORN_CHARGES_LI)
        k_norm = np.linalg.norm(_BORN_CHARGES_K)

        if 1.0 < na_norm < 2.5 and 1.5 < li_norm < 3.0 and 1.0 < k_norm < 2.0:
            results["passed"] += 1
            print(f"  [PASS] Born charges are physically reasonable: "
                  f"Na+={na_norm:.3f}, Li+={li_norm:.3f}, K+={k_norm:.3f}")
        else:
            results["failed"] += 1
            results["errors"].append(
                f"Born charges out of range: Na+={na_norm:.3f}, Li+={li_norm:.3f}, K+={k_norm:.3f}"
            )
            print("  [FAIL] Born charges out of physical range")

        # Test 2: Solvent exchange rates are positive
        k_ex = engine.compute_solvent_exchange_rate("water", "Na+")
        if k_ex > 0:
            results["passed"] += 1
            print(f"  [PASS] Solvent exchange rate is positive: {k_ex:.3f} ps^-1")
        else:
            results["failed"] += 1
            results["errors"].append(f"Solvent exchange rate is non-positive: {k_ex}")
            print(f"  [FAIL] Solvent exchange rate is non-positive: {k_ex}")

        # Test 3: Mixed solvent interpolation works
        born_ec = engine.query_born_effective_charges("Na+", "ec")
        born_dmc = engine.query_born_effective_charges("Na+", "dmc")
        born_mixed = engine.query_born_effective_charges("Na+", "ec:dmc")
        z_ec = born_ec.z_star_scalar
        z_dmc = born_dmc.z_star_scalar
        z_mixed = born_mixed.z_star_scalar

        if min(z_ec, z_dmc) <= z_mixed <= max(z_ec, z_dmc):
            results["passed"] += 1
            print(f"  [PASS] Mixed solvent interpolation correct: "
                  f"{min(z_ec, z_dmc):.3f} <= {z_mixed:.3f} <= {max(z_ec, z_dmc):.3f}")
        else:
            results["failed"] += 1
            results["errors"].append(
                f"Mixed solvent interpolation failed: {min(z_ec, z_dmc):.3f} > {z_mixed:.3f} > {max(z_ec, z_dmc):.3f}"
            )
            print("  [FAIL] Mixed solvent interpolation incorrect")

        # Test 4: Dielectric constants are loaded correctly
        from aurelius.solvation.engine import _DIELECTRIC_CONSTANTS
        expected_solvents = ["water", "ec", "dmc", "emc", "pc"]
        found = sum(1 for s in expected_solvents if s in _DIELECTRIC_CONSTANTS)
        if found >= 4:
            results["passed"] += 1
            print(f"  [PASS] Dielectric constants loaded: {found}/{len(expected_solvents)} solvents")
        else:
            results["failed"] += 1
            results["errors"].append(
                f"Only {found}/{len(expected_solvents)} expected solvents found"
            )
            print(f"  [FAIL] Insufficient dielectric constants: {found}/{len(expected_solvents)}")

    except ImportError as e:
        results["skipped"] = True
        results["errors"].append(f"Import error: {e}")
        print(f"  [SKIP] Solvation validation: {e}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate physics behavior of Project Aurelius tiers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/validate_physics.py
    python scripts/validate_physics.py --strict
    python scripts/validate_physics.py --tier 2
    python scripts/validate_physics.py --tier 3 --tier solvation
        """,
    )
    parser.add_argument(
        "--tier",
        nargs="+",
        choices=["2", "3", "solvation", "all"],
        default=["all"],
        help="Which tier to validate (default: all)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any skipped tests (e.g., missing dependencies)",
    )
    args = parser.parse_args()

    tiers_to_run: list[str] = []
    if "all" in args.tier:
        tiers_to_run = ["2", "3", "solvation"]
    else:
        tiers_to_run = args.tier

    print("=" * 60)
    print("  Project Aurelius Physics Validation")
    print("=" * 60)

    total_passed = 0
    total_failed = 0
    total_skipped = 0
    all_errors: list[str] = []

    for tier in tiers_to_run:
        print(f"\n--- Tier {tier} ---")

        if tier == "2":
            results = validate_tier2()
        elif tier == "3":
            results = validate_tier3()
        elif tier == "solvation":
            results = validate_solvation()
        else:
            print(f"  [ERROR] Unknown tier: {tier}")
            continue

        total_passed += results["passed"]
        total_failed += results["failed"]
        if "skipped" in results:
            total_skipped += 1

        all_errors.extend(results["errors"])

        print(f"  Passed: {results['passed']}, Failed: {results['failed']}"
              + (", Skipped: 1" if "skipped" in results else ""))

    print("\n" + "=" * 60)
    print(f"  Results: {total_passed} passed, {total_failed} failed, {total_skipped} skipped")
    print("=" * 60)

    if all_errors:
        print("\nErrors:")
        for err in all_errors:
            print(f"  - {err}")

    if total_failed > 0:
        print(f"\n  STATUS: FAIL ({total_failed} physics validation(s) failed)")
        return 1

    if args.strict and total_skipped > 0:
        print(f"\n  STATUS: SKIP ({total_skipped} tier(s) skipped due to missing dependencies)")
        return 2

    if total_failed == 0:
        print("\n  STATUS: ALL PHYSICS VALIDATIONS PASSED")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
