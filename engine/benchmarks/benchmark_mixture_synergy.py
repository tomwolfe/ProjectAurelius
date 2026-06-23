#!/usr/bin/env python3
"""Benchmark: Mixture Synergy Bonus — pure-function validation.

Asserts that the Margules-inspired synergy bonus correctly rewards
complementary pairs (high-dielectric + low-viscosity) while rejecting
"Frankenstein" pairs (both high-dielectric).

Usage:
    python -m benchmarks.benchmark_mixture_synergy
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aurelius.scoring.oracle.gc import mixture_synergy_bonus, mixture_synergy_bonus_ternary


def main() -> None:
    # Complementary pair: high-dielectric + low-viscosity
    synergy = mixture_synergy_bonus(d1=8.0, v1=2.5, d2=2.0, v2=0.5)
    assert synergy > 0.5, f"Complementary pair synergy={synergy:.4f}, expected >0.5"
    print(f"  PASS: Complementary pair synergy={synergy:.4f} > 0.5")

    # Frankenstein pair: both high-dielectric, neither low-viscosity
    synergy = mixture_synergy_bonus(d1=8.0, v1=2.5, d2=7.0, v2=2.3)
    assert synergy == 0.0, f"Frankenstein pair synergy={synergy:.4f}, expected 0.0"
    print(f"  PASS: Frankenstein pair synergy={synergy:.4f} == 0.0")

    # Ternary mixture: EC (high-dielectric) + DMC (low-viscosity) + EMC (low-viscosity)
    synergy_ternary = mixture_synergy_bonus_ternary(
        d1=8.0, d2=3.0, d3=2.0,
        v1=2.5, v2=0.5, v3=0.4,
        frac1=0.33, frac2=0.33,
    )
    assert synergy_ternary > 0.5, f"Ternary complementary synergy={synergy_ternary:.4f}, expected >0.5"
    print(f"  PASS: Ternary complementary synergy={synergy_ternary:.4f} > 0.5")

    # Ternary mixture: all homogeneous (no complementarity)
    synergy_ternary_hom = mixture_synergy_bonus_ternary(
        d1=5.0, d2=5.0, d3=5.0,
        v1=2.0, v2=2.0, v3=2.0,
        frac1=0.33, frac2=0.33,
    )
    assert synergy_ternary_hom == 0.0, f"Ternary homogeneous synergy={synergy_ternary_hom:.4f}, expected 0.0"
    print(f"  PASS: Ternary homogeneous synergy={synergy_ternary_hom:.4f} == 0.0")

    print()
    print("MIXTURE SYNERGY BONUS: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
