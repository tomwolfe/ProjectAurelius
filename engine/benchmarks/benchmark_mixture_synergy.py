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

from aurelius.scoring.oracle.gc import mixture_synergy_bonus


def main() -> None:
    # Complementary pair: high-dielectric + low-viscosity
    synergy = mixture_synergy_bonus(d1=8.0, v1=2.5, d2=2.0, v2=0.5)
    assert synergy > 0.5, f"Complementary pair synergy={synergy:.4f}, expected >0.5"
    print(f"  PASS: Complementary pair synergy={synergy:.4f} > 0.5")

    # Frankenstein pair: both high-dielectric, neither low-viscosity
    synergy = mixture_synergy_bonus(d1=8.0, v1=2.5, d2=7.0, v2=2.3)
    assert synergy == 0.0, f"Frankenstein pair synergy={synergy:.4f}, expected 0.0"
    print(f"  PASS: Frankenstein pair synergy={synergy:.4f} == 0.0")

    print()
    print("MIXTURE SYNERGY BONUS: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
