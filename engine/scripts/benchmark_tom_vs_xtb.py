"""Benchmark TOM vs xTB/DFT on challenging electrolyte molecules.

Selects 20 molecules from external_property_benchmark.json that stress
the Topological Orbital Model (particle-in-a-box + Hueckel perturbations):

  - Highly conjugated (polyenes, vinyl carbonates)
  - Cross-conjugated (branched pi-systems)
  - Sterically hindered (non-planar aromatics, bulky substituents)
  - Aromatic heterocycles (pyridine, anisole)
  - Heavy-atom systems (sulfones, phosphates with d-orbital participation)

Computes:
  - Mean Absolute Error (MAE) for HOMO, LUMO, and gap
  - Spearman rank correlation coefficient (rho) for each orbital
  - Lists specific failure modes where TOM deviates > 1.0 eV from reference

Usage:
    python scripts/benchmark_tom_vs_xtb.py [--output benchmark_report.txt]
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem

from aurelius.scoring.oracle.quantum import (
    _generate_multi_xyz,
    _run_xtb,
    has_xtb,
    predict_tom_orbitals,
)
from aurelius.types import MoleculeContext

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = REPO_ROOT / "src" / "aurelius" / "data" / "external_property_benchmark.json"

# ---------------------------------------------------------------------------
# Challenging molecules — selected for known TOM failure modes
# ---------------------------------------------------------------------------
# Each tuple: (index in benchmark, reason_for_challenge)
_CHALLENGING_INDICES: list[tuple[int, str]] = [
    (0, "cyclic carbonate — reference DFT"),
    (2, "linear carbonate — simple reference"),
    (7, "nitrile — heteroatom perturbation"),
    (8, "sulfoxide — S d-orbital participation"),
    (9, "cyclic sulfone — heavy-atom ring"),
    (11, "linear ether — simple reference"),
    (12, "lactone — ring ester"),
    (14, "cyclic ether — reference"),
    (17, "ester — simple reference"),
    (19, "amide — N perturbation"),
    (20, "phosphate — P d-orbital"),
    (21, "phosphate — P d-orbital"),
    (23, "aromatic ether — pi conjugation + heteroatom"),
    (24, "aromatic heterocycle — N in ring"),
    (25, "fluorinated cyclic carbonate — F inductive"),
    (26, "vinyl carbonate — conjugated pi-system"),
    (41, "aromatic sulfone — extended conjugation + S"),
    (48, "aromatic nitrile — conjugated pi + CN"),
    (49, "dinitrile — dual CN perturbation"),
    (52, "multi-aromatic phosphate — steric + cross-conjugation"),
]


def _select_challenging_molecules(
    data: list[dict],
) -> list[tuple[dict, str]]:
    """Select 20 challenging molecules and return (entry, reason) pairs."""
    selected: list[tuple[dict, str]] = []
    for idx, reason in _CHALLENGING_INDICES:
        if idx < len(data):
            selected.append((data[idx], reason))
    return selected


def _compute_spearman_rho(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation coefficient."""
    n = len(x)
    if n < 3:
        return 0.0
    x_ranks = np.argsort(np.argsort(x)).astype(float)
    y_ranks = np.argsort(np.argsort(y)).astype(float)
    d = x_ranks - y_ranks
    rho = 1.0 - (6.0 * np.sum(d * d)) / (n * (n * n - 1.0))
    return float(rho)


def _run_xtb_on_mol(smiles: str) -> dict[str, float] | None:
    """Run xTB on a molecule, return HOMO/LUMO or None."""
    ctx = MoleculeContext.from_smiles(smiles)
    if ctx is None:
        return None
    conformers = _generate_multi_xyz(ctx.mol, n_conformers=3)
    if not conformers:
        return None
    for xyz, _ in conformers[:3]:
        result = _run_xtb(xyz)
        if result is not None:
            return result
    return None


def main() -> None:
    if not os.path.exists(BENCHMARK_PATH):
        print(f"ERROR: Benchmark file not found at {BENCHMARK_PATH}")
        sys.exit(1)

    with open(BENCHMARK_PATH) as f:
        all_data = json.load(f)

    selected = _select_challenging_molecules(all_data)
    print(f"Selected {len(selected)} challenging molecules for benchmark.\n")

    tom_homo: list[float] = []
    tom_lumo: list[float] = []
    ref_homo: list[float] = []
    ref_lumo: list[float] = []
    names: list[str] = []
    smiles_list: list[str] = []
    failure_modes: list[dict] = []

    has_xtb_available = has_xtb()
    if has_xtb_available:
        print("xTB binary found — will run xTB on each molecule.\n")
    else:
        print("xTB binary NOT found — comparing TOM against DFT/TOM references only.\n")

    xtb_homo: list[float] = []
    xtb_lumo: list[float] = []

    for entry, reason in selected:
        smi = entry["smiles"]
        name = entry["name"]
        ref_h = entry.get("homo_eV")
        ref_l = entry.get("lumo_eV")
        ref_hsrc = entry.get("homo_source", "unknown")

        if ref_h is None or ref_l is None:
            print(f"  SKIP {name}: missing reference HOMO/LUMO")
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"  SKIP {name}: invalid SMILES")
            continue

        # Run TOM
        try:
            tom_h, tom_l = predict_tom_orbitals(mol)
        except Exception as e:
            print(f"  TOM FAIL {name}: {e}")
            continue

        tom_homo.append(tom_h)
        tom_lumo.append(tom_l)
        ref_homo.append(ref_h)
        ref_lumo.append(ref_l)
        names.append(name)
        smiles_list.append(smi)

        # Compute TOM deviation
        homo_err = abs(tom_h - ref_h)
        lumo_err = abs(tom_l - ref_l)

        flag = ""
        if homo_err > 1.0 or lumo_err > 1.0:
            flag = " *** FAILURE MODE ***"
            failure_modes.append({
                "name": name,
                "smiles": smi,
                "homo_ref": ref_h,
                "homo_tom": tom_h,
                "lumo_ref": ref_l,
                "lumo_tom": tom_l,
                "homo_err": round(homo_err, 3),
                "lumo_err": round(lumo_err, 3),
                "reason": reason,
                "reference_source": ref_hsrc,
            })

        print(f"  {name:45s} | TOM HOMO={tom_h:+.3f} LUMO={tom_l:+.3f} | "
              f"Ref HOMO={ref_h:+.3f} LUMO={ref_l:+.3f} | "
              f"|ΔH|={homo_err:.3f} |ΔL|={lumo_err:.3f}{flag}")

        # Run xTB if available
        if has_xtb_available:
            xtb_result = _run_xtb_on_mol(smi)
            if xtb_result is not None:
                xtb_homo.append(xtb_result["homo_eV"])
                xtb_lumo.append(xtb_result["lumo_eV"])
            else:
                xtb_homo.append(float("nan"))
                xtb_lumo.append(float("nan"))

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  BENCHMARK SUMMARY: TOM vs Reference (DFT/TOM)")
    print("=" * 72)

    if not tom_homo:
        print("  No valid comparisons.")
        return

    n = len(tom_homo)
    homo_errors = [abs(tom_homo[i] - ref_homo[i]) for i in range(n)]
    lumo_errors = [abs(tom_lumo[i] - ref_lumo[i]) for i in range(n)]

    mae_homo = float(np.mean(homo_errors))
    mae_lumo = float(np.mean(lumo_errors))
    mae_gap = float(np.mean([
        abs((tom_lumo[i] - tom_homo[i]) - (ref_lumo[i] - ref_homo[i]))
        for i in range(n)
    ]))

    print(f"  N molecules:              {n}")
    print(f"  MAE HOMO:                 {mae_homo:.3f} eV")
    print(f"  MAE LUMO:                 {mae_lumo:.3f} eV")
    print(f"  MAE HOMO-LUMO gap:        {mae_gap:.3f} eV")

    # Spearman rank correlation
    rho_homo = _compute_spearman_rho(tom_homo, ref_homo)
    rho_lumo = _compute_spearman_rho(tom_lumo, ref_lumo)
    print(f"  Spearman rho HOMO:        {rho_homo:.3f}")
    print(f"  Spearman rho LUMO:        {rho_lumo:.3f}")

    # Separate DFT-sourced vs TOM-sourced references
    entry_source_map = {}
    for i, (entry, _) in enumerate(selected):
        entry_source_map[i] = entry.get("homo_source", "")

    dft_indices = [i for i in range(n) if "DFT" in str(entry_source_map.get(i, ""))]
    tom_ref_indices = [i for i in range(n) if "TOM" in str(entry_source_map.get(i, ""))]

    if dft_indices:
        dft_err_h = [homo_errors[i] for i in dft_indices]
        dft_err_l = [lumo_errors[i] for i in dft_indices]
        print(f"\n  Subset: DFT-sourced references ({len(dft_indices)} molecules)")
        print(f"    MAE HOMO:              {float(np.mean(dft_err_h)):.3f} eV")
        print(f"    MAE LUMO:              {float(np.mean(dft_err_l)):.3f} eV")

    if tom_ref_indices:
        tom_err_h = [homo_errors[i] for i in tom_ref_indices]
        tom_err_l = [lumo_errors[i] for i in tom_ref_indices]
        print(f"\n  Subset: TOM-sourced references ({len(tom_ref_indices)} molecules)")
        print(f"    MAE HOMO:              {float(np.mean(tom_err_h)):.3f} eV")
        print(f"    MAE LUMO:              {float(np.mean(tom_err_l)):.3f} eV")

    # xTB comparison (if available)
    if has_xtb_available and any(not math.isnan(x) for x in xtb_homo):
        valid_xtb = [
            i for i in range(len(xtb_homo))
            if not math.isnan(xtb_homo[i]) and not math.isnan(xtb_lumo[i])
        ]
        if valid_xtb:
            print(f"\n  xTB comparison ({len(valid_xtb)} molecules with xTB results):")
            tom_vs_xtb_h = [abs(tom_homo[i] - xtb_homo[i]) for i in valid_xtb]
            tom_vs_xtb_l = [abs(tom_lumo[i] - xtb_lumo[i]) for i in valid_xtb]
            print(f"    TOM vs xTB MAE HOMO:  {float(np.mean(tom_vs_xtb_h)):.3f} eV")
            print(f"    TOM vs xTB MAE LUMO:  {float(np.mean(tom_vs_xtb_l)):.3f} eV")

    # ------------------------------------------------------------------
    # Report failure modes
    # ------------------------------------------------------------------
    print(f"\n{'=' * 72}")
    print(f"  FAILURE MODES (|Δ| > 1.0 eV): {len(failure_modes)} found")
    print(f"{'=' * 72}")

    if failure_modes:
        for fm in failure_modes:
            print(f"\n  --- {fm['name']} ---")
            print(f"  SMILES:          {fm['smiles']}")
            print(f"  Reason:          {fm['reason']}")
            print(f"  Reference src:   {fm['reference_source']}")
            print(f"  HOMO ref={fm['homo_ref']:+.3f}  TOM={fm['homo_tom']:+.3f}  Δ={fm['homo_err']:+.3f}")
            print(f"  LUMO ref={fm['lumo_ref']:+.3f}  TOM={fm['lumo_tom']:+.3f}  Δ={fm['lumo_err']:+.3f}")
    else:
        print("  No significant failures detected.")

    # Write detailed report
    output_path = sys.argv[1] if len(sys.argv) > 1 else None
    if output_path:
        with open(output_path, "w") as f:
            f.write("TOM vs Reference Benchmark Report\n")
            f.write(f"N={n}, MAE HOMO={mae_homo:.3f}, MAE LUMO={mae_lumo:.3f}, "
                    f"MAE Gap={mae_gap:.3f}\n")
            f.write(f"Spearman rho HOMO={rho_homo:.3f}, LUMO={rho_lumo:.3f}\n\n")
            f.write("Per-molecule results:\n")
            for i in range(n):
                f.write(f"{names[i]:45s} TOM HOMO={tom_homo[i]:+.3f} LUMO={tom_lumo[i]:+.3f} "
                        f"Ref HOMO={ref_homo[i]:+.3f} LUMO={ref_lumo[i]:+.3f} "
                        f"ΔH={homo_errors[i]:.3f} ΔL={lumo_errors[i]:.3f}\n")
            if failure_modes:
                f.write("\n\nFailure Modes:\n")
                for fm in failure_modes:
                    f.write(f"{fm['name']}: HOMO Δ={fm['homo_err']:.3f} LUMO Δ={fm['lumo_err']:.3f} "
                            f"({fm['reason']})\n")
        print(f"\nReport written to {output_path}")


if __name__ == "__main__":
    main()
