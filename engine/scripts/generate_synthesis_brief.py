#!/usr/bin/env python3
"""Generate a Synthesis Target Brief for top EA discoveries.

Maps top molecules from run_summary.json to commercially available
precursors via BRICS fragmentation, outputting a Markdown table.

Usage:
    python scripts/generate_synthesis_brief.py [--input RUN_SUMMARY.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rdkit import Chem
from rdkit.Chem import BRICS

from aurelius.agent.mutation.brics import (
    _BRICS_LINKER_FRAGMENTS,
    _strip_brics_dummies,
    brics_building_block_coverage,
    get_brics_types,
)
from aurelius.constants import COMMERCIAL_BUILDING_BLOCK_SMILES
from aurelius.types import MoleculeContext

# Pre-compile commercial building block molecules with canonical SMILES lookup
_BB_MOLS: list[Chem.Mol] = []
_BB_CANON: set[str] = set()
for _smi in COMMERCIAL_BUILDING_BLOCK_SMILES:
    _mol = Chem.MolFromSmiles(_smi)
    if _mol is not None:
        _BB_MOLS.append(_mol)
        _BB_CANON.add(Chem.MolToSmiles(_mol))


def _find_best_precursor(frag_smi: str) -> str | None:
    """Find the best matching commercial precursor for a BRICS fragment.

    First tries exact canonical SMILES match, then falls back to
    substructure matching for fragments with >= 3 heavy atoms.
    Returns the matching commercial SMILES, or None.
    """
    core_smi = _strip_brics_dummies(frag_smi)
    if core_smi is None:
        return None
    core_mol = Chem.MolFromSmiles(core_smi)
    if core_mol is None:
        return None
    core_canon = Chem.MolToSmiles(core_mol)
    n_heavy = core_mol.GetNumHeavyAtoms()

    # Exact canonical match
    if core_canon in _BB_CANON:
        return core_canon

    # For fragments with >= 3 heavy atoms, try substructure matching
    if n_heavy >= 3:
        best: tuple[int, str | None] = (0, None)
        for bb_smi, bb_mol in zip(COMMERCIAL_BUILDING_BLOCK_SMILES, _BB_MOLS):
            if core_mol.HasSubstructMatch(bb_mol):
                bb_heavy = bb_mol.GetNumHeavyAtoms()
                if bb_heavy > best[0]:
                    best = (bb_heavy, bb_smi)
        return best[1]

    return None


def _infer_linkers(mol: Chem.Mol) -> list[str]:
    """Infer which BRICS linker types would be needed for reassembly."""
    try:
        frags = list(BRICS.BRICSDecompose(mol))
    except Exception:
        return []
    if not frags:
        return []
    all_types: set[int] = set()
    for fs in frags:
        frag_ctx = MoleculeContext.from_brics_fragment(fs)
        if frag_ctx is None:
            continue
        all_types.update(get_brics_types(frag_ctx.mol))
    linker_names = []
    for linker_smi, linker_desc in _BRICS_LINKER_FRAGMENTS:
        linker_mol = Chem.MolFromSmiles(linker_smi)
        if linker_mol is not None:
            l_types = get_brics_types(linker_mol)
            if l_types & all_types:
                linker_names.append(linker_desc)
    return linker_names if linker_names else ["direct coupling"]


def _is_trivial_precursor(smi: str) -> bool:
    """Check if a precursor is too trivial to be informative."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return True
    n_heavy = mol.GetNumHeavyAtoms()
    return n_heavy <= 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Synthesis Target Brief")
    parser.add_argument("--input", default="run_summary.json", help="Path to run_summary.json")
    parser.add_argument("--output", default="docs/synthesis_brief.md", help="Output Markdown path")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)

    discoveries = data.get("discoveries", [])
    if not discoveries:
        print("No discoveries found in run_summary.json", file=sys.stderr)
        sys.exit(1)

    top = sorted(discoveries, key=lambda d: d.get("total_score", 0.0), reverse=True)[:10]

    rows: list[str] = []
    for rank, disc in enumerate(top, 1):
        smiles = disc["smiles"]
        score = disc.get("total_score", 0.0)
        ctx = MoleculeContext.from_smiles(smiles)
        if ctx is None:
            continue
        mol = ctx.mol

        coverage = brics_building_block_coverage(mol)
        frags: list[str] = []
        try:
            frags = list(BRICS.BRICSDecompose(mol))
        except Exception:
            pass

        precursors: list[str] = []
        for fs in frags:
            match = _find_best_precursor(fs)
            if match is not None and not _is_trivial_precursor(match):
                if match not in precursors:
                    precursors.append(match)

        linkers = _infer_linkers(mol)

        precursor_str = "; ".join(precursors) if precursors else "*none*"
        linker_str = "; ".join(linkers)

        rows.append(
            f"| {rank} | `{smiles}` | {score:.1f} | {coverage:.0%} | "
            f"`{precursor_str}` | {linker_str} |"
        )

    header = (
        "# Synthesis Target Brief\n\n"
        "*Auto-generated by `scripts/generate_synthesis_brief.py`.*\n\n"
        "Maps top-10 EA discoveries to commercially available precursors.\n\n"
        "| Rank | Target SMILES | Score | Coverage | Required Precursors | BRICS Linker |\n"
        "|------|--------------|-------|----------|--------------------|-------------|\n"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        f.write(header)
        f.write("\n".join(rows))
        f.write("\n")

    print(f"Synthesis brief written to {output_path}")


if __name__ == "__main__":
    main()
