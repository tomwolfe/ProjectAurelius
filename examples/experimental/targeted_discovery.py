#!/usr/bin/env python3
"""
Phase 3-4: Targeted Autonomous Discovery with Tier 0 Injection.

Generates diverse borate/fluorinated/unsaturated candidates,
screens them through the full pipeline with dynamic Ea calibration,
and extracts viable molecules (score >= 65.0).
"""

import gc
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, "src")

from rdkit import Chem
from rdkit.Chem import BRICS, Descriptors, rdFingerprintGenerator
from rdkit.DataStructs import BitVectToText, CreateFromBitString, FingerprintSimilarity

from aurelius.config import AureliusConfig, initialize_environment
from aurelius.pipeline import AureliusPipeline
from aurelius.screening.tier3_gcmtwin import Tier0ActivationPredictor

# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
np.random.seed(42)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def safe_mol_from_smiles(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def mol_to_fp(mol) -> Any:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return gen.GetFingerprint(mol)


def serialize_fp(fp) -> str:
    return BitVectToText(fp)


def deserialize_fp(hex_str: str):
    return CreateFromBitString(hex_str)


def tanimoto(fp1, fp2) -> float:
    return FingerprintSimilarity(fp1, fp2)


def is_valid_mol(mol) -> bool:
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return False
    mw = Descriptors.ExactMolWt(mol)
    return mw < 400.0  # Slightly higher limit for borates


def memory_cleanup():
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Candidate Generation Engine
# ---------------------------------------------------------------------------


class CandidateGenerator:
    """Generates diverse electrolyte-relevant candidate molecules."""

    def __init__(self):
        self.rng = np.random.RandomState(42)
        self.seen_smiles: set[str] = set()
        self.known_fps: list = []

    def add_known(self, smiles: str):
        self.seen_smiles.add(smiles)
        mol = safe_mol_from_smiles(smiles)
        if mol is not None:
            self.known_fps.append(mol_to_fp(mol))

    def is_novel(self, smiles: str) -> bool:
        if smiles in self.seen_smiles:
            return False
        mol = safe_mol_from_smiles(smiles)
        if mol is None:
            return False
        fp = mol_to_fp(mol)
        return all(tanimoto(fp, known) < 0.75 for known in self.known_fps)

    def _add_fluorine(self, mol) -> list[str]:
        """Add fluorine to non-carbonyl carbons."""
        generated = []
        try:
            rxn = Chem.ReactionFromSmarts("[*:1][H]>>[*:1]F")
            for prod in rxn.RunReactants((mol,)):
                s = Chem.MolToSmiles(prod[0], isomericSmiles=True)
                if s and self.is_novel(s) and is_valid_mol(safe_mol_from_smiles(s)):
                    generated.append(s)
        except Exception:
            pass
        return generated

    def _add_unsaturation(self, mol) -> list[str]:
        """Introduce C=C double bonds."""
        generated = []
        try:
            rxn = Chem.ReactionFromSmarts("[*:1][H]-[*:2][H]>>[*:1]=[*:2]")
            for prod in rxn.RunReactants((mol,)):
                s = Chem.MolToSmiles(prod[0], isomericSmiles=True)
                if s and self.is_novel(s) and is_valid_mol(safe_mol_from_smiles(s)):
                    generated.append(s)
        except Exception:
            pass
        return generated

    def _brics_reassemble(self, mol) -> list[str]:
        """BRICS decomposition + reassembly."""
        generated = []
        try:
            fragments = BRICS.BRICSDecompose(mol)
            if len(fragments) >= 2:
                for _ in range(10):
                    idx = self.rng.choice(len(fragments), size=2, replace=False)
                    try:
                        reassembled = BRICS.BRICSBuild(fragments[idx[0]], fragments[idx[1]])
                        if reassembled:
                            s = Chem.MolToSmiles(reassembled, isomericSmiles=True)
                            if s and self.is_novel(s) and is_valid_mol(safe_mol_from_smiles(s)):
                                generated.append(s)
                    except Exception:
                        pass
        except Exception:
            pass
        return generated

    def mutate(self, smiles: str, n_attempts: int = 20) -> list[str]:
        """Generate mutated variants."""
        mol = safe_mol_from_smiles(smiles)
        if mol is None:
            return []

        candidates = set()
        candidates.add(smiles)

        for _ in range(n_attempts):
            candidates.update(self._add_fluorine(mol))
            candidates.update(self._add_unsaturation(mol))
            candidates.update(self._brics_reassemble(mol))

        return [s for s in candidates if s != smiles]

    def generate_borate_candidates(self) -> list[str]:
        """Generate borate-containing candidates directly."""
        borate_templates = [
            # Tetraalkyl borates with fluorinated alkyl groups
            "B1OB(OCC(F)F)OB(OCC(F)F)O1",
            "B1OB(OCC(F)F)OB(OCC(F)F)O1",
            "B1OB(OCC(F)(F)F)OB(OCC(F)(F)F)O1",
            "B1OB(OCC(F)=CF)OB(OCC(F)=CF)O1",
            "B1OB(OCC=C)OB(OCC=C)O1",
            "B1OB(OCC#C)OB(OCC#C)O1",
            "B1OB(OCC(F)F)OB(OCC=C)O1",
            "B1OB(OCC(F)F)OB(OCC#N)O1",
            "B1OB(OCC(F)F)OB(N#CC)O1",
            "B1OB(OCC(F)F)OB(C(=O)OC)O1",
            # Mixed borate-carbonate
            "COC(=O)OB1OC(C(F)F)(OCC(F)F)O1",
            "CC(=O)OB1OC(C(F)F)(OCC(F)F)O1",
            "B1OB(OB(COC(F)F)(OCC(F)F))OB(OCC(F)F)O1",
            # Phenyl borates
            "B1OB(Oc2ccccc2)OB(Oc2ccccc2)O1",
            "B1OB(Oc2ccc(F)cc2)OB(Oc2ccc(F)cc2)O1",
            # Boron trifluoride adducts
            "FB(F)(F)C(=O)OC",
            "FB(F)(F)S(=O)(=O)C",
            # Boron-nitrile complexes
            "B(C#N)(C#N)C",
            "B(C#N)(C#N)C(F)(F)F",
        ]
        results = []
        for s in borate_templates:
            mol = safe_mol_from_smiles(s)
            if mol is not None and is_valid_mol(mol) and self.is_novel(s):
                results.append(s)
                self.add_known(s)
        return results

    def generate_fluorinated_candidates(self) -> list[str]:
        """Generate highly fluorinated carbonate candidates."""
        fluoro_templates = [
            "COC(=O)OC(F)(F)F",
            "COC(=O)OC(F)(F)C(F)F",
            "COC(=O)OCC(F)(F)C(F)F",
            "COC(=O)OC(F)=C(F)C(F)(F)F",
            "COC(=O)OC(F)=C(F)C(F)(F)F",
            "COC(=O)OCC(F)(F)C(F)(F)F",
            "COC(=O)OCC(F)F",
            "COC(=O)OC(F)F",
            "CCOC(=O)OCC(F)F",
            "COC(=O)OC=C(F)F",
            "COC(=O)OCC(F)F",
            "C1OC(=O)OC1(F)F",
            "C1OC(=O)OC1(F)(F)",
            "COC(=O)OC(F)(F)C(F)F",
            "FCC(F)(F)S(=O)(=O)OC=C",
            # Perfluorinated carbonates
            "COC(=O)OC(F)(C(F)F)C(F)(F)F",
            "COC(=O)OC(F)(F)C(F)(F)C(F)(F)F",
        ]
        results = []
        for s in fluoro_templates:
            mol = safe_mol_from_smiles(s)
            if mol is not None and is_valid_mol(mol) and self.is_novel(s):
                results.append(s)
                self.add_known(s)
        return results

    def generate_unsaturated_candidates(self) -> list[str]:
        """Generate unsaturated/polymerizable candidates."""
        unsat_templates = [
            "COC(=O)OCC=C",
            "COC(=O)OC(F)C=C",
            "COC(=O)OC(F)=CF",
            "COC(=O)C(F)=C(F)F",
            "FCC(F)(F)S(=O)(=O)OC=C",
            "COC(=O)OC=C(F)F",
            "COC(=O)OCC=C",
            "C=CC(=O)OC",
            "C=CC(=O)OC(=O)C",
            "C=CC(F)(F)C(=O)OC",
            "C=CC(F)(F)S(=O)(=O)C",
            "C=CC#N",
            "C=CC(=O)OC(F)(F)F",
        ]
        results = []
        for s in unsat_templates:
            mol = safe_mol_from_smiles(s)
            if mol is not None and is_valid_mol(mol) and self.is_novel(s):
                results.append(s)
                self.add_known(s)
        return results

    def generate_nitrile_sulfone_candidates(self) -> list[str]:
        """Generate nitrile/sulfone additive candidates."""
        templates = [
            "N#CCS(=O)(=O)C",
            "N#CCS(=O)(=O)CC#N",
            "N#CC(F)S(=O)(=O)C",
            "N#CCOC(=O)OC(F)S(=O)(=O)C",
            "N#CCS(=O)(=O)C(F)(F)F",
            "N#CCOCCS(=O)(=O)CC#N",
            "N#CCS(=O)(=O)C1=CC=CC=C1",
            "N#CCS(=O)(=O)C(F)(F)F",
            "N#CCS(=O)(=O)C(F)(F)C(F)F",
        ]
        results = []
        for s in templates:
            mol = safe_mol_from_smiles(s)
            if mol is not None and is_valid_mol(mol) and self.is_novel(s):
                results.append(s)
                self.add_known(s)
        return results


# ---------------------------------------------------------------------------
# Main Discovery Pipeline
# ---------------------------------------------------------------------------


def run_discovery():
    print("=" * 70)
    print("  AURELIUS v5.2 — Autonomous Discovery with Tier 0 Injection")
    print("=" * 70)

    # Initialize environment and pipeline
    initialize_environment()
    config = AureliusConfig()
    pipeline = AureliusPipeline(config, use_real_models=True)
    pipeline.initialize()

    # Inject Tier 0 Activation Energy Predictor
    twin = pipeline._gcmtwin
    if twin:
        tier0_pred = Tier0ActivationPredictor()
        twin._tier0_predictor = tier0_pred
        twin._use_tier0_prediction = True
        print("[DISCOVERY] Tier 0 Activation Energy Predictor ENABLED")
    else:
        raise RuntimeError("GCMD Twin not initialized")

    # Generate candidates
    generator = CandidateGenerator()

    # Load seed molecules
    seed_smiles = []
    for fpath in ["discovery_candidates.smi", "homogeneity_targeted_candidates.smi", "examples/molecules.smi"]:
        p = Path(fpath)
        if p.exists():
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        line = line.split(" #")[0].strip()
                        if line:
                            seed_smiles.append(line)

    # Add seeds to known
    for s in seed_smiles:
        generator.add_known(s)

    print(f"\n[DISCOVERY] Seed pool: {len(seed_smiles)} molecules")

    # Generate diverse candidates
    print("\n[DISCOVERY] Generating candidate molecules...")
    all_candidates = list(seed_smiles)  # Start with seeds
    all_candidates.extend(generator.generate_borate_candidates())
    all_candidates.extend(generator.generate_fluorinated_candidates())
    all_candidates.extend(generator.generate_unsaturated_candidates())
    all_candidates.extend(generator.generate_nitrile_sulfone_candidates())

    # Also mutate seeds to create more variants
    for smi in seed_smiles[:10]:
        variants = generator.mutate(smi, n_attempts=15)
        all_candidates.extend(variants)

    # Deduplicate
    seen = set()
    unique_candidates = []
    for s in all_candidates:
        if s and s not in seen:
            seen.add(s)
            unique_candidates.append(s)
    all_candidates = unique_candidates

    print(f"[DISCOVERY] Total unique candidates: {len(all_candidates)}")

    # Screen all candidates
    print(f"\n[DISCOVERY] Screening {len(all_candidates)} candidates with dynamic kinetics...")
    print(f"{'SMILES':50s} | {'Score':>6s} | {'Sigma':>6s} | {'Desolv':>6s} | {'Homog':>6s} | {'MX':>5s} | Viable")
    print("-" * 110)

    results = []
    discoveries = []
    start_time = time.time()

    for _i, smi in enumerate(all_candidates):
        try:
            result = pipeline.screen_molecule(smi)
            score = result.get("score")
            if score is None:
                continue

            total = score.total_score
            sigma = score.sigma_score
            desolv = score.desolvation_score
            homog = score.sei_homogeneity_score
            mx = score.mx_synthesis_score
            viable = score.is_viable

            status = "**VIABLE**" if viable else ""

            print(f"{smi:50s} | {total:>5.1f} | {sigma:>6.1f} | {desolv:>6.1f} | {homog:>6.1f} | {mx:>5.1f} | {status}")

            entry = {
                "smiles": smi,
                "total_score": total,
                "sigma_score": sigma,
                "desolvation_score": desolv,
                "sei_homogeneity_score": homog,
                "mx_synthesis_score": mx,
                "is_viable": viable,
                "rejection_reasons": score.rejection_reasons,
                "tier1_viable": score.tier1_viable,
                "tier2_viable": score.tier2_viable,
                "tier3_viable": score.tier3_viable,
            }
            results.append(entry)

            if viable:
                discoveries.append(entry)

            memory_cleanup()

        except Exception as e:
            print(f"{smi:50s} | ERROR: {e}")
            continue

    elapsed = time.time() - start_time

    # Sort results by score
    results.sort(key=lambda x: x["total_score"], reverse=True)
    discoveries.sort(key=lambda x: x["total_score"], reverse=True)

    # Save outputs
    # 1. discovery_results_final.json
    with open("discovery_results_final.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DISCOVERY] Saved {len(results)} results to discovery_results_final.json")

    # 2. top_discoveries.smi
    with open("top_discoveries.smi", "w") as f:
        f.write("# Project Aurelius v5.2 — Top Discoveries (Score >= 65.0)\n")
        for d in discoveries:
            f.write(f"{d['smiles']}  # score={d['total_score']:.1f}\n")
    print(f"[DISCOVERY] Saved {len(discoveries)} discoveries to top_discoveries.smi")

    # 3. agent_state.json
    agent_state = {
        "timestamp": datetime.now(UTC).isoformat(),
        "total_screened": len(results),
        "viable_count": len(discoveries),
        "best_score": max(r["total_score"] for r in results) if results else 0.0,
        "screening_time_seconds": elapsed,
        "candidates_generated": len(all_candidates),
        "discoveries": discoveries[:20],
    }
    with open("agent_state.json", "w") as f:
        json.dump(agent_state, f, indent=2)
    print("[DISCOVERY] Saved agent_state.json")

    # Summary
    print(f"\n{'=' * 70}")
    print("  DISCOVERY SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Candidates screened: {len(results)}")
    print(f"  Viable discoveries:  {len(discoveries)}")
    print(f"  Screening time:      {elapsed:.1f}s")

    if results:
        print("\n  Top 10 Molecules by Score:")
        for i, r in enumerate(results[:10], 1):
            v = "VIABLE" if r["is_viable"] else "rejected"
            print(
                f"    {i}. {r['smiles'][:45]:45s} Score={r['total_score']:5.1f} Homog={r['sei_homogeneity_score']:5.1f}  ({v})"
            )

    if discoveries:
        print("\n  Viable Discoveries (Score >= 65.0):")
        for i, d in enumerate(discoveries[:10], 1):
            print(
                f"    {i}. {d['smiles'][:45]:45s} Score={d['total_score']:5.1f} Homog={d['sei_homogeneity_score']:5.1f}"
            )
    else:
        print("\n  No molecules achieved viability threshold (65.0).")
        print("  Highest homogeneity candidates:")
        h_sorted = sorted(results, key=lambda x: x["sei_homogeneity_score"], reverse=True)
        for i, r in enumerate(h_sorted[:5], 1):
            print(f"    {i}. {r['smiles'][:45]:45s} Homog={r['sei_homogeneity_score']:5.1f}")

    print(f"\n{'=' * 70}")
    return results, discoveries


if __name__ == "__main__":
    run_discovery()
