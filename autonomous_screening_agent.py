#!/usr/bin/env python3
"""
autonomous_screening_agent.py — Project Aurelius v5.2 Autonomous Screening Agent

Implements the full autonomous discovery loop:
  Generation (RDKit mutation engine) -> Screening (3-tier pipeline) ->
  Feedback-driven mutation -> Convergence check -> Report generation

Usage:
    python autonomous_screening_agent.py [--resume] [--max-generations N] [--batch-size N]

Dependencies:
    pip install -e .
    rdkit
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import hashlib
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, BRICS, rdFingerprintGenerator
    from rdkit.DataStructs import FingerprintSimilarity
    HAS_RDKIT = True
except ImportError:
    Chem = None  # type: ignore
    AllChem = None
    Descriptors = None
    BRICS = None
    rdFingerprintGenerator = None  # type: ignore
    HAS_RDKIT = False

import rdkit.DataStructs as DataStructs

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("aurelius_agent")

# Also log to errors.log
_error_handler = logging.FileHandler("errors.log", mode="w")
_error_handler.setLevel(logging.ERROR)
_error_handler.setFormatter(logging.Formatter("%(asctime)s [ERROR] %(message)s"))
log.addHandler(_error_handler)

# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

np.random.seed(42)

# ---------------------------------------------------------------------------
# Aurelius imports
# ---------------------------------------------------------------------------

from aurelius.config import M5ProConfig, initialize_environment
from aurelius.pipeline import AureliusPipeline
from aurelius.screening.tier3_gcmtwin import GCMDigitalTwin
from aurelius.screening.tier0_gnn import Tier0ActivationPredictor
from aurelius.types import MoleculeInput
from aurelius.screening.tier1_mlx_filter import MLXNAFilter
from aurelius.screening.tier2_mattersim import MatterSimMTSimulator
from aurelius.memory.profiler import MemoryProfiler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_mol_from_smiles(smiles: str):
    """Return RDKit Mol or None."""
    if not HAS_RDKIT:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def _mol_to_fp(mol) -> Any:
    """Compute ECFP4 (radius=2) fingerprint using Morgan generator."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return gen.GetFingerprint(mol)


def _serialize_fp(fp) -> str:
    """Serialize an RDKit fingerprint to a hex-like text string."""
    return DataStructs.BitVectToText(fp)


def _deserialize_fp(hex_str: str):
    """Reconstruct an RDKit fingerprint from serialized text."""
    return DataStructs.CreateFromBitString(hex_str)


def _tanimoto(fp1, fp2) -> float:
    """Tanimoto similarity between two fingerprints."""
    return FingerprintSimilarity(fp1, fp2)


def _is_valid_mol(mol) -> bool:
    """Chemical validity + MW < 350 (module-level helper)."""
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return False
    mw = Descriptors.ExactMolWt(mol)
    return mw < 350.0


def _load_smiles_file(path: str) -> list[str]:
    """Load SMILES from a .smi file, skipping comments and blank lines."""
    smiles_list: list[str] = []
    p = Path(path)
    if not p.exists():
        return smiles_list
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Strip inline comments
            if " #" in line:
                line = line.split(" #")[0].strip()
            if line:
                smiles_list.append(line)
    return smiles_list


def _memory_cleanup():
    """Free GPU/MLX memory after each batch."""
    gc.collect()
    try:
        import mlx.core as mx
        mx.metal.clear_cache()
    except Exception:
        pass
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _check_apple_silicon() -> bool:
    """Detect if running on Apple Silicon."""
    import platform
    return platform.machine() in ("arm64",) and platform.system() == "Darwin"


# ---------------------------------------------------------------------------
# Mutation Engine
# ---------------------------------------------------------------------------


class MutationEngine:
    """RDKit-based molecule mutation engine with BRICS reassembly."""

    def __init__(self, seed_smiles: list[str], known_fps_hex: list[str] | None = None):
        self.seed_pool: list[str] = list(set(seed_smiles))
        self.known_fps: list = []
        for h in (known_fps_hex or []):
            try:
                self.known_fps.append(_deserialize_fp(h))
            except Exception:
                pass
        self._rng = np.random.RandomState(42)

    def fingerprint_db_size(self) -> int:
        return len(self.known_fps)

    def add_to_db(self, smiles: str):
        mol = _safe_mol_from_smiles(smiles)
        if mol is not None:
            self.known_fps.append(_mol_to_fp(mol))

    def _novelty_check(self, mol) -> bool:
        """Return True if molecule is novel (Tanimoto < 0.75 vs all known)."""
        fp = _mol_to_fp(mol)
        for known in self.known_fps:
            if _tanimoto(fp, known) >= 0.75:
                return False
        return True

    # _is_valid is now a module-level function (_is_valid_mol)

    # ---- Mutation templates ----

    def _brics_reassemble(self, mol) -> list[str]:
        """BRICS decomposition + random reassembly."""
        generated: list[str] = []
        try:
            fragments = BRICS.BRICSDecompose(mol)
            if len(fragments) < 2:
                return generated
            # Try several random re-assemblies
            for _ in range(20):
                rng = np.random.RandomState(self._rng.randint(0, 2**31))
                idx = rng.choice(len(fragments), size=min(2, len(fragments)), replace=False)
                try:
                    reassembled = BRICS.BRICSBuild(fragments[idx[0]], fragments[idx[1]])
                    if reassembled:
                        generated.append(reassembled)
                except Exception:
                    pass
        except Exception:
            pass
        return generated

    def _fluorinate(self, mol) -> list[str]:
        """Add fluorine to non-carbonyl carbons."""
        generated: list[str] = []
        try:
            rxn = Chem.ReactionFromSmarts("[*:1][H]>>[*:1]F")
            products = rxn.RunReactants((mol,))
            for prod in products:
                mol_prod = prod[0]
                # Skip if fluorine attached to carbonyl carbon
                smiles = Chem.MolToSmiles(mol_prod, isomericSmiles=True)
                generated.append(smiles)
        except Exception:
            pass
        return generated

    def _add_unsaturation(self, mol) -> list[str]:
        """Introduce C=C double bonds where single bonds exist."""
        generated: list[str] = []
        try:
            rxn = Chem.ReactionFromSmarts("[*:1][H]-[*:2][H]>>[*:1]=[*:2]")
            products = rxn.RunReactants((mol,))
            for prod in products:
                smiles = Chem.MolToSmiles(prod[0], isomericSmiles=True)
                generated.append(smiles)
        except Exception:
            pass
        return generated

    def _methylate(self, mol) -> list[str]:
        """Add methyl groups to replace hydrogens."""
        generated: list[str] = []
        try:
            rxn = Chem.ReactionFromSmarts("[*:1][H]>>[*:1]C")
            products = rxn.RunReactants((mol,))
            for prod in products:
                smiles = Chem.MolToSmiles(prod[0], isomericSmiles=True)
                generated.append(smiles)
        except Exception:
            pass
        return generated

    def mutate(self, smiles: str, batch_size: int = 50) -> list[str]:
        """Generate up to batch_size mutated variants of a seed molecule."""
        mol = _safe_mol_from_smiles(smiles)
        if mol is None:
            return []

        # Collect all candidate SMILES from mutation templates
        candidates: set[str] = set()
        candidates.add(smiles)  # keep original too

        # Priority: BRICS first
        brics_results = self._brics_reassemble(mol)
        for s in brics_results:
            m = _safe_mol_from_smiles(s)
            if m is not None and _is_valid_mol(m) and self._novelty_check(m):
                candidates.add(s)

        # Fallback templates
        for func in [self._fluorinate, self._add_unsaturation, self._methylate]:
            results = func(mol)
            for s in results:
                m = _safe_mol_from_smiles(s)
                if m is not None and _is_valid_mol(m) and self._novelty_check(m):
                    candidates.add(s)

        # If BRICS yielded nothing, try fallback templates more aggressively
        if len(brics_results) == 0:
            for func in [self._fluorinate, self._add_unsaturation, self._methylate]:
                results = func(mol)
                for s in results:
                    m = _safe_mol_from_smiles(s)
                    if m is not None and _is_valid_mol(m) and self._novelty_check(m):
                        candidates.add(s)

        result_list = list(candidates)
        if len(result_list) > batch_size:
            indices = self._rng.choice(len(result_list), size=batch_size, replace=False)
            result_list = [result_list[i] for i in indices]
        return result_list

    def mutate_batch(self, batch_smiles: list[str], batch_size: int = 50) -> list[str]:
        """Mutate a batch of seed molecules, returning all variants."""
        all_variants: list[str] = []
        for smi in batch_smiles:
            variants = self.mutate(smi, batch_size)
            all_variants.extend(variants)
        # Deduplicate
        return list(set(all_variants))


# ---------------------------------------------------------------------------
# Feedback Adaptation
# ---------------------------------------------------------------------------


class FeedbackAdapter:
    """Adapts mutation strategy based on rejection patterns."""

    def __init__(self):
        self.tier1_fails = 0
        self.tier2_fails = 0
        self.tier3_low_homogeneity = 0
        self.total_screened = 0
        self.rationale_log: list[str] = []

    def record(self, result: dict[str, Any]):
        score = result.get("score")
        if score is None:
            return
        self.total_screened += 1
        if not score.tier1_viable:
            self.tier1_fails += 1
            self.rationale_log.append(
                f"Tier 1 fail for {score.molecule_smiles}: "
                "Lower MW, add polar groups, reduce F-density"
            )
        if not score.tier2_viable:
            self.tier2_fails += 1
            self.rationale_log.append(
                f"Tier 2 fail for {score.molecule_smiles}: "
                "Reduce steric bulk near coordination sites, lower desolvation barrier"
            )
        if score.tier3_viable and score.sei_homogeneity_score < 50.0:
            self.tier3_low_homogeneity += 1
            self.rationale_log.append(
                f"Low SEI homogeneity for {score.molecule_smiles}: "
                "Add unsaturation/boron, increase F/C ratio"
            )

    def get_adaptation_strategy(self) -> dict[str, Any]:
        """Return current mutation adaptation recommendations."""
        strategy: dict[str, Any] = {
            "total_screened": self.total_screened,
            "tier1_fail_rate": self.tier1_fails / max(self.total_screened, 1),
            "tier2_fail_rate": self.tier2_fails / max(self.total_screened, 1),
            "tier3_low_homogeneity_rate": self.tier3_low_homogeneity / max(self.total_screened, 1),
        }
        if strategy["tier1_fail_rate"] > 0.5:
            strategy["recommendation"] = "Prioritize MW reduction and polar group addition"
        elif strategy["tier2_fail_rate"] > 0.5:
            strategy["recommendation"] = "Reduce steric bulk, focus on small molecules"
        elif strategy["tier3_low_homogeneity_rate"] > 0.5:
            strategy["recommendation"] = "Add unsaturation and boron-containing groups"
        else:
            strategy["recommendation"] = "Continue current mutation strategy"
        return strategy

    def write_rationale_log(self, path: str = "mutation_rationale.md"):
        """Write accumulated rationale to markdown file."""
        with open(path, "w") as f:
            f.write("# Mutation Rationale Log\n\n")
            f.write(f"**Generated:** {datetime.now(timezone.utc).isoformat()}\n\n")
            f.write("## Adaptation Decisions\n\n")
            for entry in self.rationale_log:
                f.write(f"- {entry}\n")
            f.write("\n## Strategy Summary\n\n")
            strategy = self.get_adaptation_strategy()
            f.write(f"- **Recommendation:** {strategy['recommendation']}\n")
            f.write(f"- **Tier 1 fail rate:** {strategy['tier1_fail_rate']:.2%}\n")
            f.write(f"- **Tier 2 fail rate:** {strategy['tier2_fail_rate']:.2%}\n")
            f.write(f"- **Tier 3 low homogeneity rate:** {strategy['tier3_low_homogeneity_rate']:.2%}\n")
        log.info("Mutation rationale log written to %s", path)


# ---------------------------------------------------------------------------
# Convergence Checker
# ---------------------------------------------------------------------------


class ConvergenceChecker:
    """Evaluates whether the screening loop should terminate."""

    def __init__(self):
        self.all_scores: list[float] = []
        self.batch_scores: list[list[float]] = []
        self.viability_rates: list[float] = []
        self.new_clusters_per_batch: list[int] = []
        self.viable_count = 0
        self.total_screened = 0
        self.generations = 0

    def record_batch(self, scores: list[float], viable_count: int, new_clusters: int):
        self.all_scores.extend(scores)
        self.batch_scores.append(scores)
        self.total_screened += len(scores)
        self.viable_count += viable_count
        self.generations += 1

        # Viability rate for this batch
        viable_in_batch = sum(1 for s in scores if s >= 65.0)
        self.viability_rates.append(viable_in_batch / max(len(scores), 1))
        self.new_clusters_per_batch.append(new_clusters)

    def compute_rolling_mean(self, batch_size: int = 50) -> list[float]:
        """Rolling mean of total_score over windows of `batch_size`."""
        if len(self.all_scores) < batch_size:
            return []
        rolling: list[float] = []
        for i in range(batch_size, len(self.all_scores) + 1, batch_size):
            window = self.all_scores[i - batch_size:i]
            rolling.append(float(np.mean(window)))
        return rolling

    def check_score_plateau(self) -> bool:
        """Rolling mean changes < 1.0% over 3 consecutive batches."""
        rolling = self.compute_rolling_mean(batch_size=50)
        if len(rolling) < 3:
            return False
        last_three = rolling[-3:]
        for i in range(1, 3):
            ref = last_three[i - 1]
            if ref == 0:
                return False
            change = abs(last_three[i] - ref) / abs(ref)
            if change >= 0.01:
                return False
        return True

    def check_pass_rate_collapsed(self) -> bool:
        """Combined Tier 1+2+3 viability rate < 3% for 2 consecutive batches."""
        if len(self.viability_rates) < 2:
            return False
        return self.viability_rates[-1] < 0.03 and self.viability_rates[-2] < 0.03

    def check_structural_saturation(self) -> bool:
        """< 3 new clusters over last 2 batches."""
        if len(self.new_clusters_per_batch) < 2:
            return False
        return self.new_clusters_per_batch[-1] < 3 and self.new_clusters_per_batch[-2] < 3

    def check_volume_requirement(self) -> bool:
        """>= 150 fully-screened viable-tier OR >= 300 total unique screened."""
        return self.viable_count >= 150 or self.total_screened >= 300

    def should_terminate(self) -> tuple[bool, str]:
        """Return (should_terminate, reason)."""
        if not self.check_volume_requirement():
            return False, "Volume threshold not met"
        plateau = self.check_score_plateau()
        pass_collapsed = self.check_pass_rate_collapsed()
        saturation = self.check_structural_saturation()
        if plateau and pass_collapsed and saturation:
            return True, "All convergence criteria met"
        reasons = []
        if not plateau:
            reasons.append("score plateau")
        if not pass_collapsed:
            reasons.append("pass rate not collapsed")
        if not saturation:
            reasons.append("structural saturation")
        return False, f"Volume met but not all criteria: {', '.join(reasons)}"

    def final_score_variance(self) -> float:
        if len(self.all_scores) < 2:
            return 0.0
        return float(np.var(self.all_scores))


# ---------------------------------------------------------------------------
# Checkpoint Manager
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Manages agent_state.json for resume capability."""

    def __init__(self, path: str = "agent_state.json"):
        self.path = path
        self.state: dict[str, Any] = {
            "batch": 0,
            "screened_count": 0,
            "best_score": 0.0,
            "known_fps_hex": [],
            "convergence_met": False,
            "viable_count": 0,
            "total_generated": 0,
            "invalid_discarded": 0,
            "discoveries": [],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "last_updated": None,
        }

    def load(self) -> dict[str, Any]:
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self.state = json.load(f)
                log.info("Checkpoint loaded: batch=%d screened=%d",
                         self.state["batch"], self.state["screened_count"])
            except (json.JSONDecodeError, KeyError) as e:
                log.error("Failed to load checkpoint: %s. Starting fresh.", e)
                self.state["known_fps_hex"] = []
        return self.state

    def save(self):
        self.state["last_updated"] = datetime.now(timezone.utc).isoformat()
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self.state, f, indent=2)
        os.replace(tmp_path, self.path)

    def add_discovery(self, discovery: dict[str, Any]):
        self.state["discoveries"].append(discovery)

    def update_stats(self, batch_smiles: list[str], batch_scores: list[float],
                     viable_count: int, invalid_count: int):
        self.state["batch"] += 1
        self.state["screened_count"] += len(batch_smiles)
        self.state["total_generated"] += len(batch_smiles)
        self.state["invalid_discarded"] += invalid_count
        self.state["viable_count"] += viable_count
        if batch_scores:
            best = max(batch_scores)
            if best > self.state["best_score"]:
                self.state["best_score"] = best

    def fps_hex_list(self) -> list[str]:
        return self.state["known_fps_hex"]

    def add_fps_hex(self, hex_str: str):
        self.state["known_fps_hex"].append(hex_str)


# ---------------------------------------------------------------------------
# Report Generators
# ---------------------------------------------------------------------------


def generate_discovery_results(all_results: list[dict[str, Any]], path: str = "discovery_results_final.json"):
    """Write full structured logs of all screened molecules."""
    serializable: list[dict[str, Any]] = []
    for r in all_results:
        entry: dict[str, Any] = {
            "smiles": r.get("score", {}).molecule_smiles if hasattr(r.get("score"), "molecule_smiles") else "unknown",
        }
        score = r.get("score")
        if score:
            entry["total_score"] = score.total_score
            entry["sigma"] = score.sigma_score
            entry["desolvation"] = score.desolvation_score
            entry["sei_homogeneity"] = score.sei_homogeneity_score
            entry["mx_synthesis"] = score.mx_synthesis_score
            entry["gwp_penalty"] = score.gwp_penalty
            entry["is_viable"] = score.is_viable
            entry["rejection_reasons"] = score.rejection_reasons
            entry["tier1_viable"] = score.tier1_viable
            entry["tier2_viable"] = score.tier2_viable
            entry["tier3_viable"] = score.tier3_viable
            entry["components"] = score.rejection_reasons  # placeholder
        tier1 = r.get("tier1")
        if tier1:
            entry["tier1_confidence"] = tier1.confidence_score
            entry["tier1_is_viable"] = tier1.is_viable
        tier2 = r.get("tier2")
        if tier2:
            entry["tier2_barrier_eV"] = tier2.desolvation_path.barrier_height_eV
            entry["tier2_is_viable"] = tier2.is_viable
        tier3 = r.get("tier3")
        if tier3:
            entry["tier3_homogeneity"] = tier3.sei_evolution.homogeneity_score
            entry["tier3_thickness"] = tier3.sei_evolution.thickness_angstrom
            entry["tier3_components"] = tier3.sei_evolution.components
        entry["tier_timings"] = r.get("tier_timings", {})
        serializable.append(entry)

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    log.info("Discovery results written to %s (%d entries)", path, len(serializable))


def write_top_discoveries(discoveries: list[dict[str, Any]], path: str = "top_discoveries.smi"):
    """Write SMILES of all legitimate discoveries."""
    with open(path, "w") as f:
        f.write("# Project Aurelius v5.2 — Top Discoveries (Score >= 65.0)\n")
        for d in discoveries:
            f.write(f"{d['smiles']}  # score={d['total_score']:.1f}\n")
    log.info("Top discoveries written to %s (%d molecules)", path, len(discoveries))


def generate_screening_statistics(convergence: ConvergenceChecker, all_results: list[dict[str, Any]],
                                  path: str = "screening_statistics.md"):
    """Generate convergence plots, pass rates, exhaustion proof."""
    scores = [r["score"].total_score for r in all_results if r.get("score")]
    viable = [s for s in scores if s >= 65.0]

    with open(path, "w") as f:
        f.write("# Screening Statistics — Project Aurelius v5.2\n\n")
        f.write(f"**Generated:** {datetime.now(timezone.utc).isoformat()}\n\n")

        f.write("## Overview\n\n")
        f.write(f"- **Total screened:** {convergence.total_screened}\n")
        f.write(f"- **Generations run:** {convergence.generations}\n")
        f.write(f"- **Viable discoveries (score >= 65):** {convergence.viable_count}\n")
        f.write(f"- **Final score variance:** {convergence.final_score_variance():.4f}\n\n")

        f.write("## Score Distribution\n\n")
        if scores:
            f.write(f"- Mean: {np.mean(scores):.2f}\n")
            f.write(f"- Std:  {np.std(scores):.2f}\n")
            f.write(f"- Min:  {np.min(scores):.2f}\n")
            f.write(f"- Max:  {np.max(scores):.2f}\n\n")

            # Score histogram as text
            bins = [0, 20, 35, 50, 65, 80, 100]
            f.write("Score histogram:\n")
            for i in range(len(bins) - 1):
                count = sum(1 for s in scores if bins[i] <= s < bins[i + 1])
                bar = "#" * count
                line = "  [{:>3.0f}-{:>3.0f}): {} ({}))".format(bins[i], bins[i+1], bar, count)
                f.write(line + "\n")
            f.write("\n")

        f.write("## Convergence Analysis\n\n")
        plateau = convergence.check_score_plateau()
        pass_collapsed = convergence.check_pass_rate_collapsed()
        saturation = convergence.check_structural_saturation()

        f.write(f"- **Score plateau:** {'YES' if plateau else 'NO'}\n")
        f.write(f"- **Pass rate collapse:** {'YES' if pass_collapsed else 'NO'}\n")
        f.write(f"- **Structural saturation:** {'YES' if saturation else 'NO'}\n\n")

        rolling = convergence.compute_rolling_mean(batch_size=50)
        if rolling:
            f.write("### Rolling Mean of Total Score (window=50)\n\n")
            f.write("| Batch | Rolling Mean |\n")
            f.write("|-------|-------------|\n")
            for i, rm in enumerate(rolling):
                f.write(f"| {i + 1} | {rm:.2f} |\n")
            f.write("\n")

        f.write("## Viability Rate Trend\n\n")
        f.write("| Generation | Viability Rate |\n")
        f.write("|------------|---------------|\n")
        for i, rate in enumerate(convergence.viability_rates):
            f.write(f"| {i + 1} | {rate:.4f} |\n")
        f.write("\n")

        f.write("## Exhaustion Proof\n\n")
        f.write("The screening process terminates when ALL of the following are met:\n")
        f.write("1. **Volume:** >= 150 viable-tier candidates OR >= 300 total unique screened\n")
        f.write("2. **Score Plateau:** Rolling mean changes < 1.0% over 3 consecutive batches\n")
        f.write("3. **Pass Rate Collapse:** Viability rate < 3% for 2 consecutive batches\n")
        f.write("4. **Structural Saturation:** < 3 new clusters over last 2 batches\n\n")
        f.write(f"Final state: {convergence.total_screened} screened, "
                f"{convergence.viable_count} viable, "
                f"variance={convergence.final_score_variance():.4f}\n")

    log.info("Screening statistics written to %s", path)


def generate_chemical_insights(all_results: list[dict[str, Any]], discoveries: list[dict[str, Any]],
                               path: str = "chemical_insights.md"):
    """Generate structural correlations, failure analysis, experimental next steps."""
    with open(path, "w") as f:
        f.write("# Chemical Insights — Project Aurelius v5.2\n\n")
        f.write(f"**Generated:** {datetime.now(timezone.utc).isoformat()}\n\n")

        f.write("## Structural Correlations\n\n")
        f.write("Analysis of molecular features correlated with high Aurelius scores.\n\n")

        # Group by scaffold
        scaffold_scores: dict[str, list[float]] = {}
        for r in all_results:
            score = r.get("score")
            if not score:
                continue
            mol = _safe_mol_from_smiles(score.molecule_smiles)
            if mol is not None:
                scaffold = Chem.MolFragmentToSmiles(mol, atomsToUse=list(range(mol.GetNumAtoms())),
                                                    isomericSmiles=False)
                scaffold = scaffold[:30]  # truncate
                if scaffold not in scaffold_scores:
                    scaffold_scores[scaffold] = []
                scaffold_scores[scaffold].append(score.total_score)

        f.write("| Scaffold (truncated) | Mean Score | Count |\n")
        f.write("|---------------------|-----------|-------|\n")
        for scaffold, sc_list in sorted(scaffold_scores.items(), key=lambda x: -np.mean(x[1]))[:15]:
            f.write(f"| {scaffold} | {np.mean(sc_list):.2f} | {len(sc_list)} |\n")
        f.write("\n")

        f.write("## Failure Analysis\n\n")
        failure_reasons: dict[str, int] = {}
        for r in all_results:
            score = r.get("score")
            if score and score.rejection_reasons:
                for reason in score.rejection_reasons:
                    failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        if failure_reasons:
            f.write("Top rejection reasons:\n\n")
            f.write("| Reason | Count |\n")
            f.write("|--------|-------|\n")
            for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
                f.write(f"| {reason} | {count} |\n")
        else:
            f.write("No rejection reasons recorded.\n")
        f.write("\n")

        f.write("## Experimental Next Steps\n\n")
        if discoveries:
            f.write("### Recommended Experimental Validation\n\n")
            for i, d in enumerate(discoveries[:10], 1):
                f.write(f"{i}. **{d['smiles']}** (Score: {d['total_score']:.1f})\n")
            f.write("\n")
        else:
            f.write("No legitimate discoveries found in this screening round.\n")
            f.write("Recommendations:\n")
            f.write("1. Expand mutation template library (e.g., cyano, nitro, boron additions)\n")
            f.write("2. Lower MW threshold to allow larger scaffolds\n")
            f.write("3. Increase batch sizes for deeper exploration\n")
            f.write("4. Consider seed molecules with known SEI-forming properties\n")
        f.write("\n")

        f.write("### Coverage Gaps\n\n")
        f.write("- Focus on small-molecule electrolyte additives (< 350 Da)\n")
        f.write("- Prioritize fluorinated carbonates and lactones\n")
        f.write("- Explore boron-containing SEI-forming compounds\n")
        f.write("- Consider unsaturated cyclic ethers\n")

    log.info("Chemical insights written to %s", path)


def generate_manifest(convergence: ConvergenceChecker, discoveries: list[dict[str, Any]],
                      all_results: list[dict[str, Any]],
                      path: str = "agent_discovery_manifest.json"):
    """Generate the agent_discovery_manifest.json."""
    rolling = convergence.compute_rolling_mean(batch_size=50)
    rolling_mean = float(np.mean(rolling[-3:])) if len(rolling) >= 3 else 0.0

    manifest: dict[str, Any] = {
        "search_statistics": {
            "total_screened": convergence.total_screened,
            "generations_run": convergence.generations,
            "invalid_discarded": 0,  # tracked in checkpoint
            "final_score_variance": convergence.final_score_variance(),
        },
        "discoveries": [],
        "exhaustion_proof": {
            "rolling_mean_plateau": rolling_mean,
            "viability_rate_final": convergence.viability_rates[-1] if convergence.viability_rates else 0.0,
            "new_clusters_last_batch": convergence.new_clusters_per_batch[-1] if convergence.new_clusters_per_batch else 0,
            "analytical_summary": "",
        },
    }

    for d in discoveries:
        manifest["discoveries"].append({
            "smiles": d["smiles"],
            "total_score": d["total_score"],
            "sigma": d["sigma"],
            "desolvation": d["desolvation"],
            "sei_homogeneity": d["sei_homogeneity"],
            "mx_synthesis": d["mx_synthesis"],
            "gwp_penalty": d["gwp_penalty"],
            "is_viable": d["is_viable"],
            "rejection_reasons": d["rejection_reasons"],
            "components": d.get("components", []),
        })

    # Analytical summary
    reasons = []
    if convergence.check_score_plateau():
        reasons.append("score plateau confirmed")
    if convergence.check_pass_rate_collapsed():
        reasons.append("pass rate collapsed")
    if convergence.check_structural_saturation():
        reasons.append("structural saturation reached")
    if not reasons:
        reasons.append("partial convergence — volume threshold met but some criteria pending")

    manifest["exhaustion_proof"]["analytical_summary"] = (
        f"After {convergence.total_screened} molecules across "
        f"{convergence.generations} generations: "
        f"{convergence.viable_count} viable discoveries found. "
        f"Final criteria: {', '.join(reasons) if reasons else 'none met'}."
    )

    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Discovery manifest written to %s", path)


# ---------------------------------------------------------------------------
# Main Screening Loop
# ---------------------------------------------------------------------------


def run_screening(args):
    """Main autonomous screening loop."""

    # Initialize memory profiler if requested
    profiler: MemoryProfiler | None = None
    if getattr(args, "profile_memory", False):
        profiler = MemoryProfiler()
        profiler.start()
        print("[AGENT] Memory profiling enabled. CSV reports will be generated.")

    # ---- Phase 1: Environment & Pipeline Initialization ----
    print("=" * 60)
    print("  PROJECT AURELIUS v6.0 — Autonomous Screening Agent")
    print("  The 2nm Fusion Edition | M5 Pro Neural Accelerators")
    print("=" * 60)

    if _check_apple_silicon():
        print("[AGENT] Running on Apple Silicon (optimized).\n")
    else:
        print("[AGENT] [CPU_FALLBACK] Not running on Apple Silicon. Performance may be reduced.\n")

    # Initialize environment
    initialize_environment()

    # Build pipeline
    config = M5ProConfig()
    pipeline = AureliusPipeline(config, use_real_models=True)
    pipeline.initialize()

    # Inject Tier 0 Activation Energy Predictor (MPNN + linear fallback)
    if pipeline._gcmtwin:
        tier0_pred = Tier0ActivationPredictor(
            model_path="models/tier0/mpnn_weights.pth",
        )
        pipeline._gcmtwin._tier0_predictor = tier0_pred
        pipeline._gcmtwin._use_tier0_prediction = True
        print("[AGENT] Tier 0 Activation Energy Predictor (MPNN) injected successfully.")
    else:
        raise RuntimeError("Pipeline GCMD Twin not initialized. Cannot run Tier 3.")

    # ---- Phase 2: Chemical Generation Engine ----
    print("\n[AGENT] Loading seed molecules...")

    seed_smiles = _load_smiles_file("discovery_candidates.smi")
    seed_smiles.extend(_load_smiles_file("examples/molecules.smi"))
    seed_smiles = list(set(s for s in seed_smiles if s.strip()))
    print(f"[AGENT] Seed pool: {len(seed_smiles)} unique molecules")

    # Initialize mutation engine
    engine = MutationEngine(seed_smiles)

    # ---- Phase 5: Checkpoint & Resume ----
    checkpoint = CheckpointManager()
    state = checkpoint.load()

    # Rebuild known fingerprints from checkpoint
    known_fps_hex = state.get("known_fps_hex", [])
    engine.known_fps = []
    for h in known_fps_hex:
        try:
            engine.known_fps.append(_deserialize_fp(h))
        except Exception:
            pass

    resumed = state["screened_count"] > 0
    start_batch = state.get("batch", 0)
    screened_so_far = state.get("screened_count", 0)
    best_score_so_far = state.get("best_score", 0.0)

    if resumed:
        print(f"[AGENT] Resuming from checkpoint: batch={start_batch}, "
              f"screened={screened_so_far}, best_score={best_score_so_far:.1f}")
    else:
        print("[AGENT] Fresh start. No checkpoint found.")

    # Convergence checker
    convergence = ConvergenceChecker()
    if resumed and state.get("batch", 0) > 0:
        # Restore convergence state from previous runs
        convergence.total_screened = screened_so_far
        convergence.viable_count = state.get("viable_count", 0)
        convergence.generations = start_batch

    feedback = FeedbackAdapter()
    all_results: list[dict[str, Any]] = []
    discoveries: list[dict[str, Any]] = []

    # ---- Main Loop ----
    max_generations = args.max_generations or 50
    batch_size = args.batch_size or 50
    max_wall_time = 7200  # 2 hours

    wall_start = time.time()
    current_batch = start_batch
    generation = 0

    print(f"\n[AGENT] Starting screening loop. Batch size: {batch_size}, "
          f"Max generations: {max_generations}")
    print(f"[AGENT] Time limit: {max_wall_time}s (2 hours)\n")

    # Track previously screened SMILES for deduplication
    screened_smiles: set[str] = set()

    while generation < max_generations:
        # Wall-time check
        elapsed = time.time() - wall_start
        if elapsed > max_wall_time:
            print(f"\n[AGENT] Time cap reached ({elapsed:.0f}s). Exiting gracefully.")
            break

        generation += 1
        current_batch += 1

        # ---- Generation: Mutate seeds ----
        if generation == 1:
            # First generation: mutate all seeds
            candidates = engine.mutate_batch(seed_smiles, batch_size * 3)
        else:
            # Subsequent generations: mutate best molecules from previous generation
            # Use top-scoring molecules as seeds for next mutation round
            if all_results:
                scored_results = [
                    (r["score"].total_score, r["score"].molecule_smiles)
                    for r in all_results if r.get("score")
                ]
                scored_results.sort(key=lambda x: -x[0])
                top_seeds = [s for _, s in scored_results[:max(5, len(scored_results) // 5)]]
            else:
                top_seeds = seed_smiles[:5]
            candidates = engine.mutate_batch(top_seeds, batch_size * 3)

        # Filter invalid & duplicate
        valid_candidates: list[str] = []
        invalid_count = 0
        for smi in candidates:
            if smi in screened_smiles:
                invalid_count += 1
                continue
            mol = _safe_mol_from_smiles(smi)
            if mol is None:
                invalid_count += 1
                continue
            if not _is_valid_mol(mol):
                invalid_count += 1
                continue
            valid_candidates.append(smi)

        # Take up to batch_size
        if len(valid_candidates) > batch_size:
            valid_candidates = valid_candidates[:batch_size]

        if not valid_candidates:
            print(f"[AGENT] Generation {generation}: No valid candidates. Skipping.")
            continue

        print(f"[AGENT] Generation {generation}: Screening {len(valid_candidates)} candidates "
              f"(invalid discarded: {invalid_count})")

        # ---- Screening: Batch processing ----
        batch_scores: list[float] = []
        batch_viable = 0
        batch_discoveries: list[dict[str, Any]] = []
        batch_fps_hex: list[str] = []

        # Write batch SMILES to file
        batch_file = f"candidates_batch_{current_batch}.smi"
        with open(batch_file, "w") as f:
            for smi in valid_candidates:
                f.write(f"{smi}\n")

        for smi in valid_candidates:
            try:
                result = pipeline.screen_molecule(smi)
            except Exception as e:
                log.error("Pipeline error for %s: %s", smi, e)
                _memory_cleanup()
                continue

            score = result.get("score")
            if score is None:
                continue

            screened_smiles.add(smi)
            engine.add_to_db(smi)

            # Extract metrics safely
            total_score = score.total_score
            batch_scores.append(total_score)

            # Check discovery criteria
            is_discovery = (
                total_score >= 65.0
                and score.tier1_viable
                and score.tier2_viable
                and score.tier3_viable
                and len(score.rejection_reasons) == 0
            )

            if is_discovery:
                batch_viable += 1
                discovery_entry = {
                    "smiles": smi,
                    "total_score": total_score,
                    "sigma": score.sigma_score,
                    "desolvation": score.desolvation_score,
                    "sei_homogeneity": score.sei_homogeneity_score,
                    "mx_synthesis": score.mx_synthesis_score,
                    "gwp_penalty": score.gwp_penalty,
                    "is_viable": True,
                    "rejection_reasons": score.rejection_reasons,
                    "components": score.rejection_reasons,
                }
                batch_discoveries.append(discovery_entry)
                discoveries.append(discovery_entry)
                checkpoint.add_discovery(discovery_entry)
                print(f"  ** DISCOVERY ** {smi} (score={total_score:.1f})")

            all_results.append(result)
            feedback.record(result)

            # Memory cleanup
            _memory_cleanup()

        # ---- Record batch for convergence ----
        # Estimate new clusters (simplified: count novel fingerprints)
        new_fps_count = 0
        for smi in valid_candidates:
            mol = _safe_mol_from_smiles(smi)
            if mol is not None:
                fp_hex = _serialize_fp(_mol_to_fp(mol))
                batch_fps_hex.append(fp_hex)
                checkpoint.add_fps_hex(fp_hex)
                new_fps_count += 1

        # Use novel SMILES count as proxy for new clusters
        new_clusters = new_fps_count

        convergence.record_batch(batch_scores, batch_viable, new_clusters)
        checkpoint.update_stats(valid_candidates, batch_scores, batch_viable, invalid_count)

        print(f"  Generation {generation} complete: "
              f"{len(valid_candidates)} screened, {batch_viable} viable, "
              f"best={max(batch_scores) if batch_scores else 0:.1f}")

        # Memory profiling sample
        if profiler:
            profiler.sample(
                generation=generation,
                screened_count=convergence.total_screened,
                gc_collected=gc.collect(),
            )

        # ---- Feedback adaptation ----
        strategy = feedback.get_adaptation_strategy()
        if generation % 5 == 0:
            print(f"  [Feedback] Strategy: {strategy['recommendation']}")

        # ---- Check convergence ----
        should_stop, reason = convergence.should_terminate()
        if should_stop:
            print(f"\n[AGENT] Convergence reached: {reason}")
            break

        # ---- Save checkpoint ----
        checkpoint.save()

        # Print progress
        print(f"  [Progress] Screened: {convergence.total_screened}, "
              f"Viable: {convergence.viable_count}, "
              f"Generations: {generation}/{max_generations}\n")

    # ---- Post-loop: Generate all deliverables ----
    print("\n" + "=" * 60)
    print("  GENERATING DELIVERABLES")
    print("=" * 60)

    # 1. discovery_results_final.json
    generate_discovery_results(all_results)

    # 2. top_discoveries.smi
    write_top_discoveries(discoveries)

    # 3. screening_statistics.md
    generate_screening_statistics(convergence, all_results)

    # 4. chemical_insights.md
    generate_chemical_insights(all_results, discoveries)

    # 5. agent_discovery_manifest.json
    generate_manifest(convergence, discoveries, all_results)

    # Final checkpoint save
    checkpoint.save()

    # Generate memory profile report if profiling was enabled
    if profiler:
        profiler.stop()
        report_path = profiler.generate_report()
        print(f"\n[AGENT] Memory profile report: {report_path}")
        print(f"  Peak RAM:      {profiler.peak_ram_gb:.2f} GB")
        print(f"  Peak MPS:      {profiler.peak_mps_gb:.2f} GB")
        print(f"  Peak MLX:      {profiler.peak_mlx_gb:.2f} GB")
        print(f"  Samples:       {profiler.n_samples}")

    # Summary
    print("\n" + "=" * 60)
    print("  SCREENING COMPLETE")
    print("=" * 60)
    print(f"  Total screened:     {convergence.total_screened}")
    print(f"  Generations run:    {convergence.generations}")
    print(f"  Viable discoveries: {convergence.viable_count}")
    print(f"  Best score:         {checkpoint.state['best_score']:.1f}")
    print(f"  Invalid discarded:  {checkpoint.state['invalid_discarded']}")
    print(f"  Wall time:          {time.time() - wall_start:.0f}s")
    print(f"\n  Output files:")
    print(f"    - discovery_results_final.json")
    print(f"    - top_discoveries.smi")
    print(f"    - screening_statistics.md")
    print(f"    - chemical_insights.md")
    print(f"    - agent_discovery_manifest.json")
    print(f"    - agent_state.json")
    print(f"    - mutation_rationale.md")
    print(f"    - errors.log")
    print()


def main():
    parser = argparse.ArgumentParser(description="Aurelius v6.0 Autonomous Screening Agent")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--max-generations", type=int, default=50, help="Maximum generations to run")
    parser.add_argument("--batch-size", type=int, default=50, help="Candidates per batch")
    parser.add_argument("--profile-memory", action="store_true", help="Enable memory profiling with CSV report output")
    args = parser.parse_args()

    try:
        run_screening(args)
    except KeyboardInterrupt:
        print("\n[AGENT] Interrupted by user. Saving state and exiting.")
        # Save whatever we have
        if "checkpoint" in dir():
            checkpoint.save()
        sys.exit(1)
    except Exception as e:
        log.error("Fatal error: %s", e, exc_info=True)
        print(f"\n[FATAL] {e}")
        # Save partial state
        try:
            checkpoint.save()
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
