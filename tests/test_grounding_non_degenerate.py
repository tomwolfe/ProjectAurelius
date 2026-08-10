"""Synthesizability grounding must be a real selection pressure (ADR-2026-08-10-02).

Before this work the grounding score was effectively constant: every one of the
15 molecules in ``discoveries.sdf`` scored exactly 0.7731, and the whole known-
electrolyte set produced only 7 distinct values. A constant cannot rank
anything, so synthesizability exerted no selection pressure at all despite
being wired into both the scalar score and NSGA-II.

Three defects caused it, and each has a regression test here:

1. ``_direct_precursor_match`` called ``GetSubstructMatch`` with reversed
   arguments, pinning direct confidence to 0.000.
2. ``_cached_coverage`` counted the *fraction of fragments* passing a binary
   test that nearly everything passes, saturating at 1.000.
3. ``compute_synthesis_feasibility`` returned one of three literals, and 96%
   of realistic candidates landed on 0.9.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from rdkit import Chem

from aurelius.agent.mutation.brics import (
    _direct_precursor_match,
    brics_building_block_coverage,
    combined_grounding_score,
)
from aurelius.agent.mutation.retrosynthetic import (
    as_ring_aware_query,
    compute_synthesis_feasibility,
    infeasibility_penalty,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "src", "aurelius", "data")

# Structures that a surrogate might score well but no chemist would make:
# peroxides, azides, polysilyl ethers, strained polyepoxides, boroxines.
FRANKENSTEIN_SMILES = [
    "FC(F)(F)[Si](C)(C)O[Si](C)(C)C(F)(F)F",
    "O=S(=O)(N=[N+]=[N-])C1=CC=CC=C1",
    "[SiH3]C(F)(F)OC(=O)OC(F)(F)[SiH3]",
    "N#CC(C#N)(C#N)C#N",
    "FC(F)(F)OOC(F)(F)F",
    "O=[N+]([O-])C(N=[N+]=[N-])CN=[N+]=[N-]",
    "C1OC1C1OC1C1OC1",
    "B1OB(OB(O1)C)C",
    "O=C1OC(=O)C(=O)OC(=O)O1",
    "N#CC(F)(OOC(F)(F)C#N)C#N",
]


@pytest.fixture(scope="module")
def known_mols():
    with open(os.path.join(DATA_DIR, "known_electrolytes.json")) as fh:
        smiles = json.load(fh)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    return [m for m in mols if m is not None]


@pytest.fixture(scope="module")
def frankenstein_mols():
    return [m for s in FRANKENSTEIN_SMILES if (m := Chem.MolFromSmiles(s)) is not None]


def test_grounding_has_real_variance(known_mols):
    """The score must actually vary across a realistic population."""
    scores = np.array([combined_grounding_score(m) for m in known_mols])
    assert scores.std() > 0.10, f"grounding std {scores.std():.4f} is degenerate"
    assert len(set(np.round(scores, 3))) >= 0.5 * len(scores), (
        "too many tied grounding values — the score is near-constant"
    )


def test_template_feasibility_is_graded(known_mols):
    """Template feasibility must not collapse onto a handful of literals."""
    scores = [compute_synthesis_feasibility(m) for m in known_mols]
    assert len(set(np.round(scores, 3))) > 10, (
        f"only {len(set(np.round(scores, 3)))} distinct template scores"
    )


def test_brics_coverage_does_not_saturate(known_mols):
    """Coverage pinned at 1.000 carries no information."""
    cov = np.array([brics_building_block_coverage(m) for m in known_mols])
    saturated = float(np.mean(cov >= 0.999))
    assert saturated < 0.6, f"{saturated:.0%} of coverage values are saturated at 1.0"


def test_direct_precursor_match_is_not_always_zero():
    """Regression: the reversed-argument bug pinned this to 0.0 for everything."""
    confidences = [
        _direct_precursor_match(Chem.MolFromSmiles(s))[1]
        for s in ["COCC#N", "COC(=O)OC", "CCOC(=O)OCC", "O=C1OCCO1", "COCCOC"]
    ]
    assert max(confidences) > 0.0
    assert any(c > 0.3 for c in confidences), (
        "no molecule has meaningful direct precursor overlap — "
        "substructure arguments are probably reversed again"
    )


def test_known_electrolytes_outrank_frankensteins(known_mols, frankenstein_mols):
    """The headline adversarial claim."""
    known = np.array([combined_grounding_score(m) for m in known_mols])
    junk = np.array([combined_grounding_score(m) for m in frankenstein_mols])

    assert known.mean() - junk.mean() > 0.30, (
        f"separation too small: known {known.mean():.3f} vs junk {junk.mean():.3f}"
    )

    # No more than one junk molecule may reach the lower quartile of real ones.
    bar = float(np.percentile(known, 25))
    escapees = int(np.sum(junk > bar))
    assert escapees <= 1, f"{escapees}/{len(junk)} Frankenstein molecules cleared {bar:.3f}"


def test_frankensteins_fail_the_report_gate(frankenstein_mols):
    """The 0.75 wet-lab handoff gate must reject unmakeable structures."""
    from aurelius.reporting import CANDIDATE_CASCADE

    threshold = next(t for key, t, _ in CANDIDATE_CASCADE
                     if key == "combined_grounding_score")
    for mol in frankenstein_mols:
        score = combined_grounding_score(mol)
        assert score < threshold, (
            f"{Chem.MolToSmiles(mol)} scored {score:.3f} >= gate {threshold}"
        )


@pytest.mark.parametrize("smiles,motif", [
    ("FC(F)(F)OOC(F)(F)F", "peroxide"),
    ("O=S(=O)(N=[N+]=[N-])C1=CC=CC=C1", "azide"),
    ("FC(F)(F)[Si](C)(C)O[Si](C)(C)C(F)(F)F", "disiloxane"),
    ("B1OB(OB(O1)C)C", "exotic_heteroatom"),
    ("N#CC(C#N)(C#N)C#N", "polynitrile_carbon"),
])
def test_infeasible_motifs_are_detected(smiles, motif):
    penalty, hits = infeasibility_penalty(Chem.MolFromSmiles(smiles))
    assert motif in hits
    assert penalty < 1.0


def test_common_solvents_carry_no_infeasibility_penalty():
    """Real electrolytes must not be penalised by the adversarial filter."""
    for smiles in ["O=C1OCCO1", "COC(=O)OC", "COCCOC", "CC#N", "O=S1(=O)CCCC1"]:
        penalty, hits = infeasibility_penalty(Chem.MolFromSmiles(smiles))
        assert penalty == 1.0, f"{smiles} wrongly penalised for {hits}"


def test_ring_aware_matching_rejects_chain_for_ring():
    """A linear glyme is not a precursor for a strained triepoxide.

    Regression for the topology-blind lookup that let C1OC1C1OC1C1OC1 match
    triglyme at 100% coverage and be scored directly purchasable.
    """
    triepoxide = Chem.MolFromSmiles("C1OC1C1OC1C1OC1")
    triglyme = Chem.MolFromSmiles("COCCOCCOC")

    assert triepoxide.HasSubstructMatch(triglyme), "precondition: naive match succeeds"
    assert not triepoxide.HasSubstructMatch(as_ring_aware_query(triglyme)), (
        "ring-aware query must reject a chain precursor for a ring target"
    )


def test_ring_aware_matching_keeps_true_ring_precursors():
    """The fix must not break legitimate ring-to-ring matches."""
    ec = Chem.MolFromSmiles("O=C1OCCO1")
    assert ec.HasSubstructMatch(as_ring_aware_query(Chem.MolFromSmiles("O=C1OCCO1")))
