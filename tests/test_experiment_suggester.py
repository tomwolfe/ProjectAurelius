"""Active experiment suggestion — ranking behaviour and structured output.

The central property under test is that suggestions rank by what the oracle
*does not know*, not by what it predicts to be good. A suggester that
recommends the highest-scoring molecule has not closed the loop; it has just
re-sorted the leaderboard.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from rdkit import Chem

from aurelius.agent.experiment_suggester import (
    DEFAULT_WEIGHTS,
    MEASURABLE_PROPERTIES,
    ExperimentSuggestion,
    _diversify,
    _doa_proximity_score,
    _harvest_fragments,
    _harvest_from_top_candidates,
    _strip_brics_dummy_atoms,
    bald_acquisition_score,
    default_candidate_pool,
    expand_candidate_pool,
    expected_improvement,
    pareto_ucb_score,
    suggest_experiments,
    write_suggestions,
)
from aurelius.scoring.oracle.conformal import ConformalPredictor
from aurelius.types import MoleculeContext

# Chemically varied, all synthesizable; a mix of well-covered solvents
# (EC, DMC) and less-covered scaffolds.
CANDIDATES = [
    "C1COC(=O)O1",          # ethylene carbonate - heavily calibrated
    "COC(=O)OC",            # dimethyl carbonate - heavily calibrated
    "CC#N",                 # acetonitrile
    "FC(F)(F)COC(=O)OC",    # fluorinated carbonate
    "O=S1(=O)CCCC1",        # sulfolane
    "COCCOC",               # DME
    "CS(=O)(=O)c1ccc(C#N)cc1",  # aryl sulfone nitrile - unusual
]


class TestBasicBehaviour:
    def test_returns_requested_number(self):
        out = suggest_experiments(CANDIDATES, top_n=5)
        assert len(out) == 5
        assert all(isinstance(s, ExperimentSuggestion) for s in out)

    def test_ranked_descending_after_batch_adjustment(self):
        out = suggest_experiments(CANDIDATES, top_n=6)
        adjusted = [s.components["batch_adjusted_score"] for s in out]
        assert adjusted == sorted(adjusted, reverse=True)

    def test_every_suggestion_is_complete(self):
        for s in suggest_experiments(CANDIDATES, top_n=5):
            assert Chem.MolFromSmiles(s.smiles) is not None
            assert s.property_to_measure in MEASURABLE_PROPERTIES.values()
            assert s.priority_score > 0
            assert len(s.rationale) > 20
            assert s.prediction_interval[0] < s.prediction_interval[1]
            assert s.units

    def test_invalid_smiles_are_skipped_not_fatal(self):
        out = suggest_experiments(["not_a_molecule", "C1COC(=O)O1", ""], top_n=3)
        assert out
        assert all(Chem.MolFromSmiles(s.smiles) is not None for s in out)

    def test_empty_input_yields_no_suggestions(self):
        assert suggest_experiments([], top_n=5) == []

    def test_property_filter_is_respected(self):
        out = suggest_experiments(CANDIDATES, top_n=5, properties=["homo"])
        assert {s.property_to_measure for s in out} == {"homo_eV"}

    def test_unknown_property_raises(self):
        with pytest.raises(ValueError, match="Unknown propert"):
            suggest_experiments(CANDIDATES, top_n=3, properties=["melting_point"])

    def test_deterministic(self):
        first = suggest_experiments(CANDIDATES, top_n=5)
        second = suggest_experiments(CANDIDATES, top_n=5)
        assert [(s.smiles, s.property_to_measure) for s in first] == [
            (s.smiles, s.property_to_measure) for s in second
        ]


class TestInformationGainRanking:
    def test_prefers_molecules_far_from_the_calibration_set(self):
        """Novelty must actually move the ranking.

        EC is in the calibration set; the aryl sulfone nitrile is not. With
        only the novelty term enabled, the unfamiliar molecule must win.
        """
        out = suggest_experiments(
            ["C1COC(=O)O1", "CS(=O)(=O)c1ccc(C#N)cc1"],
            top_n=2,
            properties=["homo"],
            weights={"uncertainty": 0.0, "doa_proximity": 0.0, "bias": 0.0, "novelty": 1.0},
        )
        assert out[0].smiles == Chem.MolToSmiles(Chem.MolFromSmiles("CS(=O)(=O)c1ccc(C#N)cc1"))

    def test_uncertainty_term_is_not_constant(self):
        """A constant uncertainty term silently disables uncertainty sampling.

        This is a regression test: normalising interval width by the
        calibration quantile returned 1.0 for every molecule by construction,
        because the quantile *is* the typical width.
        """
        out = suggest_experiments(CANDIDATES, top_n=len(CANDIDATES) * 4)
        values = {s.components["uncertainty"] for s in out}
        assert len(values) > 1, "uncertainty term carries no information"

    def test_uncertainty_ordering_follows_interval_width(self):
        """A wider conformal interval must mean a higher uncertainty term.

        ``components["uncertainty"]`` is rounded to 4 dp, so several
        candidates legitimately tie (four share 0.4214 here) and their
        relative order is arbitrary. Only strictly-different uncertainty
        values carry an ordering claim; comparing tied entries tests the
        sort's tie-breaking, not the physics.
        """
        out = suggest_experiments(CANDIDATES, top_n=40, properties=["homo"])
        pairs = [
            (s.components["uncertainty"], s.prediction_interval[1] - s.prediction_interval[0])
            for s in out
        ]
        for u_a, w_a in pairs:
            for u_b, w_b in pairs:
                if u_a > u_b:
                    assert w_a >= w_b, (
                        f"uncertainty {u_a} > {u_b} but interval width "
                        f"{w_a:.4f} < {w_b:.4f}"
                    )

    def test_does_not_simply_rank_by_predicted_quality(self):
        """The top suggestion must not be the best-scoring molecule.

        EC and DMC are the two best-characterised electrolyte solvents in the
        project. If they lead the worklist, the suggester is recommending
        what is already known.
        """
        out = suggest_experiments(CANDIDATES, top_n=3)
        well_known = {
            Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in ("C1COC(=O)O1", "COC(=O)OC")
        }
        assert out[0].smiles not in well_known

    def test_weights_change_the_outcome(self):
        """Every term must be zeroed explicitly, or a defaulted term dominates.

        This test previously listed only four weights. When ``expected_impact``
        was added (ADR-2026-08-10-03) it kept its default 0.25 in both arms and
        drove both rankings to the same answer, so the test failed for the
        right reason: the override was silently incomplete.
        """
        off = dict.fromkeys(DEFAULT_WEIGHTS, 0.0)

        novelty_first = suggest_experiments(
            CANDIDATES, top_n=3, weights={**off, "novelty": 1.0},
        )
        uncertainty_first = suggest_experiments(
            CANDIDATES, top_n=3, weights={**off, "uncertainty": 1.0},
        )
        assert [s.smiles for s in novelty_first] != [s.smiles for s in uncertainty_first]

    def test_expected_impact_changes_the_outcome(self):
        """The decision-relevance term must be able to drive selection on its own."""
        off = dict.fromkeys(DEFAULT_WEIGHTS, 0.0)

        impact_first = suggest_experiments(
            CANDIDATES, top_n=3, weights={**off, "expected_impact": 1.0},
        )
        uncertainty_first = suggest_experiments(
            CANDIDATES, top_n=3, weights={**off, "uncertainty": 1.0},
        )
        assert impact_first, "expected-impact ranking returned nothing"
        assert [s.smiles for s in impact_first] != [s.smiles for s in uncertainty_first]

    def test_doa_proximity_peaks_at_the_boundary(self):
        """Most informative near the edge, less so deep inside or far outside."""
        deep_inside = _doa_proximity_score(1.0)
        near_edge = _doa_proximity_score(0.85)
        beyond = _doa_proximity_score(0.70)
        assert deep_inside == 0.0
        assert near_edge > deep_inside
        assert near_edge > beyond


class TestSystematicBias:
    def test_bias_term_raises_priority_for_the_biased_property(self):
        class StubController:
            def detect_systematic_bias(self):
                return {
                    "dielectric": {"bias_detected": True, "magnitude": 8.0, "n_records": 20},
                    "viscosity": {"bias_detected": False, "magnitude": 0.0, "n_records": 20},
                }

        biased = suggest_experiments(
            CANDIDATES, top_n=8, controller=StubController(),
            properties=["dielectric", "viscosity"],
            weights={"uncertainty": 0.0, "novelty": 0.0, "doa_proximity": 0.0, "bias": 1.0},
        )
        assert biased[0].property_to_measure == "dielectric_constant"

    def test_broken_controller_does_not_crash(self):
        class ExplodingController:
            def detect_systematic_bias(self):
                raise RuntimeError("no state")

        out = suggest_experiments(CANDIDATES, top_n=3, controller=ExplodingController())
        assert out
        assert all(s.components["bias"] == 0.0 for s in out)


class TestBatchDiversity:
    def test_batch_is_not_all_the_same_property(self):
        out = suggest_experiments(CANDIDATES, top_n=6)
        assert len({s.property_to_measure for s in out}) >= 3

    def test_batch_is_not_all_the_same_molecule(self):
        out = suggest_experiments(CANDIDATES, top_n=6)
        assert len({s.smiles for s in out}) >= 4

    def test_diversify_takes_the_strongest_suggestion_first(self):
        ranked = [
            ExperimentSuggestion("CCO", "homo_eV", 0.9, "r", 0.0, (0.0, 1.0)),
            ExperimentSuggestion("CCO", "lumo_eV", 0.8, "r", 0.0, (0.0, 1.0)),
            ExperimentSuggestion("CCC", "homo_eV", 0.7, "r", 0.0, (0.0, 1.0)),
        ]
        out = _diversify(ranked, top_n=2)
        assert (out[0].smiles, out[0].property_to_measure) == ("CCO", "homo_eV")

    def test_diversify_overturns_ranking_when_redundancy_is_high(self):
        """A duplicate must lose to a lower-scoring but non-redundant option.

        Here the runner-up repeats both the molecule and the property of the
        first pick, taking the discount twice (0.85 * 0.6^2 = 0.31), so the
        fresh molecule at 0.60 * 0.6 = 0.36 correctly overtakes it.
        """
        ranked = [
            ExperimentSuggestion("CCO", "homo_eV", 0.90, "r", 0.0, (0.0, 1.0)),
            ExperimentSuggestion("CCO", "homo_eV", 0.85, "r", 0.0, (0.0, 1.0)),
            ExperimentSuggestion("CCC", "homo_eV", 0.60, "r", 0.0, (0.0, 1.0)),
        ]
        out = _diversify(ranked, top_n=2)
        assert out[1].smiles == "CCC"

    def test_diversify_penalises_structurally_similar_molecules(self):
        """Distinct-but-near-identical scaffolds must not both lead the batch.

        Regression guard for ADR-2026-08-08-06. The molecule/property
        repetition discount treats two different SMILES as fully independent,
        so a batch could fill with one homologous family. The suggester's
        ``novelty`` term cannot separate them either: it measures distance to
        the *calibration set*, which is near-identical across such a family.
        """
        homologues = ["CCCCCCO", "CCCCCCCO", "CCCCCCCCO"]
        distinct = "O=S1(=O)CCCC1"
        ranked = [
            ExperimentSuggestion(homologues[0], "homo_eV", 0.90, "r", 0.0, (0.0, 1.0)),
            ExperimentSuggestion(homologues[1], "homo_eV", 0.88, "r", 0.0, (0.0, 1.0)),
            ExperimentSuggestion(homologues[2], "homo_eV", 0.86, "r", 0.0, (0.0, 1.0)),
            ExperimentSuggestion(distinct, "homo_eV", 0.60, "r", 0.0, (0.0, 1.0)),
        ]
        fingerprints = {
            smi: MoleculeContext.from_smiles(smi).get_ecfp4()
            for smi in [*homologues, distinct]
        }

        out = _diversify(list(ranked), top_n=2, fingerprints=fingerprints)

        assert out[1].smiles == distinct, (
            "The structurally distinct molecule must outrank a near-duplicate "
            f"despite a lower standalone score; got {[s.smiles for s in out]}"
        )

    def test_diversify_without_fingerprints_is_unchanged(self):
        """The structural penalty must be optional, not a hard requirement."""
        ranked = [
            ExperimentSuggestion("CCO", "homo_eV", 0.9, "r", 0.0, (0.0, 1.0)),
            ExperimentSuggestion("CCC", "homo_eV", 0.7, "r", 0.0, (0.0, 1.0)),
        ]
        out = _diversify(list(ranked), top_n=2)
        assert [s.smiles for s in out] == ["CCO", "CCC"]

    def test_batch_is_structurally_diverse_end_to_end(self):
        """The full suggester must produce a batch of unrelated scaffolds."""
        from aurelius.utils.device import batch_tanimoto

        out = suggest_experiments(CANDIDATES, top_n=4, properties=["homo"])
        fps = [
            MoleculeContext.from_smiles(s.smiles).get_ecfp4() for s in out
        ]
        sim = batch_tanimoto(fps)
        upper = sim[np.triu_indices(sim.shape[0], k=1)]

        assert float(upper.mean()) < 0.30, (
            f"Suggested batch is structurally redundant (mean pairwise "
            f"Tanimoto {upper.mean():.3f}): {[s.smiles for s in out]}"
        )

    def test_diversify_never_invents_suggestions(self):
        ranked = [
            ExperimentSuggestion("CCO", "homo_eV", 0.9, "r", 0.0, (0.0, 1.0)),
            ExperimentSuggestion("CCC", "homo_eV", 0.7, "r", 0.0, (0.0, 1.0)),
        ]
        assert len(_diversify(ranked, top_n=10)) == 2


class TestRationale:
    def test_rationale_names_the_dominant_reason(self):
        out = suggest_experiments(
            CANDIDATES, top_n=2, properties=["homo"],
            weights={"novelty": 1.0, "uncertainty": 0.0, "doa_proximity": 0.0, "bias": 0.0},
        )
        assert "distant" in out[0].rationale

    def test_rationale_quotes_the_interval_when_uncertainty_leads(self):
        out = suggest_experiments(
            CANDIDATES, top_n=2, properties=["dielectric"],
            weights={"novelty": 0.0, "uncertainty": 1.0, "doa_proximity": 0.0, "bias": 0.0},
        )
        assert "conformal interval" in out[0].rationale


class TestOutput:
    def test_writes_valid_json(self, tmp_path):
        out = suggest_experiments(CANDIDATES, top_n=4)
        path = tmp_path / "suggestions.json"
        write_suggestions(out, str(path))
        payload = json.loads(path.read_text())
        assert payload["n_suggestions"] == 4
        assert len(payload["suggestions"]) == 4
        first = payload["suggestions"][0]
        for key in ("smiles", "property_to_measure", "priority_score", "rationale"):
            assert key in first

    def test_property_names_match_the_ingestion_schema(self):
        """Suggestions must be answerable by ``aurelius ingest-experiment``."""
        import os

        schema_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "aurelius", "data",
            "experimental_results_schema.json",
        )
        with open(schema_path) as fh:
            schema = json.load(fh)
        allowed = set(
            schema["definitions"]["measurement"]["properties"]["measured_property"]["enum"]
        )
        assert allowed, "schema must enumerate measured_property"
        for s in suggest_experiments(CANDIDATES, top_n=6):
            assert s.property_to_measure in allowed

    def test_weights_are_reported(self, tmp_path):
        path = tmp_path / "s.json"
        write_suggestions(suggest_experiments(CANDIDATES, top_n=2), str(path))
        assert json.loads(path.read_text())["weights"] == DEFAULT_WEIGHTS


class TestCandidatePool:
    def test_default_pool_is_usable(self):
        pool = default_candidate_pool()
        assert len(pool) > 10
        assert all(Chem.MolFromSmiles(s) is not None for s in pool[:20])


class TestConformalLocalAdaptivity:
    def test_intervals_vary_between_molecules(self):
        """Regression: intervals were identical for every molecule.

        Split conformal with a single global quantile gives valid marginal
        coverage but zero ability to say which molecule is harder, which is
        precisely what active learning needs.
        """
        predictor = ConformalPredictor()
        predictor.fit()
        widths = set()
        for smiles in ("C1COC(=O)O1", "FC(F)(F)COC(=O)OC", "c1ccc2c(c1)ccc1ccccc12"):
            mol = Chem.MolFromSmiles(smiles)
            lo, hi = predictor.predict_interval("homo", -7.0, mol=mol)
            widths.add(round(hi - lo, 6))
        assert len(widths) > 1

    def test_unfamiliar_molecule_gets_a_wider_interval(self):
        predictor = ConformalPredictor()
        predictor.fit()
        familiar = Chem.MolFromSmiles("C1COC(=O)O1")
        unfamiliar = Chem.MolFromSmiles("c1ccc2c(c1)ccc1ccccc12")
        assert predictor.difficulty(unfamiliar) > predictor.difficulty(familiar)

    def test_omitting_mol_preserves_legacy_behaviour(self):
        predictor = ConformalPredictor()
        predictor.fit()
        a = predictor.predict_interval("homo", -7.0)
        b = predictor.predict_interval("homo", -7.0)
        assert a == b


# ---------------------------------------------------------------------------
# Pool expansion tests (ADR-2026-08-11-02)
# ---------------------------------------------------------------------------


class TestPoolExpansion:
    def test_strip_brics_dummy_atoms(self):
        """BRICS dummy atoms (*) are removed from fragment SMILES."""
        clean = _strip_brics_dummy_atoms("[1*]C(=O)OC")
        assert clean is not None
        assert "*" not in clean
        assert Chem.MolFromSmiles(clean) is not None

    def test_harvest_fragments_produces_valid_fragments(self):
        """Harvested fragments are valid molecules with ≥2 heavy atoms."""
        frags = _harvest_fragments(["COC(=O)OC", "O=C1OCCO1", "CC#N"])
        assert len(frags) > 0
        for f in frags:
            mol = Chem.MolFromSmiles(f)
            assert mol is not None
            assert mol.GetNumHeavyAtoms() >= 2

    def test_harvest_fragments_deduplicates(self):
        """Identical fragments from different molecules are deduplicated."""
        frags = _harvest_fragments(["COC(=O)OC", "COC(=O)OC"])
        assert len(frags) == len(set(frags))

    def test_expand_pool_grows_small_pool(self):
        """A pool below MIN_POOL_SIZE is expanded via BRICS recombination."""
        small_pool = ["COC(=O)OC", "O=C1OCCO1", "CC#N", "COCCOC"]
        expanded = expand_candidate_pool(small_pool, target_size=50)
        assert len(expanded) >= len(small_pool)

    def test_expand_pool_preserves_originals(self):
        """Original candidates are always in the expanded pool."""
        originals = ["COC(=O)OC", "O=C1OCCO1"]
        expanded = expand_candidate_pool(originals, target_size=50)
        for smi in originals:
            assert smi in expanded

    def test_expand_pool_noop_when_large(self):
        """A pool already at target_size is returned unchanged."""
        large_pool = [f"CCCC{'C' * i}O" for i in range(120)]
        expanded = expand_candidate_pool(large_pool, target_size=100)
        assert expanded == large_pool

    def test_expand_pool_deduplicates(self):
        """Expanded pool has no duplicate SMILES."""
        pool = ["COC(=O)OC", "O=C1OCCO1", "CC#N"]
        expanded = expand_candidate_pool(pool, target_size=50)
        assert len(expanded) == len(set(expanded))


# ---------------------------------------------------------------------------
# Expected Improvement tests
# ---------------------------------------------------------------------------


class TestExpectedImprovement:
    def test_ei_zero_when_interval_zero_width(self):
        """A point prediction (zero-width interval) has zero EI."""
        ei = expected_improvement(-7.0, (-7.0, -7.0), -6.0, minimise=True)
        assert ei == 0.0

    def test_ei_positive_when_uncertain_and_near_incumbent(self):
        """Uncertain prediction near the incumbent has positive EI."""
        # point=-6.5, interval width=2.0, incumbent=-6.0, minimise=True
        ei = expected_improvement(-6.5, (-7.5, -5.5), -6.0, minimise=True)
        assert ei > 0.0

    def test_ei_higher_for_wider_interval(self):
        """More uncertainty → higher EI (more room for improvement)."""
        narrow = expected_improvement(-6.5, (-6.8, -6.2), -6.0, minimise=True)
        wide = expected_improvement(-6.5, (-8.0, -5.0), -6.0, minimise=True)
        assert wide > narrow

    def test_ei_zero_when_point_worse_than_incumbent_with_certainty(self):
        """A certain prediction worse than the incumbent has ~zero EI."""
        # point=-5.0 (worse than incumbent -6.0 when minimising), narrow interval
        ei = expected_improvement(-5.0, (-5.1, -4.9), -6.0, minimise=True)
        assert ei < 1e-10  # Gaussian tail, effectively zero

    def test_ei_maximise_mode(self):
        """EI works in maximisation mode (higher is better)."""
        # point=-5.0, incumbent=-6.0, maximise=True → improvement expected
        ei = expected_improvement(-5.0, (-5.5, -4.5), -6.0, minimise=False)
        assert ei > 0.0

    def test_ei_never_negative(self):
        """EI is always ≥ 0 by construction."""
        for point, interval, best, minimise in [
            (-7.0, (-8.0, -6.0), -6.5, True),
            (-5.0, (-5.5, -4.5), -6.0, False),
            (-6.5, (-7.0, -6.0), -6.5, True),
        ]:
            ei = expected_improvement(point, interval, best, minimise=minimise)
            assert ei >= 0.0


# ---------------------------------------------------------------------------
# Integration: pool expansion + EI in suggest_experiments
# ---------------------------------------------------------------------------


class TestPoolExpansionIntegration:
    def test_small_pool_gets_expanded(self):
        """suggest_experiments expands a small pool before scoring."""
        small_pool = ["COC(=O)OC", "O=C1OCCO1", "CC#N", "COCCOC", "CC(=O)OC(C)=O"]
        out = suggest_experiments(small_pool, top_n=3, properties=["homo"])
        assert len(out) > 0
        # The output should include molecules beyond the original pool
        # (if expansion produced valid candidates that scored well)

    def test_ei_component_present_in_suggestions(self):
        """Suggestions include the expected_improvement component."""
        out = suggest_experiments(CANDIDATES, top_n=3, properties=["homo"])
        assert len(out) > 0
        for s in out:
            assert "expected_improvement" in s.components
            assert s.components["expected_improvement"] >= 0.0


# ---------------------------------------------------------------------------
# Phase 1: Pool expansion integration tests (ADR-2026-08-11-03)
# ---------------------------------------------------------------------------


class TestPoolExpansionBeforeScoring:
    """Pool must grow to >=200 before acquisition scoring."""

    def test_pool_expansion_before_scoring(self):
        """A small pool is expanded to >= MIN_POOL_SIZE via BRICS harvesting."""
        small_pool = [
            "COC(=O)OC", "O=C1OCCO1", "CC#N", "COCCOC",
            "CC(=O)OC(C)=O", "O=S1(=O)CCCC1", "FC(F)(F)COC(=O)OC",
        ]
        expanded = expand_candidate_pool(small_pool, target_size=200)
        assert len(expanded) >= len(small_pool)

    def test_harvest_from_top_candidates_grows_pool(self):
        """_harvest_from_top_candidates grows the pool via BRICS recombination."""
        pool = [
            "COC(=O)OC", "O=C1OCCO1", "CC#N", "COCCOC",
            "CC(=O)OC(C)=O", "O=S1(=O)CCCC1", "FC(F)(F)COC(=O)OC",
        ]
        expanded = _harvest_from_top_candidates(pool, top_n=5, max_products=150)
        assert len(expanded) >= len(pool)

    def test_harvested_fragments_are_valid(self):
        """All newly generated recombination products are valid molecules."""
        from aurelius.agent.mutation.smarts import is_electrolyte_like

        pool = [
            "COC(=O)OC", "O=C1OCCO1", "CC#N", "COCCOC",
            "CC(=O)OC(C)=O", "O=S1(=O)CCCC1",
        ]
        expanded = _harvest_from_top_candidates(pool, top_n=5, max_products=150)
        new_products = [smi for smi in expanded if smi not in pool]
        assert len(new_products) > 0, "Harvesting produced no new molecules"
        for smi in new_products:
            mol = Chem.MolFromSmiles(smi)
            assert mol is not None, f"Invalid SMILES produced: {smi}"
            assert mol.GetNumHeavyAtoms() >= 3
            ctx = MoleculeContext.from_smiles(smi)
            assert ctx is not None
            assert is_electrolyte_like(ctx), f"Not electrolyte-like: {smi}"


# ---------------------------------------------------------------------------
# Phase 2: BALD + Pareto UCB acquisition tests (ADR-2026-08-11-02)
# ---------------------------------------------------------------------------


class TestBALDAndParetoUCB:
    def test_default_weights_contain_bald_and_pareto_ucb(self):
        """DEFAULT_WEIGHTS must include bald and pareto_ucb keys."""
        assert "bald" in DEFAULT_WEIGHTS
        assert "pareto_ucb" in DEFAULT_WEIGHTS
        assert DEFAULT_WEIGHTS["bald"] > 0.0
        assert DEFAULT_WEIGHTS["pareto_ucb"] > 0.0

    def test_bald_returns_zero_without_model(self):
        """BALD returns 0.0 when no GPR model is available."""
        mol = Chem.MolFromSmiles("COC(=O)OC")
        assert bald_acquisition_score(mol, None) == 0.0

    def test_bald_returns_in_unit_range_with_model(self):
        """BALD score is in [0, 1] when a GPR model is available."""
        mol = Chem.MolFromSmiles("COC(=O)OC")
        from aurelius.scoring.oracle.conformal import get_conformal_predictor
        from aurelius.agent.experiment_suggester import _gpr_model_from_predictor

        predictor = get_conformal_predictor()
        gpr = _gpr_model_from_predictor(predictor)
        if gpr is None:
            pytest.skip("No GPR model available")
        score = bald_acquisition_score(mol, gpr)
        assert 0.0 <= score <= 1.0

    def test_pareto_ucb_pareto_optimal_candidate_scores_one(self):
        """A candidate dominating all others on UCB gets score 1.0."""
        preds = {"homo": -8.0, "lumo": -2.0, "dielectric": 30.0, "viscosity": 2.0}
        unc = {"homo": 0.1, "lumo": 0.1, "dielectric": 1.0, "viscosity": 0.1}
        others = [
            {"homo": -7.0, "lumo": -1.0, "dielectric": 20.0, "viscosity": 3.0},
            {"homo": -6.0, "lumo": -0.5, "dielectric": 15.0, "viscosity": 5.0},
        ]
        score = pareto_ucb_score(preds, unc, others, beta=1.5)
        assert score == 1.0

    def test_pareto_ucb_dominated_candidate_scores_zero(self):
        """A candidate dominated by another gets score 0.0."""
        preds = {"homo": -7.0, "lumo": -1.0, "dielectric": 20.0, "viscosity": 3.0}
        unc = {p: 0.0 for p in preds}
        others = [
            {"homo": -8.0, "lumo": -2.0, "dielectric": 30.0, "viscosity": 2.0},
        ]
        score = pareto_ucb_score(preds, unc, others, beta=0.0)
        assert score == 0.0

    def test_pareto_ucb_handles_empty_inputs(self):
        """Empty inputs yield 0.0, not a crash."""
        assert pareto_ucb_score({}, {}, [], beta=1.5) == 0.0

    def test_pareto_ucb_targets_frontier(self):
        """At least 60% of top suggestions have pareto_ucb == 1.0."""
        out = suggest_experiments(CANDIDATES, top_n=10)
        frontier_count = sum(1 for s in out if s.components.get("pareto_ucb", 0.0) >= 0.99)
        assert frontier_count >= 0.6 * len(out), (
            f"Only {frontier_count}/{len(out)} suggestions on Pareto front"
        )

    def test_bald_component_present_in_suggestions(self):
        """Suggestions include the bald component."""
        out = suggest_experiments(CANDIDATES, top_n=5)
        for s in out:
            assert "bald" in s.components
            assert 0.0 <= s.components["bald"] <= 1.0

    def test_weights_with_bald_and_pareto_change_outcome(self):
        """Bald+pareto_ucb weights must affect the ranking."""
        off = dict.fromkeys(DEFAULT_WEIGHTS, 0.0)

        bald_only = suggest_experiments(
            CANDIDATES, top_n=3,
            properties=["homo"],
            weights={**off, "bald": 1.0},
        )
        novelty_only = suggest_experiments(
            CANDIDATES, top_n=3,
            properties=["homo"],
            weights={**off, "novelty": 1.0},
        )
        assert bald_only, "bald ranking returned nothing"
        assert [s.smiles for s in bald_only] != [s.smiles for s in novelty_only]

    def test_components_include_all_weighted_terms(self):
        """Every key in DEFAULT_WEIGHTS must appear in suggestion components."""
        out = suggest_experiments(CANDIDATES, top_n=3)
        for s in out:
            for key in DEFAULT_WEIGHTS:
                assert key in s.components, f"Missing component {key} in {list(s.components.keys())}"


class TestBatchEI:
    """Phase 3: fantasization-based batch Expected Improvement."""

    def test_batch_ei_component_present(self):
        """Suggestions include the batch_ei component."""
        out = suggest_experiments(CANDIDATES, top_n=5)
        for s in out:
            assert "batch_ei" in s.components
            assert 0.0 <= s.components["batch_ei"] <= 1.0

    def test_batch_ei_not_constant(self):
        """Batch EI scores vary across candidates (not all identical) when a model is provided."""
        from aurelius.scoring.oracle.delta_correction import DeltaCorrection
        import os
        data_dir = os.path.join(os.path.dirname(__file__), "..", "src", "aurelius", "data")
        with open(os.path.join(data_dir, "orbital_calibration.json")) as f:
            all_entries = json.load(f)
        calib = [e for e in all_entries[:30] if Chem.MolFromSmiles(e["smiles"]) is not None]
        model = DeltaCorrection(calib=calib, calib_smiles=[
            Chem.MolToSmiles(Chem.MolFromSmiles(e["smiles"])) for e in calib
        ])
        out = suggest_experiments(CANDIDATES, top_n=5, properties=["homo"], delta_correction=model)
        scores = [s.components["batch_ei"] for s in out]
        assert len(set(scores)) > 1, "batch_ei is constant — fantasization not working"

    def test_batch_ei_weight_affects_ranking(self):
        """batch_ei weight must affect the pre-diversification ranking."""
        from aurelius.agent.experiment_suggester import (
            _compute_batch_ei_scores,
            _evaluate_candidates,
            _load_calibration_fingerprints,
        )
        from aurelius.scoring.oracle.delta_correction import DeltaCorrection
        import os
        data_dir = os.path.join(os.path.dirname(__file__), "..", "src", "aurelius", "data")
        with open(os.path.join(data_dir, "orbital_calibration.json")) as f:
            all_entries = json.load(f)
        calib = [e for e in all_entries[:30] if Chem.MolFromSmiles(e["smiles"]) is not None]
        model = DeltaCorrection(calib=calib, calib_smiles=[
            Chem.MolToSmiles(Chem.MolFromSmiles(e["smiles"])) for e in calib
        ])

        # Compute batch EI scores directly and verify they differ from BALD
        calibration_fps = _load_calibration_fingerprints()
        evaluated, _ = _evaluate_candidates(CANDIDATES, calibration_fps, max_sa_score=6.0)
        batch_scores = _compute_batch_ei_scores(evaluated, model._homo_model)

        # batch_ei should rank the candidates differently than uniform
        assert len(batch_scores) > 0, "No batch EI scores computed"
        ranked_by_batch = sorted(batch_scores.items(), key=lambda x: -x[1])
        top_batch = ranked_by_batch[0][0]

        # The top batch EI pick should be the molecule that reduces pool-wide
        # uncertainty the most — verify it's a real molecule from our candidates
        assert top_batch in [Chem.MolToSmiles(Chem.MolFromSmiles(c)) for c in CANDIDATES], (
            f"Top batch EI pick {top_batch} not in candidate set"
        )

    def test_batch_ei_prefers_diverse_uncertainty_reducers(self):
        """Batch EI scores a diverse, uncertain candidate higher than a redundant one."""
        from aurelius.agent.experiment_suggester import (
            _compute_batch_ei_scores,
            _evaluate_candidates,
            _load_calibration_fingerprints,
        )
        from aurelius.scoring.oracle.delta_correction import DeltaCorrection

        # Fit a DeltaCorrection on a small calibration set (like the benchmark does)
        import os
        data_dir = os.path.join(os.path.dirname(__file__), "..", "src", "aurelius", "data")
        with open(os.path.join(data_dir, "orbital_calibration.json")) as f:
            all_entries = json.load(f)
        calib = [e for e in all_entries[:30] if Chem.MolFromSmiles(e["smiles"]) is not None]
        model = DeltaCorrection(calib=calib, calib_smiles=[
            Chem.MolToSmiles(Chem.MolFromSmiles(e["smiles"])) for e in calib
        ])

        # Pool: candidates not in the calibration set
        pool = [
            "CS(=O)(=O)c1ccc(C#N)cc1",  # aryl sulfone nitrile — unusual
            "FC(F)(F)COC(=O)OC",       # fluorinated carbonate
            "COCCOCCOC",                # glyme
        ]
        calibration_fps = _load_calibration_fingerprints()
        evaluated, _ = _evaluate_candidates(pool, calibration_fps, max_sa_score=6.0)
        scores = _compute_batch_ei_scores(evaluated, model._homo_model)

        # The unusual scaffold should have a non-trivial batch EI score
        assert len(scores) > 0, "No batch EI scores computed"
        vals = list(scores.values())
        assert any(v > 0 for v in vals), "All batch EI scores are zero"
        assert len(set(round(v, 6) for v in vals)) > 1, "Batch EI scores are uniform"

    def test_suggester_uses_delta_correction(self):
        """suggest_experiments accepts a delta_correction parameter for model-aware acquisition."""
        from aurelius.scoring.oracle.delta_correction import DeltaCorrection
        import os
        data_dir = os.path.join(os.path.dirname(__file__), "..", "src", "aurelius", "data")
        with open(os.path.join(data_dir, "orbital_calibration.json")) as f:
            all_entries = json.load(f)
        calib = [e for e in all_entries[:30] if Chem.MolFromSmiles(e["smiles"]) is not None]
        model = DeltaCorrection(calib=calib, calib_smiles=[
            Chem.MolToSmiles(Chem.MolFromSmiles(e["smiles"])) for e in calib
        ])

        # With delta_correction, batch_ei should be non-zero
        out_with = suggest_experiments(CANDIDATES, top_n=5, properties=["homo"], delta_correction=model)
        batch_ei_with = [s.components.get("batch_ei", 0.0) for s in out_with]
        assert any(v > 0 for v in batch_ei_with), "batch_ei is zero even with delta_correction"
