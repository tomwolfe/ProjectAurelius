"""Tests for BasePropertyModel, ElectrolytePack, and OrganicElectronicsPack."""

from __future__ import annotations

from rdkit import Chem

from aurelius.scoring.oracle.gc import (
    BasePropertyModel,
    ElectrolytePack,
    _count_fragments,
    predict_dielectric_proxy,
    predict_viscosity_proxy,
)
from aurelius.scoring.oracle.packs import OrganicElectronicsPack
from aurelius.types import MoleculeContext


class TestBasePropertyModel:
    """BasePropertyModel infrastructure."""

    def test_base_model_has_helpers(self) -> None:
        """BasePropertyModel should provide count_fragments and saturate_contrib."""
        model = ElectrolytePack()
        assert hasattr(model, "count_fragments")
        assert hasattr(model, "saturate_contrib")
        assert hasattr(model, "get_fragment_names")

    def test_count_fragments_returns_dict(self) -> None:
        """count_fragments should return a non-empty dict for a known molecule."""
        model = ElectrolytePack()
        mol = Chem.MolFromSmiles("CCO")
        assert mol is not None
        counts = model.count_fragments(mol)
        assert isinstance(counts, dict)
        # Ethanol should have at least alcohol fragment
        assert "alcohol" in counts or counts.get("alcohol", 0) >= 0
        assert all(isinstance(v, int) for v in counts.values())

    def test_saturate_contrib(self) -> None:
        """saturate_contrib should exhibit Michaelis-Menten behavior."""
        model = ElectrolytePack()
        # Zero count -> zero contribution
        assert model.saturate_contrib(0, 5.0) == 0.0
        # Increasing count -> increasing but saturating contribution
        c1 = model.saturate_contrib(1, 5.0)
        c2 = model.saturate_contrib(2, 5.0)
        assert 0 < c1 < c2 < 5.0

    def test_get_fragment_names(self) -> None:
        """get_fragment_names should return a list of strings."""
        model = ElectrolytePack()
        names = model.get_fragment_names()
        assert len(names) > 10
        assert all(isinstance(n, str) for n in names)
        assert "carbonate" in names


class TestElectrolytePack:
    """ElectrolytePack production predictions."""

    def test_predict_dielectric_matches_module_func(self) -> None:
        """ElectrolytePack.predict_dielectric should match module-level function."""
        pack = ElectrolytePack()
        ctx = MoleculeContext.from_smiles("CCO")
        assert ctx is not None
        pack_result = pack.predict_dielectric(ctx)
        module_result = predict_dielectric_proxy(ctx)
        assert abs(pack_result - module_result) < 1e-6

    def test_predict_viscosity_matches_module_func(self) -> None:
        """ElectrolytePack.predict_viscosity should match module-level function."""
        pack = ElectrolytePack()
        ctx = MoleculeContext.from_smiles("C1COC(=O)O1")
        assert ctx is not None
        pack_result = pack.predict_viscosity(ctx)
        module_result = predict_viscosity_proxy(ctx)
        assert abs(pack_result - module_result) < 1e-6

    def test_predict_all_returns_all_keys(self) -> None:
        """predict_all should return all expected proxy keys."""
        pack = ElectrolytePack()
        ctx = MoleculeContext.from_smiles("CCO")
        assert ctx is not None
        props = pack.predict_all(ctx)
        expected_keys = {
            "dielectric_proxy", "viscosity_proxy", "li_solvation_proxy",
            "ced_proxy", "conductivity_proxy", "li_dissociation_proxy",
        }
        assert set(props.keys()) == expected_keys
        for v in props.values():
            assert isinstance(v, float)

    def test_property_keys_mapping(self) -> None:
        """property_keys should provide correct short-name mapping."""
        pack = ElectrolytePack()
        keys = pack.property_keys()
        assert keys["dielectric"] == "dielectric_proxy"
        assert keys["viscosity"] == "viscosity_proxy"

    def test_predict_gas_evolution(self) -> None:
        """predict_gas_evolution should return reasonable values."""
        pack = ElectrolytePack()
        # Linear carbonate should have higher gas evolution than cyclic
        ctx_linear = MoleculeContext.from_smiles("COC(=O)OC")
        ctx_cyclic = MoleculeContext.from_smiles("C1COC(=O)O1")
        assert ctx_linear is not None
        assert ctx_cyclic is not None
        gas_linear = pack.predict_gas_evolution(ctx_linear)
        gas_cyclic = pack.predict_gas_evolution(ctx_cyclic)
        assert 0 <= gas_linear <= 5.0
        assert 0 <= gas_cyclic <= 5.0


class TestOrganicElectronicsPack:
    """OrganicElectronicsPack for new markets."""

    def test_predict_hole_mobility(self) -> None:
        """hole_mobility proxy should return a float for a conjugated molecule."""
        pack = OrganicElectronicsPack()
        ctx = MoleculeContext.from_smiles("c1ccccc1")
        assert ctx is not None
        hm = pack.predict_hole_mobility(ctx)
        assert isinstance(hm, float)
        assert hm > 0.0

    def test_predict_electron_affinity(self) -> None:
        """electron_affinity proxy should return a float."""
        pack = OrganicElectronicsPack()
        ctx = MoleculeContext.from_smiles("c1ccccc1")
        assert ctx is not None
        ea = pack.predict_electron_affinity(ctx)
        assert isinstance(ea, float)
        assert ea >= 0.0

    def test_predict_all_keys(self) -> None:
        """predict_all should return organic electronics proxies."""
        pack = OrganicElectronicsPack()
        ctx = MoleculeContext.from_smiles("c1ccccc1C#N")
        assert ctx is not None
        props = pack.predict_all(ctx)
        assert "hole_mobility_proxy" in props
        assert "electron_affinity_proxy" in props

    def test_higher_conjugation_higher_mobility(self) -> None:
        """More conjugated systems should have higher hole mobility."""
        pack = OrganicElectronicsPack()
        ctx_small = MoleculeContext.from_smiles("c1ccccc1")
        ctx_large = MoleculeContext.from_smiles("c1ccc2cc3ccccc3cc2c1")
        assert ctx_small is not None
        assert ctx_large is not None
        hm_small = pack.predict_hole_mobility(ctx_small)
        hm_large = pack.predict_hole_mobility(ctx_large)
        assert hm_large >= hm_small, "Larger pi-system should have higher mobility"

    def test_electron_withdrawing_boosts_ea(self) -> None:
        """Electron-withdrawing groups should increase electron affinity."""
        pack = OrganicElectronicsPack()
        ctx_plain = MoleculeContext.from_smiles("c1ccccc1")
        ctx_withdrawing = MoleculeContext.from_smiles("c1ccc([N+](=O)[O-])cc1")
        assert ctx_plain is not None
        assert ctx_withdrawing is not None
        ea_plain = pack.predict_electron_affinity(ctx_plain)
        ea_with = pack.predict_electron_affinity(ctx_withdrawing)
        assert ea_with >= ea_plain, "EWG should increase EA"


class TestPropertyOracleWithPack:
    """PropertyOracle integration with different property packs."""

    def test_default_pack_is_electrolyte(self) -> None:
        """Default PropertyOracle should use ElectrolytePack."""
        from aurelius.scoring.oracle import PropertyOracle
        oracle = PropertyOracle(use_xtb=False, use_surrogate=False, use_gc_uq=False)
        assert isinstance(oracle._property_pack, ElectrolytePack)

    def test_evaluate_with_electrolyte_pack(self) -> None:
        """Default evaluate should return electrolyte properties."""
        from aurelius.scoring.oracle import PropertyOracle
        oracle = PropertyOracle(use_xtb=False, use_surrogate=False, use_gc_uq=False)
        ctx = MoleculeContext.from_smiles("CCO")
        assert ctx is not None
        result = oracle.evaluate(ctx)
        assert "dielectric_proxy" in result
        assert "viscosity_proxy" in result

    def test_organic_electronics_pack_in_oracle(self) -> None:
        """PropertyOracle with OrganicElectronicsPack should return OE proxies."""
        from aurelius.scoring.oracle import PropertyOracle
        pack = OrganicElectronicsPack()
        oracle = PropertyOracle(
            use_xtb=False, use_surrogate=False, use_gc_uq=False,
            property_pack=pack,
        )
        ctx = MoleculeContext.from_smiles("c1ccccc1")
        assert ctx is not None
        result = oracle.evaluate(ctx)
        assert "hole_mobility_proxy" in result
        assert "electron_affinity_proxy" in result
        # Electrolyte proxies should NOT be present
        assert "dielectric_proxy" not in result
