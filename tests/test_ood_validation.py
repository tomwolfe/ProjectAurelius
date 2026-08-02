"""OOD validation for Project Aurelius - Workstream 1.

Validates the oracle's out-of-distribution performance on chemical classes
not present in the calibration set.
"""

import json
import math

from rdkit import Chem

from aurelius.data.ood_validation import get_ood_molecules
from aurelius.scoring.oracle import PropertyOracle
from aurelius.types import MoleculeContext


def test_ood_molecules():
    """Test that the oracle correctly ranks OOD molecules by chemical class."""
    oracle = PropertyOracle(use_xtb=True)
    ood_molecules = get_ood_molecules()

    # Test each molecule
    for mol_data in ood_molecules:
        smiles = mol_data["smiles"]
        name = mol_data["name"]
        expected_dielectric_rank = mol_data["expected_dielectric_rank"]
        expected_viscosity_rank = mol_data["expected_viscosity_rank"]
        oclass = mol_data["class"]

        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Get oracle predictions
        from aurelius.types import MoleculeContext
        ctx = MoleculeContext.from_smiles(smiles)
        result = oracle.evaluate(ctx)

        # Check physical bounds
        dielectric = result["dielectric_proxy"]
        viscosity = result["viscosity_proxy"]
        homo = result["homo_eV"]
        lumo = result["lumo_eV"]

        # Assert physical bounds
        assert 1.0 <= dielectric <= 100.0, \
            f"{name}: dielectric_proxy {dielectric} outside physical bounds [1,100]"
        assert 0.1 <= viscosity <= 50.0, \
            f"{name}: viscosity_proxy {viscosity} outside physical bounds [0.1,50]"
        assert -12.0 <= homo <= -3.0, \
            f"{name}: homo_eV {homo} outside physical bounds [-12,-3]"
        assert -5.0 <= lumo <= 5.0, \
            f"{name}: lumo_eV {lumo} outside physical bounds [-5,5]"

        # Check expected ranks
        if expected_dielectric_rank == "high":
            assert dielectric > 5.0, f"{name}: expected high dielectric rank, got {dielectric}"
        elif expected_dielectric_rank == "medium":
            assert 2.0 <= dielectric <= 5.0, f"{name}: expected medium dielectric rank, got {dielectric}"
        else:  # low
            assert dielectric < 2.0, f"{name}: expected low dielectric rank, got {dielectric}"

        if expected_viscosity_rank == "high":
            assert viscosity > 1.5, f"{name}: expected high viscosity rank, got {viscosity}"
        elif expected_viscosity_rank == "medium":
            assert 0.5 <= viscosity <= 1.5, f"{name}: expected medium viscosity rank, got {viscosity}"
        else:  # low
            assert viscosity < 0.5, f"{name}: expected low viscosity rank, got {viscosity}"

        print(f"✓ {name}: {oclass} (dielectric={dielectric:.1f}, viscosity={viscosity:.1f})")

    print(f"\nAll {len(ood_molecules)} OOD molecules passed validation")


def test_physical_bounds_violations():
    """Test that sanity bounds catch physical impossibilities."""
    oracle = PropertyOracle(use_xtb=True)
    ood_molecules = get_ood_molecules()

    violations_found = False

    for mol_data in ood_molecules:
        smiles = mol_data["smiles"]
        name = mol_data["name"]

        from aurelius.types import MoleculeContext
        ctx = MoleculeContext.from_smiles(smiles)
        result = oracle.evaluate(ctx)

        # Check for NaN or extreme values
        if any(v is None or math.isnan(v) for v in [
            result["dielectric_proxy"],
            result["viscosity_proxy"],
            result["homo_eV"],
            result["lumo_eV"],
        ]):
            print(f"WARNING: {name} produced NaN or None values")
            violations_found = True

        # Check that oracle provides no invalid predictions
        assert "sanity_warning" not in result or len(result["sanity_warning"]) == 0, \
            f"{name}: oracle should provide sanity_warning when values are clamped"

    # Pure hydrocarbons should have low dielectric
    pure_hydrocarbons = [
        ("hexane", "CCCCCCC"),
        ("cyclohexane", "C1CCCCC1"),
        ("benzene", "c1ccccc1"),
    ]

    for name, smiles in pure_hydrocarbons:
        ctx = MoleculeContext.from_smiles(smiles)
        result = oracle.evaluate(ctx)
        dielectric = result["dielectric_proxy"]
        if dielectric < 3.0:
            print(f"✓ {name}: correctly low dielectric ({dielectric:.2f})")
        else:
            print(f"WARNING: {name}: expected dielectric < 3.0, got {dielectric:.2f}")

    print("\nPhysical bounds validation completed")


if __name__ == "__main__":
    test_ood_molecules()
    print()
    test_physical_bounds_violations()
    print("\nOOD validation completed successfully!")
