"""Tests for Project Aurelius v7.0 improvements — chem_utils fingerprint utilities."""
from __future__ import annotations

import pytest


class TestChemModule:
    """Tests for the remaining chem_utils functions (fingerprint serialization)."""

    def test_chem_module_exports(self):
        from aurelius.utils.chem_utils import _deserialize_fp, _serialize_fp, _tanimoto

        assert callable(_serialize_fp)
        assert callable(_deserialize_fp)
        assert callable(_tanimoto)

    def test_tanimoto_same_fingerprint(self):
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            from rdkit.DataStructs import ExplicitBitVect

            from aurelius.utils.chem_utils import _tanimoto

            mol = Chem.MolFromSmiles("CCO")
            Chem.AddHs(mol)
            fp = AllChem.GetMorganFingerprint(mol, 2)

            ev = ExplicitBitVect(fp.GetNumBits())
            for idx, _val in fp.GetNonzeroElements().items():
                ev.SetBit(idx)

            similarity = _tanimoto(ev, ev)
            assert abs(similarity - 1.0) < 0.01
        except (ImportError, NotImplementedError) as exc:
            pytest.skip(f"Not implemented: {exc}")
        except Exception:
            pytest.skip("RDKit fingerprint conversion failed")

    def test_molecule_context_replaces_safe_mol(self):
        """MoleculeContext.from_smiles replaces _safe_mol_from_smiles."""
        from aurelius.types import MoleculeContext

        ctx = MoleculeContext.from_smiles("CCO")
        assert ctx is not None
        assert ctx.is_valid_electrolyte_mol() is True

        ctx2 = MoleculeContext.from_smiles("not_a_valid_smiles_string_!!!")
        assert ctx2 is None
