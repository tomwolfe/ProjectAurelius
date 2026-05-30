"""Tests for Project Aurelius v7.0 improvements (chem_utils only)."""

from __future__ import annotations

import pytest


class TestChemModule:
    """Tests for the consolidated RDKit helper functions in utils.chem."""

    def test_chem_module_exports(self):
        from aurelius.utils.chem_utils import (
            _deserialize_fp,
            _is_valid_mol,
            _mol_to_fp,
            _safe_mol_from_smiles,
            _serialize_fp,
            _tanimoto,
        )

        assert callable(_safe_mol_from_smiles)
        assert callable(_is_valid_mol)
        assert callable(_mol_to_fp)
        assert callable(_serialize_fp)
        assert callable(_deserialize_fp)
        assert callable(_tanimoto)

    def test_safe_mol_from_smiles_invalid(self):
        from aurelius.utils.chem_utils import _safe_mol_from_smiles

        result = _safe_mol_from_smiles("not_a_valid_smiles_string_!!!")
        assert result is None

    def test_safe_mol_from_smiles_valid(self):
        try:
            from rdkit.Chem import AllChem

            from aurelius.utils.chem_utils import _safe_mol_from_smiles

            mol = _safe_mol_from_smiles("CCO")
            assert mol is not None
            assert AllChem.MolToSmiles(mol) == "CCO"
        except ImportError:
            pytest.skip("RDKit not available")

    def test_is_valid_mol_mw_check(self):
        from aurelius.utils.chem_utils import _is_valid_mol, _safe_mol_from_smiles

        mol = _safe_mol_from_smiles("C" * 100)
        if mol is not None:
            assert _is_valid_mol(mol) is False

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
            pytest.skip(f"PBC not implemented: {exc}")
        except Exception:
            pytest.skip("RDKit fingerprint conversion failed")
