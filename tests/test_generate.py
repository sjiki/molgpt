"""Unit tests for safe_mol_properties in generate/generate.py.

moses and sascorer are shimmed in conftest.py so the tests run without
the full molsets installation or the RDKit SA_Score contrib directory.
"""
import sys
import os
import unittest.mock as mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "generate"))

from generate import safe_mol_properties  # noqa: E402 (must come after path setup)
from rdkit import Chem


def _mol(smiles: str):
    return Chem.MolFromSmiles(smiles)


class TestSafeMolProperties:
    def test_returns_all_keys(self):
        props = safe_mol_properties(_mol("CCO"))
        assert set(props.keys()) == {"qed", "sas", "logp", "tpsa"}

    def test_valid_mol_all_values_present(self):
        props = safe_mol_properties(_mol("CCO"))
        assert props["qed"] is not None
        assert props["sas"] is not None
        assert props["logp"] is not None
        assert props["tpsa"] is not None

    def test_qed_in_unit_range(self):
        props = safe_mol_properties(_mol("CCO"))
        assert 0.0 <= props["qed"] <= 1.0

    def test_logp_is_float(self):
        props = safe_mol_properties(_mol("CCO"))
        assert isinstance(props["logp"], float)

    def test_tpsa_is_non_negative(self):
        props = safe_mol_properties(_mol("CCO"))
        assert props["tpsa"] >= 0.0

    def test_sas_uses_mock(self):
        props = safe_mol_properties(_mol("CCO"))
        assert props["sas"] == 2.5

    def test_qed_error_returns_none(self):
        """ValueError from QED should leave qed=None without crashing."""
        with mock.patch("generate.QED.qed", side_effect=ValueError("bad")):
            props = safe_mol_properties(_mol("CCO"))
        assert props["qed"] is None
        assert props["logp"] is not None  # other properties unaffected

    def test_logp_error_returns_none(self):
        with mock.patch("generate.Crippen.MolLogP", side_effect=RuntimeError("err")):
            props = safe_mol_properties(_mol("CCO"))
        assert props["logp"] is None

    def test_tpsa_error_returns_none(self):
        with mock.patch("generate.CalcTPSA", side_effect=AttributeError("err")):
            props = safe_mol_properties(_mol("CCO"))
        assert props["tpsa"] is None
