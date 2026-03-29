"""Unit tests for generate/utils.py: check_novelty and canonic_smiles.

Note: moses.utils is shimmed in conftest.py so that tests run without
the full molsets installation.
"""
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "generate"))

from utils import check_novelty, canonic_smiles


class TestCheckNovelty:
    def test_empty_generated(self):
        ratio = check_novelty([], {"CC", "CCC"})
        assert ratio == 0.0

    def test_all_novel(self):
        ratio = check_novelty(["CCO", "CCCO"], {"CC", "CCC"})
        assert ratio == 100.0

    def test_none_novel(self):
        ratio = check_novelty(["CC", "CCC"], {"CC", "CCC"})
        assert ratio == 0.0

    def test_partial_novelty(self):
        ratio = check_novelty(["CC", "CCO"], {"CC"})
        assert ratio == 50.0

    def test_returns_float(self):
        ratio = check_novelty(["CCO"], {"CC"})
        assert isinstance(ratio, float)


class TestCanonicSmiles:
    def test_valid_smiles(self):
        result = canonic_smiles("C(C)O")
        assert result is not None
        assert isinstance(result, str)

    def test_invalid_smiles(self):
        result = canonic_smiles("invalid_xyz_not_a_smiles")
        assert result is None

    def test_canonical_form_consistent(self):
        """Different SMILES for the same molecule should produce the same canonical form."""
        s1 = canonic_smiles("CCO")
        s2 = canonic_smiles("OCC")
        assert s1 is not None
        assert s1 == s2

    def test_canonical_form_benzene(self):
        result = canonic_smiles("c1ccccc1")
        assert result is not None
