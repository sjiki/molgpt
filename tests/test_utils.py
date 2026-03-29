"""
Tests for train/utils.py (set_seed, top_k_logits, check_novelty, canonic_smiles,
SmilesEnumerator, Iterator).

The generate/utils.py file is byte-for-byte identical to train/utils.py, so full
coverage here covers both copies.
"""

import math
import threading

import numpy as np
import pytest
import torch

from utils import (
    set_seed,
    top_k_logits,
    check_novelty,
    canonic_smiles,
    SmilesEnumerator,
    Iterator,
    SmilesIterator,
)


# ---------------------------------------------------------------------------
# set_seed
# ---------------------------------------------------------------------------

class TestSetSeed:
    def test_reproducible_random(self):
        set_seed(42)
        import random
        a = random.random()
        set_seed(42)
        b = random.random()
        assert a == b

    def test_reproducible_numpy(self):
        set_seed(7)
        a = np.random.rand(5)
        set_seed(7)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_reproducible_torch(self):
        set_seed(99)
        a = torch.rand(5)
        set_seed(99)
        b = torch.rand(5)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        set_seed(1)
        a = torch.rand(10)
        set_seed(2)
        b = torch.rand(10)
        assert not torch.equal(a, b)


# ---------------------------------------------------------------------------
# top_k_logits
# ---------------------------------------------------------------------------

class TestTopKLogits:
    def test_keeps_exactly_k_non_inf(self):
        logits = torch.tensor([[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]])
        k = 3
        out = top_k_logits(logits, k)
        finite = (out != float("-inf")).sum().item()
        assert finite == k

    def test_top_values_preserved(self):
        logits = torch.tensor([[1.0, 5.0, 3.0, 7.0, 2.0]])
        out = top_k_logits(logits, 2)
        # top-2 are 7.0 and 5.0
        assert out[0, 3].item() == 7.0
        assert out[0, 1].item() == 5.0

    def test_non_top_are_neg_inf(self):
        logits = torch.tensor([[1.0, 5.0, 3.0, 7.0, 2.0]])
        out = top_k_logits(logits, 2)
        assert out[0, 0].item() == float("-inf")
        assert out[0, 2].item() == float("-inf")
        assert out[0, 4].item() == float("-inf")

    def test_k_equals_vocab_size(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        out = top_k_logits(logits, 3)
        assert torch.equal(out, logits)

    def test_does_not_modify_original(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        original = logits.clone()
        top_k_logits(logits, 2)
        assert torch.equal(logits, original)


# ---------------------------------------------------------------------------
# check_novelty
# ---------------------------------------------------------------------------

class TestCheckNovelty:
    def test_empty_gen_returns_zero(self):
        ratio = check_novelty([], {"CCO", "CC"})
        assert ratio == 0.0

    def test_all_novel(self):
        gen = ["CCC", "CCCC", "CCN"]
        train = {"CCO", "CC"}
        ratio = check_novelty(gen, train)
        assert ratio == 100.0

    def test_none_novel(self):
        gen = ["CCO", "CC"]
        train = {"CCO", "CC"}
        ratio = check_novelty(gen, train)
        assert ratio == 0.0

    def test_partial_novelty(self):
        gen = ["CCO", "CCC", "CC", "CCN"]  # CCO and CC are in train
        train = {"CCO", "CC"}
        ratio = check_novelty(gen, train)
        assert ratio == pytest.approx(50.0)

    def test_returns_float(self):
        ratio = check_novelty(["CCC"], {"CCO"})
        assert isinstance(ratio, float)


# ---------------------------------------------------------------------------
# canonic_smiles
# ---------------------------------------------------------------------------

class TestCanonicSmiles:
    def test_valid_smiles_returns_string(self):
        result = canonic_smiles("CCO")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_canonical_form_is_consistent(self):
        # Two different representations of ethanol
        a = canonic_smiles("OCC")
        b = canonic_smiles("CCO")
        assert a == b

    def test_invalid_smiles_returns_none(self):
        result = canonic_smiles("not_a_smiles_$$$$")
        assert result is None

    def test_empty_string_returns_empty_or_none(self):
        # RDKit parses an empty SMILES as an empty molecule; the function
        # returns an empty string rather than None in that case.
        result = canonic_smiles("")
        assert result is None or result == ""

    def test_mol_object_input(self):
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCO")
        result = canonic_smiles(mol)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# SmilesEnumerator
# ---------------------------------------------------------------------------

class TestSmilesEnumerator:
    def test_default_init(self):
        se = SmilesEnumerator()
        assert se.pad == 120
        assert se.leftpad is True
        assert se.isomericSmiles is True
        assert se.enumerate is True
        assert se.canonical is False

    def test_charset_setter_updates_derived_attributes(self):
        se = SmilesEnumerator(charset="ABC")
        assert se._charlen == 3
        assert se._char_to_int == {"A": 0, "B": 1, "C": 2}
        assert se._int_to_char == {0: "A", 1: "B", 2: "C"}

    def test_charset_getter(self):
        se = SmilesEnumerator(charset="XYZ")
        assert se.charset == "XYZ"

    def test_fit_sets_pad_and_charset(self):
        smiles = np.array(["CCO", "CC", "C"])
        se = SmilesEnumerator()
        se.fit(smiles, extra_pad=2)
        # longest is "CCO" (len=3), extra_pad=2 → pad=5
        assert se.pad == 5

    def test_fit_charset_contains_all_chars(self):
        smiles = np.array(["CCO", "CN"])
        se = SmilesEnumerator()
        se.fit(smiles)
        for ch in "CCNO":
            assert ch in se.charset

    def test_randomize_smiles_valid_output(self):
        se = SmilesEnumerator()
        result = se.randomize_smiles("CCO")
        from rdkit import Chem

        mol = Chem.MolFromSmiles(result)
        assert mol is not None

    def test_transform_shape_leftpad(self):
        charset = "CO()"
        se = SmilesEnumerator(charset=charset, pad=10, leftpad=True, enum=False)
        smiles = np.array(["CO"])
        result = se.transform(smiles)
        assert result.shape == (1, 10, 4)

    def test_transform_shape_no_leftpad(self):
        charset = "CO()"
        se = SmilesEnumerator(charset=charset, pad=10, leftpad=False, enum=False)
        smiles = np.array(["CO"])
        result = se.transform(smiles)
        assert result.shape == (1, 10, 4)

    def test_transform_one_hot_encoding(self):
        charset = "CO"
        se = SmilesEnumerator(charset=charset, pad=4, leftpad=False, enum=False)
        smiles = np.array(["CO"])
        result = se.transform(smiles)
        # Each row that has a character should sum to 1
        for row in result[0]:
            row_sum = row.sum()
            assert row_sum in (0, 1)

    def test_reverse_transform_roundtrip(self):
        charset = "CO"
        se = SmilesEnumerator(charset=charset, pad=4, leftpad=False, enum=False)
        smiles = np.array(["CO"])
        encoded = se.transform(smiles)
        decoded = se.reverse_transform(encoded)
        assert decoded[0] == "CO"

    def test_transform_batch_size(self):
        charset = "CO()"
        se = SmilesEnumerator(charset=charset, pad=10, leftpad=False, enum=False)
        smiles = np.array(["CO", "CC"])
        result = se.transform(smiles)
        assert result.shape[0] == 2


# ---------------------------------------------------------------------------
# Iterator
# ---------------------------------------------------------------------------

class TestIterator:
    def test_basic_flow(self):
        it = Iterator(n=10, batch_size=3, shuffle=False, seed=None)
        index_array, current_index, batch_size = next(it.index_generator)
        assert len(index_array) == 3
        assert current_index == 0
        assert batch_size == 3

    def test_raises_when_n_less_than_batch_size(self):
        with pytest.raises(ValueError, match="Input data length is shorter"):
            Iterator(n=2, batch_size=5, shuffle=False, seed=None)

    def test_reset_sets_batch_index_to_zero(self):
        it = Iterator(n=10, batch_size=3, shuffle=False, seed=None)
        it.batch_index = 5
        it.reset()
        assert it.batch_index == 0

    def test_iter_returns_self(self):
        it = Iterator(n=10, batch_size=3, shuffle=False, seed=None)
        assert iter(it) is it

    def test_shuffle_produces_different_order(self):
        it_no_shuffle = Iterator(n=20, batch_size=20, shuffle=False, seed=42)
        it_shuffled = Iterator(n=20, batch_size=20, shuffle=True, seed=42)
        idx_no_shuffle, _, _ = next(it_no_shuffle.index_generator)
        idx_shuffled, _, _ = next(it_shuffled.index_generator)
        # Un-shuffled order must be [0, 1, 2, ..., 19]
        np.testing.assert_array_equal(idx_no_shuffle, np.arange(20))
        # Shuffled order must differ from sequential order
        assert not np.array_equal(idx_shuffled, np.arange(20))

    def test_wraps_around_correctly(self):
        """After exhausting all items, batch_index resets to 0."""
        n, bs = 6, 3
        it = Iterator(n=n, batch_size=bs, shuffle=False, seed=None)
        gen = it.index_generator
        next(gen)  # batch 1
        next(gen)  # batch 2 → wraps
        assert it.batch_index == 0

    def test_seeded_reproducibility(self):
        it1 = Iterator(n=10, batch_size=3, shuffle=True, seed=5)
        it2 = Iterator(n=10, batch_size=3, shuffle=True, seed=5)
        idx1, _, _ = next(it1.index_generator)
        idx2, _, _ = next(it2.index_generator)
        np.testing.assert_array_equal(idx1, idx2)


# ---------------------------------------------------------------------------
# SmilesIterator
# ---------------------------------------------------------------------------

class TestSmilesIterator:
    @pytest.fixture()
    def se(self):
        charset = "CO()"
        return SmilesEnumerator(charset=charset, pad=4, leftpad=False, enum=False)

    def test_basic_next(self, se):
        x = np.array(["CO", "CO", "CO", "CO"])
        y = np.array([0, 1, 0, 1])
        it = SmilesIterator(x, y, se, batch_size=2, shuffle=False, seed=None)
        batch_x, batch_y = it.next()
        assert batch_x.shape[0] == 2
        assert batch_y.shape[0] == 2

    def test_mismatched_xy_raises(self, se):
        x = np.array(["CO", "CO"])
        y = np.array([0, 1, 2])
        with pytest.raises(ValueError):
            SmilesIterator(x, y, se, batch_size=2, shuffle=False, seed=None)

    def test_none_y_returns_only_x(self, se):
        x = np.array(["CO", "CO", "CO", "CO"])
        it = SmilesIterator(x, None, se, batch_size=2, shuffle=False, seed=None)
        result = it.next()
        # Should be just x batch, not a tuple
        assert not isinstance(result, tuple)

    def test_batch_size_matches(self, se):
        x = np.array(["CO"] * 10)
        y = np.array(list(range(10)))
        it = SmilesIterator(x, y, se, batch_size=4, shuffle=False, seed=None)
        batch_x, batch_y = it.next()
        assert len(batch_y) == 4
