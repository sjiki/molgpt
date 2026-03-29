"""
Tests for train/dataset.py (SmileDataset).

The generate/dataset.py file is byte-for-byte identical to train/dataset.py, so
full coverage here covers both copies.
"""

import types
import math

import numpy as np
import pytest
import torch

from dataset import SmileDataset


# ---------------------------------------------------------------------------
# Minimal args stub used throughout
# ---------------------------------------------------------------------------

def _make_args(debug=False):
    args = types.SimpleNamespace()
    args.debug = debug
    return args


# ---------------------------------------------------------------------------
# Shared vocabulary / content for the test molecules
# ---------------------------------------------------------------------------

# A minimal set of valid SMILES that use only a small character set.
# The tokenisation regex in dataset.py splits SMILES into atom/bond tokens.
# Using simple single-letter atoms keeps the vocabulary small and predictable.

import re as _re

SMILES = [
    "CC",
    "CO",
    "CN",
    "CCO",
    "CCN",
]

SCAFFOLDS = [
    "C",
    "C",
    "C",
    "CC",
    "CC",
]

PROPS = [0.5, 0.6, 0.7, 0.8, 0.9]

# Build the vocabulary content string from the test molecules plus the
# padding character '<'.  This mirrors what train.py does (it passes a
# whole_string built from the full dataset) without duplicating any
# hard-coded character list from the source code.
_TOKEN_PATTERN = (
    r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p"
    r"|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)
_regex = _re.compile(_TOKEN_PATTERN)
_all_tokens: list[str] = sorted(
    set(_regex.findall("".join(SMILES + SCAFFOLDS))) | {"<"}
)
WHOLE_STRING = _all_tokens

# max_len must be >= longest tokenised SMILES
MAX_LEN = 10
SCAFFOLD_MAX_LEN = 5


def _make_dataset(smiles=None, props=None, scaffolds=None, block_size=MAX_LEN,
                  scaffold_maxlen=SCAFFOLD_MAX_LEN, aug_prob=0.0, debug=False):
    """Helper to build a SmileDataset with sensible defaults."""
    smiles = smiles or SMILES
    props = props or PROPS
    scaffolds = scaffolds or SCAFFOLDS
    args = _make_args(debug=debug)
    return SmileDataset(
        args,
        smiles,
        WHOLE_STRING,
        block_size,
        aug_prob=aug_prob,
        prop=props,
        scaffold=scaffolds,
        scaffold_maxlen=scaffold_maxlen,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSmileDatasetInit:
    def test_vocab_size_equals_unique_chars_in_content(self):
        ds = _make_dataset()
        assert ds.vocab_size == len(set(WHOLE_STRING))

    def test_stoi_and_itos_are_inverse(self):
        ds = _make_dataset()
        for ch, idx in ds.stoi.items():
            assert ds.itos[idx] == ch

    def test_max_len_set_correctly(self):
        ds = _make_dataset(block_size=12)
        assert ds.max_len == 12

    def test_data_stored(self):
        ds = _make_dataset()
        assert ds.data == SMILES

    def test_prop_stored(self):
        ds = _make_dataset()
        assert ds.prop == PROPS

    def test_scaffold_stored(self):
        ds = _make_dataset()
        assert ds.sca == SCAFFOLDS


class TestSmileDatasetLen:
    def test_len_equals_number_of_smiles(self):
        ds = _make_dataset()
        assert len(ds) == len(SMILES)

    def test_len_with_debug_mode(self):
        ds = _make_dataset(debug=True)
        expected = math.ceil(len(SMILES) / (MAX_LEN + 1))
        assert len(ds) == expected


class TestSmileDatasetGetItem:
    @pytest.fixture()
    def ds(self):
        return _make_dataset(aug_prob=0.0)

    def test_returns_four_elements(self, ds):
        item = ds[0]
        assert len(item) == 4

    def test_x_and_y_are_long_tensors(self, ds):
        x, y, prop, sca = ds[0]
        assert x.dtype == torch.long
        assert y.dtype == torch.long

    def test_prop_is_float_tensor(self, ds):
        _, _, prop, _ = ds[0]
        assert prop.dtype == torch.float

    def test_scaffold_is_long_tensor(self, ds):
        _, _, _, sca = ds[0]
        assert sca.dtype == torch.long

    def test_x_length_is_block_size_minus_one(self, ds):
        x, y, _, _ = ds[0]
        assert x.shape[0] == MAX_LEN - 1

    def test_y_length_is_block_size_minus_one(self, ds):
        x, y, _, _ = ds[0]
        assert y.shape[0] == MAX_LEN - 1

    def test_scaffold_length_equals_scaffold_maxlen(self, ds):
        _, _, _, sca = ds[0]
        assert sca.shape[0] == SCAFFOLD_MAX_LEN

    def test_x_and_y_are_shifted_by_one(self, ds):
        """y[i] should equal x[i+1] for a language model."""
        x, y, _, _ = ds[0]
        # x = dix[:-1], y = dix[1:], so x[1:] == y[:-1]
        assert torch.equal(x[1:], y[:-1])

    def test_all_token_ids_in_vocabulary(self, ds):
        for idx in range(len(ds)):
            x, y, _, sca = ds[idx]
            for tok in x.tolist() + y.tolist() + sca.tolist():
                assert 0 <= tok < ds.vocab_size

    def test_prop_value_matches_input(self, ds):
        _, _, prop, _ = ds[2]
        assert prop.item() == pytest.approx(PROPS[2])

    def test_different_indices_give_different_smiles_tokens(self, ds):
        x0, _, _, _ = ds[0]
        x1, _, _, _ = ds[1]
        # "CC" and "CO" differ, so their token sequences should differ
        assert not torch.equal(x0, x1)


class TestSmileDatasetPadding:
    def test_short_smiles_padded_to_max_len(self):
        """A SMILES shorter than block_size must be padded with '<' tokens."""
        ds = _make_dataset(smiles=["C"] * 5, scaffolds=["C"] * 5, aug_prob=0.0)
        x, y, _, _ = ds[0]
        # The token for '<' should appear (as padding).
        pad_id = ds.stoi["<"]
        # After the real tokens there should be pad tokens in y
        assert pad_id in y.tolist()

    def test_scaffold_padding(self):
        """A scaffold shorter than scaffold_maxlen must be padded with '<'."""
        ds = _make_dataset(scaffolds=["C"] * len(SMILES), aug_prob=0.0)
        _, _, _, sca = ds[0]
        pad_id = ds.stoi["<"]
        assert pad_id in sca.tolist()


class TestSmileDatasetAugmentation:
    def test_aug_prob_zero_no_randomisation(self):
        """With aug_prob=0 the original SMILES is always used."""
        np.random.seed(0)
        ds = _make_dataset(aug_prob=0.0)
        ds_fixed = _make_dataset(aug_prob=0.0)
        x1, _, _, _ = ds[0]
        x2, _, _, _ = ds_fixed[0]
        assert torch.equal(x1, x2)

    def test_aug_prob_one_uses_randomised_smiles(self):
        """With aug_prob=1 the SMILES is (always) randomised – result is still
        a valid tokenised sequence of the correct length.

        We use "CC" (ethane) because its only valid SMILES form is "CC", so
        randomisation cannot introduce characters outside the vocabulary that
        was built from the same molecule.
        """
        np.random.seed(42)
        ds = _make_dataset(smiles=["CC"] * 5, scaffolds=["C"] * 5, aug_prob=1.0)
        x, y, _, _ = ds[0]
        assert x.shape[0] == MAX_LEN - 1
