"""
Shared pytest fixtures and configuration for the MolGPT test suite.

Moses is an optional dependency not always installed, so we mock it in
sys.modules before any project code is imported.
"""

import sys
import types
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock the `moses` package so that modules that do
#   from moses.utils import get_mol
# can be imported even when the real `moses` wheel is not installed.
# ---------------------------------------------------------------------------

def _make_moses_mock():
    """Return a mock `moses` package hierarchy."""
    moses_mock = types.ModuleType("moses")
    utils_mock = types.ModuleType("moses.utils")

    # get_mol: convert a SMILES string to an RDKit Mol using RDKit directly
    from rdkit import Chem

    def get_mol(smiles_or_mol):
        if smiles_or_mol is None:
            return None
        if isinstance(smiles_or_mol, str):
            return Chem.MolFromSmiles(smiles_or_mol)
        return smiles_or_mol

    utils_mock.get_mol = get_mol
    moses_mock.utils = utils_mock

    # Stub out other moses sub-modules that may be imported transitively
    for submod in ("moses.metrics", "moses.metrics.metrics"):
        stub = types.ModuleType(submod)
        sys.modules[submod] = stub

    return moses_mock, utils_mock


_moses_mock, _moses_utils_mock = _make_moses_mock()
sys.modules.setdefault("moses", _moses_mock)
sys.modules.setdefault("moses.utils", _moses_utils_mock)

# ---------------------------------------------------------------------------
# Add project source directories to sys.path so tests can import them.
# ---------------------------------------------------------------------------

import os

_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))

for _subdir in ("train", "generate", "evaluate"):
    _path = os.path.join(_REPO_ROOT, _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

# ---------------------------------------------------------------------------
# Common pytest fixtures
# ---------------------------------------------------------------------------

import pytest
import torch


@pytest.fixture()
def small_gpt_config():
    """Return a small GPTConfig suitable for unit tests (fast to instantiate)."""
    # Import here so the sys.path/mock setup above is already in effect.
    from model import GPTConfig  # train/model.py or generate/model.py

    return GPTConfig(
        vocab_size=10,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=16,
        num_props=0,
        scaffold=False,
        scaffold_maxlen=0,
        lstm=False,
        lstm_layers=0,
    )


@pytest.fixture()
def small_gpt_model(small_gpt_config):
    """Return a small GPT model for unit tests."""
    from model import GPT

    return GPT(small_gpt_config)
