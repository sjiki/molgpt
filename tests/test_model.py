"""Unit tests for generate/model.py: GPTConfig and GPT."""
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "generate"))

from model import GPTConfig, GPT


def _make_config(**overrides):
    """Return a minimal GPTConfig suitable for fast CPU tests."""
    defaults = dict(
        vocab_size=32,
        block_size=16,
        num_props=0,
        n_layer=2,
        n_head=2,
        n_embd=64,
        scaffold=False,
        scaffold_maxlen=0,
        lstm=False,
        lstm_layers=0,
    )
    defaults.update(overrides)
    return GPTConfig(**defaults)


class TestGPTConfig:
    def test_basic_attributes(self):
        cfg = GPTConfig(vocab_size=100, block_size=128)
        assert cfg.vocab_size == 100
        assert cfg.block_size == 128

    def test_kwargs_stored_as_attributes(self):
        cfg = GPTConfig(vocab_size=10, block_size=50, n_layer=6)
        assert cfg.n_layer == 6

    def test_class_level_defaults(self):
        cfg = GPTConfig(vocab_size=10, block_size=50)
        assert cfg.embd_pdrop == 0.1
        assert cfg.resid_pdrop == 0.1
        assert cfg.attn_pdrop == 0.1


class TestGPTModel:
    def test_instantiation_no_props_no_scaffold(self):
        model = GPT(_make_config())
        assert model is not None
        assert model.get_block_size() == 16

    def test_forward_no_props_no_scaffold(self):
        model = GPT(_make_config())
        model.eval()
        idx = torch.zeros(2, 8, dtype=torch.long)
        logits, loss, attn_maps = model(idx)
        assert logits.shape == (2, 8, 32)
        assert loss is None
        assert len(attn_maps) == 2  # n_layer=2

    def test_forward_with_targets_returns_loss(self):
        model = GPT(_make_config())
        model.eval()
        idx = torch.zeros(2, 8, dtype=torch.long)
        targets = torch.zeros(2, 8, dtype=torch.long)
        _, loss, _ = model(idx, targets=targets)
        assert loss is not None
        assert loss.item() >= 0.0

    def test_forward_with_num_props(self):
        model = GPT(_make_config(num_props=2))
        model.eval()
        idx = torch.zeros(2, 8, dtype=torch.long)
        # prop shape: (batch, 1, num_props) as used in generate.py
        prop = torch.zeros(2, 1, 2)
        logits, loss, _ = model(idx, prop=prop)
        assert logits.shape == (2, 8, 32)
