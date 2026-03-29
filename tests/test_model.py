"""
Tests for train/model.py (GPTConfig, GPT1Config, CausalSelfAttention, Block, GPT).

The generate/model.py file is byte-for-byte identical to train/model.py, so
full coverage here covers both copies.
"""

import math
import pytest
import torch
import torch.nn as nn

from model import GPTConfig, GPT1Config, CausalSelfAttention, Block, GPT


# ---------------------------------------------------------------------------
# GPTConfig
# ---------------------------------------------------------------------------

class TestGPTConfig:
    def test_required_attributes(self):
        cfg = GPTConfig(vocab_size=50, block_size=32)
        assert cfg.vocab_size == 50
        assert cfg.block_size == 32

    def test_default_dropout_values(self):
        cfg = GPTConfig(vocab_size=10, block_size=8)
        assert cfg.embd_pdrop == 0.1
        assert cfg.resid_pdrop == 0.1
        assert cfg.attn_pdrop == 0.1

    def test_kwargs_set_as_attributes(self):
        cfg = GPTConfig(
            vocab_size=10,
            block_size=8,
            n_layer=4,
            n_head=4,
            n_embd=64,
            custom_flag=True,
        )
        assert cfg.n_layer == 4
        assert cfg.n_head == 4
        assert cfg.n_embd == 64
        assert cfg.custom_flag is True

    def test_kwargs_override_class_defaults(self):
        cfg = GPTConfig(vocab_size=10, block_size=8, embd_pdrop=0.5)
        assert cfg.embd_pdrop == 0.5


class TestGPT1Config:
    def test_defaults(self):
        cfg = GPT1Config(vocab_size=10, block_size=8)
        assert cfg.n_layer == 12
        assert cfg.n_head == 12
        assert cfg.n_embd == 768

    def test_override_defaults(self):
        cfg = GPT1Config(vocab_size=10, block_size=8, n_layer=6)
        assert cfg.n_layer == 6
        # Other defaults still intact
        assert cfg.n_head == 12


# ---------------------------------------------------------------------------
# CausalSelfAttention
# ---------------------------------------------------------------------------

class TestCausalSelfAttention:
    @pytest.fixture()
    def attn(self):
        cfg = GPTConfig(
            vocab_size=10,
            block_size=16,
            n_layer=1,
            n_head=2,
            n_embd=16,
            num_props=0,
            scaffold_maxlen=0,
        )
        return CausalSelfAttention(cfg)

    def test_output_shape(self, attn):
        B, T, C = 2, 8, 16
        x = torch.randn(B, T, C)
        y, attn_weights = attn(x)
        assert y.shape == (B, T, C)

    def test_attention_weights_shape(self, attn):
        B, T, C = 2, 8, 16
        x = torch.randn(B, T, C)
        _, attn_weights = attn(x)
        # (B, n_head, T, T)
        assert attn_weights.shape == (B, 2, T, T)

    def test_causal_mask_is_lower_triangular(self, attn):
        # The registered mask should be a lower-triangular matrix so that
        # position i cannot attend to positions > i.
        mask = attn.mask.squeeze()
        T = mask.shape[0]
        for i in range(T):
            for j in range(T):
                expected = 1.0 if j <= i else 0.0
                assert mask[i, j].item() == expected, f"mask[{i},{j}] wrong"

    def test_attention_weights_sum_to_one(self, attn):
        attn.eval()
        B, T, C = 1, 4, 16
        x = torch.randn(B, T, C)
        _, attn_weights = attn(x)
        row_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_with_num_props(self):
        cfg = GPTConfig(
            vocab_size=10,
            block_size=16,
            n_layer=1,
            n_head=2,
            n_embd=16,
            num_props=1,
            scaffold_maxlen=0,
        )
        attn = CausalSelfAttention(cfg)
        # mask should be block_size+1 in size
        assert attn.mask.shape[-1] == cfg.block_size + 1

    def test_with_scaffold_maxlen(self):
        cfg = GPTConfig(
            vocab_size=10,
            block_size=16,
            n_layer=1,
            n_head=2,
            n_embd=16,
            num_props=0,
            scaffold_maxlen=5,
        )
        attn = CausalSelfAttention(cfg)
        assert attn.mask.shape[-1] == cfg.block_size + 5


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------

class TestBlock:
    @pytest.fixture()
    def block(self):
        cfg = GPTConfig(
            vocab_size=10,
            block_size=16,
            n_layer=1,
            n_head=2,
            n_embd=16,
            num_props=0,
            scaffold_maxlen=0,
        )
        return Block(cfg)

    def test_output_shape(self, block):
        B, T, C = 2, 8, 16
        x = torch.randn(B, T, C)
        out, attn = block(x)
        assert out.shape == (B, T, C)

    def test_attention_maps_shape(self, block):
        B, T, C = 2, 8, 16
        x = torch.randn(B, T, C)
        _, attn = block(x)
        assert attn.shape == (B, 2, T, T)

    def test_residual_connection_changes_input(self, block):
        B, T, C = 2, 8, 16
        x = torch.randn(B, T, C)
        out, _ = block(x)
        # Output should differ from input (the block adds a residual)
        assert not torch.allclose(out, x)


# ---------------------------------------------------------------------------
# GPT (full model)
# ---------------------------------------------------------------------------

def _make_config(**kwargs):
    defaults = dict(
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
    defaults.update(kwargs)
    return GPTConfig(**defaults)


class TestGPTModel:
    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def test_get_block_size(self):
        model = GPT(_make_config())
        assert model.get_block_size() == 16

    def test_init_weights_linear_std(self):
        """Linear weights should be initialised with std≈0.02."""
        model = GPT(_make_config())
        for m in model.modules():
            if isinstance(m, nn.Linear):
                # std should be ~0.02; allow generous tolerance
                assert m.weight.data.std().item() < 0.2
                break

    def test_init_weights_layernorm(self):
        model = GPT(_make_config())
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                assert torch.all(m.weight.data == 1.0)
                assert torch.all(m.bias.data == 0.0)
                break

    def test_prop_nn_created_when_num_props_nonzero(self):
        model = GPT(_make_config(num_props=2))
        assert hasattr(model, "prop_nn")

    def test_prop_nn_not_created_when_num_props_zero(self):
        model = GPT(_make_config(num_props=0))
        assert not hasattr(model, "prop_nn")

    def test_lstm_created_when_lstm_true(self):
        model = GPT(_make_config(scaffold=True, scaffold_maxlen=4, lstm=True, lstm_layers=1))
        assert hasattr(model, "lstm")

    def test_lstm_not_created_when_lstm_false(self):
        model = GPT(_make_config())
        assert not hasattr(model, "lstm")

    # ------------------------------------------------------------------
    # Forward pass – plain (no props, no scaffold)
    # ------------------------------------------------------------------

    def test_forward_no_props_no_scaffold(self):
        model = GPT(_make_config())
        model.eval()
        B, T = 2, 8
        idx = torch.randint(0, 10, (B, T))
        logits, loss, attn_maps = model(idx)
        assert logits.shape == (B, T, 10)
        assert loss is None
        assert len(attn_maps) == 2  # n_layer

    def test_forward_with_targets_returns_loss(self):
        model = GPT(_make_config())
        model.eval()
        B, T = 2, 8
        idx = torch.randint(0, 10, (B, T))
        targets = torch.randint(0, 10, (B, T))
        logits, loss, _ = model(idx, targets=targets)
        assert loss is not None
        assert loss.item() >= 0.0

    def test_forward_block_size_exceeded_raises(self):
        model = GPT(_make_config(block_size=8))
        model.eval()
        idx = torch.randint(0, 10, (1, 9))  # 9 > block_size=8
        with pytest.raises(AssertionError):
            model(idx)

    # ------------------------------------------------------------------
    # Forward pass – with num_props
    # ------------------------------------------------------------------

    def test_forward_with_single_prop(self):
        model = GPT(_make_config(num_props=1))
        model.eval()
        B, T = 2, 8
        idx = torch.randint(0, 10, (B, T))
        prop = torch.randn(B, 1)
        logits, loss, _ = model(idx, prop=prop)
        assert logits.shape == (B, T, 10)

    def test_forward_with_multi_props(self):
        model = GPT(_make_config(num_props=3))
        model.eval()
        B, T = 2, 8
        idx = torch.randint(0, 10, (B, T))
        prop = torch.randn(B, 1, 3)
        logits, loss, _ = model(idx, prop=prop)
        assert logits.shape == (B, T, 10)

    def test_forward_wrong_num_props_raises(self):
        model = GPT(_make_config(num_props=2))
        model.eval()
        B, T = 2, 8
        idx = torch.randint(0, 10, (B, T))
        prop = torch.randn(B, 1)  # wrong last dim (1 instead of 2)
        with pytest.raises(AssertionError):
            model(idx, prop=prop)

    # ------------------------------------------------------------------
    # Forward pass – with scaffold (no lstm)
    # ------------------------------------------------------------------

    def test_forward_with_scaffold_no_lstm(self):
        scaf_len = 4
        model = GPT(_make_config(scaffold=True, scaffold_maxlen=scaf_len))
        model.eval()
        B, T = 2, 8
        idx = torch.randint(0, 10, (B, T))
        scaffold = torch.randint(0, 10, (B, scaf_len))
        logits, loss, _ = model(idx, scaffold=scaffold)
        assert logits.shape == (B, T, 10)

    # ------------------------------------------------------------------
    # Forward pass – with both props and scaffold
    # ------------------------------------------------------------------

    def test_forward_with_props_and_scaffold(self):
        scaf_len = 4
        model = GPT(_make_config(num_props=1, scaffold=True, scaffold_maxlen=scaf_len))
        model.eval()
        B, T = 2, 8
        idx = torch.randint(0, 10, (B, T))
        prop = torch.randn(B, 1)
        scaffold = torch.randint(0, 10, (B, scaf_len))
        logits, loss, _ = model(idx, prop=prop, scaffold=scaffold)
        assert logits.shape == (B, T, 10)

    # ------------------------------------------------------------------
    # configure_optimizers
    # ------------------------------------------------------------------

    def test_configure_optimizers_returns_adamw(self, small_gpt_model):
        from trainer import TrainerConfig

        tconf = TrainerConfig(
            max_epochs=1,
            batch_size=2,
            learning_rate=3e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
        optimizer = small_gpt_model.configure_optimizers(tconf)
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_configure_optimizers_no_param_overlap(self, small_gpt_model):
        from trainer import TrainerConfig

        tconf = TrainerConfig(
            max_epochs=1,
            batch_size=2,
            learning_rate=3e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
        # The function uses assertions internally; if it doesn't raise, params
        # are cleanly partitioned.
        optimizer = small_gpt_model.configure_optimizers(tconf)
        decay_ids = {id(p) for g in optimizer.param_groups if g["weight_decay"] > 0 for p in g["params"]}
        no_decay_ids = {id(p) for g in optimizer.param_groups if g["weight_decay"] == 0 for p in g["params"]}
        assert decay_ids.isdisjoint(no_decay_ids)

    def test_configure_optimizers_covers_all_params(self, small_gpt_model):
        from trainer import TrainerConfig

        tconf = TrainerConfig(
            max_epochs=1,
            batch_size=2,
            learning_rate=3e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
        optimizer = small_gpt_model.configure_optimizers(tconf)
        opt_param_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
        model_param_ids = {id(p) for p in small_gpt_model.parameters()}
        assert opt_param_ids == model_param_ids

    # ------------------------------------------------------------------
    # Attention maps
    # ------------------------------------------------------------------

    def test_attn_maps_count_equals_n_layers(self):
        model = GPT(_make_config(n_layer=3))
        model.eval()
        idx = torch.randint(0, 10, (1, 5))
        _, _, attn_maps = model(idx)
        assert len(attn_maps) == 3
