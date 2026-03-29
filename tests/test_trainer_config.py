"""
Tests for train/trainer.py (TrainerConfig).

The Trainer class itself requires a full training loop with a wandb integration
and GPU setup; its behaviour is better exercised via integration tests.
Here we focus on the configuration class and the save_checkpoint utility.
"""

import os
import tempfile

import pytest
import torch

from trainer import TrainerConfig, Trainer


# ---------------------------------------------------------------------------
# TrainerConfig
# ---------------------------------------------------------------------------

class TestTrainerConfig:
    def test_default_max_epochs(self):
        cfg = TrainerConfig()
        assert cfg.max_epochs == 10

    def test_default_batch_size(self):
        cfg = TrainerConfig()
        assert cfg.batch_size == 64

    def test_default_learning_rate(self):
        cfg = TrainerConfig()
        assert cfg.learning_rate == pytest.approx(3e-4)

    def test_default_betas(self):
        cfg = TrainerConfig()
        assert cfg.betas == (0.9, 0.95)

    def test_default_grad_norm_clip(self):
        cfg = TrainerConfig()
        assert cfg.grad_norm_clip == pytest.approx(1.0)

    def test_default_weight_decay(self):
        cfg = TrainerConfig()
        assert cfg.weight_decay == pytest.approx(0.1)

    def test_default_lr_decay_false(self):
        cfg = TrainerConfig()
        assert cfg.lr_decay is False

    def test_default_ckpt_path_none(self):
        cfg = TrainerConfig()
        assert cfg.ckpt_path is None

    def test_default_num_workers(self):
        cfg = TrainerConfig()
        assert cfg.num_workers == 0

    def test_kwargs_override_defaults(self):
        cfg = TrainerConfig(
            max_epochs=5,
            batch_size=128,
            learning_rate=1e-3,
            weight_decay=0.01,
        )
        assert cfg.max_epochs == 5
        assert cfg.batch_size == 128
        assert cfg.learning_rate == pytest.approx(1e-3)
        assert cfg.weight_decay == pytest.approx(0.01)

    def test_arbitrary_kwargs_stored_as_attributes(self):
        cfg = TrainerConfig(custom_key="hello", another=42)
        assert cfg.custom_key == "hello"
        assert cfg.another == 42

    def test_ckpt_path_set(self):
        cfg = TrainerConfig(ckpt_path="/tmp/model.pt")
        assert cfg.ckpt_path == "/tmp/model.pt"

    def test_warmup_and_final_tokens_defaults(self):
        cfg = TrainerConfig()
        assert cfg.warmup_tokens == pytest.approx(375e6)
        assert cfg.final_tokens == pytest.approx(260e9)


# ---------------------------------------------------------------------------
# Trainer.save_checkpoint
# ---------------------------------------------------------------------------

class TestTrainerSaveCheckpoint:
    def test_save_checkpoint_writes_file(self, small_gpt_model):
        """save_checkpoint should persist the model's state_dict to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "model.pt")
            cfg = TrainerConfig(ckpt_path=ckpt)
            trainer = Trainer(
                model=small_gpt_model,
                train_dataset=None,
                test_dataset=None,
                config=cfg,
                stoi={},
                itos={},
            )
            trainer.save_checkpoint()
            assert os.path.exists(ckpt)

    def test_save_checkpoint_restores_weights(self, small_gpt_model):
        """The saved state_dict should exactly match the model's parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "model.pt")
            cfg = TrainerConfig(ckpt_path=ckpt)
            trainer = Trainer(
                model=small_gpt_model,
                train_dataset=None,
                test_dataset=None,
                config=cfg,
                stoi={},
                itos={},
            )
            trainer.save_checkpoint()

            saved = torch.load(ckpt, map_location="cpu")
            for name, param in small_gpt_model.named_parameters():
                assert name in saved
                assert torch.equal(param.data.cpu(), saved[name])
