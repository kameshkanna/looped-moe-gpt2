"""Integration tests for Trainer checkpoint resume: correctness of the actual resume mechanics.

These tests exist because a resume path that silently does the wrong thing (resets to step 0,
loses optimizer momentum, restarts the LR schedule) is worse than no resume support at all --
it looks like it worked but quietly produces a different training trajectory than an
uninterrupted run. Each test asserts on the SPECIFIC state that must survive a resume, not just
"it doesn't crash."
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from looped_moe_gpt2.model.config import (
    AttentionConfig,
    LoopConfig,
    ModelConfig,
    MoEConfig,
    PositionEncodingType,
    SharingPattern,
)
from looped_moe_gpt2.train.config import TrainConfig
from looped_moe_gpt2.train.trainer import Trainer

TINY_VOCAB = 100
TINY_HIDDEN = 32
TINY_SEQ_LEN = 16


@pytest.fixture
def tiny_token_bins(tmp_path: Path) -> tuple[Path, Path]:
    """Write small train/val token binaries large enough for a few dozen samples."""
    rng = np.random.default_rng(0)
    train_tokens = rng.integers(0, TINY_VOCAB, size=5000, dtype=np.uint16)
    val_tokens = rng.integers(0, TINY_VOCAB, size=1000, dtype=np.uint16)
    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)
    return train_path, val_path


def _tiny_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=TINY_VOCAB,
        hidden_size=TINY_HIDDEN,
        num_layers=4,
        num_heads=4,
        max_seq_len=TINY_SEQ_LEN,
        dropout=0.0,
        position_encoding=PositionEncodingType.DECOUPLED_ROPE,
        attention=AttentionConfig(use_mla=True, kv_compression_dim=8, q_compression_dim=12, rope_head_dim=4),
        moe=MoEConfig(enabled=True, num_routed_experts=4, num_shared_experts=1, top_k=2, expert_intermediate_size=16),
        loop=LoopConfig(enabled=True, sharing_pattern=SharingPattern.MIDDLE_CYCLE, num_loops=2, num_unique_prefix_layers=1, num_unique_suffix_layers=1),
    )


def _tiny_train_config(tmp_path: Path, bins: tuple[Path, Path], max_steps: int) -> TrainConfig:
    train_bin, val_bin = bins
    return TrainConfig(
        train_bin_path=train_bin,
        val_bin_path=val_bin,
        output_dir=tmp_path / "runs",
        batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        learning_rate=1e-3,
        min_learning_rate=1e-4,
        warmup_steps=1,
        eval_interval=2,
        eval_iters=2,
        log_interval=1,
        checkpoint_interval=2,
        use_amp=False,
    )


def test_resumed_training_continues_from_checkpointed_step(tmp_path: Path, tiny_token_bins) -> None:
    """A resumed run's first executed step must be the checkpoint's step, not 0."""
    model_config = _tiny_model_config()
    train_config = _tiny_train_config(tmp_path, tiny_token_bins, max_steps=4)
    trainer = Trainer(model_config=model_config, train_config=train_config)
    trainer.train()

    checkpoint_path = train_config.output_dir / "checkpoint_step4.pt"
    assert checkpoint_path.exists()

    train_config_extended = _tiny_train_config(tmp_path, tiny_token_bins, max_steps=6)
    resumed_trainer = Trainer(model_config=model_config, train_config=train_config_extended, resume_from=checkpoint_path)
    assert resumed_trainer.start_step == 4


def test_resume_restores_optimizer_state_not_fresh_init(tmp_path: Path, tiny_token_bins) -> None:
    """Resumed optimizer must carry over Adam moment estimates, not restart from zero."""
    model_config = _tiny_model_config()
    train_config = _tiny_train_config(tmp_path, tiny_token_bins, max_steps=4)
    trainer = Trainer(model_config=model_config, train_config=train_config)
    trainer.train()

    checkpoint_path = train_config.output_dir / "checkpoint_step4.pt"
    fresh_trainer = Trainer(model_config=model_config, train_config=train_config)
    resumed_trainer = Trainer(model_config=model_config, train_config=train_config, resume_from=checkpoint_path)

    fresh_state = fresh_trainer.optimizer.state_dict()["state"]
    resumed_state = resumed_trainer.optimizer.state_dict()["state"]
    assert len(resumed_state) > 0, "Resumed optimizer has no per-parameter state at all."
    assert resumed_state != fresh_state or len(fresh_state) == 0, (
        "Resumed optimizer state is identical to a freshly-initialized optimizer -- "
        "checkpoint's optimizer_state_dict was not actually applied."
    )


def test_resume_restores_model_weights_not_reinitialized(tmp_path: Path, tiny_token_bins) -> None:
    """Resumed model weights must match the checkpoint, not a fresh random init."""
    model_config = _tiny_model_config()
    train_config = _tiny_train_config(tmp_path, tiny_token_bins, max_steps=4)
    trainer = Trainer(model_config=model_config, train_config=train_config)
    trainer.train()
    trained_first_param = next(iter(trainer.model.parameters())).detach().clone()

    checkpoint_path = train_config.output_dir / "checkpoint_step4.pt"
    resumed_trainer = Trainer(model_config=model_config, train_config=train_config, resume_from=checkpoint_path)
    resumed_first_param = next(iter(resumed_trainer.model.parameters())).detach()

    assert torch.allclose(trained_first_param, resumed_first_param), (
        "Resumed model's parameters differ from the checkpoint -- weights were not restored."
    )


def test_resume_beyond_max_steps_raises(tmp_path: Path, tiny_token_bins) -> None:
    """Resuming from a checkpoint whose step already meets/exceeds max_steps must fail loudly,
    not silently no-op or crash inside the training loop."""
    model_config = _tiny_model_config()
    train_config = _tiny_train_config(tmp_path, tiny_token_bins, max_steps=4)
    trainer = Trainer(model_config=model_config, train_config=train_config)
    trainer.train()

    checkpoint_path = train_config.output_dir / "checkpoint_step4.pt"
    same_max_steps_config = _tiny_train_config(tmp_path, tiny_token_bins, max_steps=4)
    resumed_trainer = Trainer(model_config=model_config, train_config=same_max_steps_config, resume_from=checkpoint_path)

    with pytest.raises(ValueError, match="already completed"):
        resumed_trainer.train()


def test_resume_with_mismatched_config_raises_clear_error(tmp_path: Path, tiny_token_bins) -> None:
    """Resuming with a DIFFERENT architecture config than the checkpoint was trained with must
    fail with a clear error, not silently load a partially-mismatched state_dict."""
    model_config = _tiny_model_config()
    train_config = _tiny_train_config(tmp_path, tiny_token_bins, max_steps=2)
    trainer = Trainer(model_config=model_config, train_config=train_config)
    trainer.train()

    checkpoint_path = train_config.output_dir / "checkpoint_step2.pt"
    mismatched_config = ModelConfig(
        vocab_size=TINY_VOCAB,
        hidden_size=64,  # different hidden size -- shapes won't match
        num_layers=4,
        num_heads=4,
        max_seq_len=TINY_SEQ_LEN,
        attention=AttentionConfig(use_mla=True, kv_compression_dim=8, q_compression_dim=12, rope_head_dim=4),
        moe=MoEConfig(enabled=True, num_routed_experts=4, num_shared_experts=1, top_k=2, expert_intermediate_size=16),
        loop=LoopConfig(enabled=True, sharing_pattern=SharingPattern.MIDDLE_CYCLE, num_loops=2, num_unique_prefix_layers=1, num_unique_suffix_layers=1),
        position_encoding=PositionEncodingType.DECOUPLED_ROPE,
    )

    with pytest.raises(RuntimeError, match="SAME"):
        Trainer(model_config=mismatched_config, train_config=train_config, resume_from=checkpoint_path)


def test_resumed_run_reaches_new_max_steps(tmp_path: Path, tiny_token_bins) -> None:
    """End-to-end: a resumed run must actually run the remaining steps and checkpoint at the end."""
    model_config = _tiny_model_config()
    train_config = _tiny_train_config(tmp_path, tiny_token_bins, max_steps=4)
    trainer = Trainer(model_config=model_config, train_config=train_config)
    trainer.train()

    checkpoint_path = train_config.output_dir / "checkpoint_step4.pt"
    train_config_extended = _tiny_train_config(tmp_path, tiny_token_bins, max_steps=6)
    resumed_trainer = Trainer(model_config=model_config, train_config=train_config_extended, resume_from=checkpoint_path)
    resumed_trainer.train()

    assert (train_config_extended.output_dir / "checkpoint_step6.pt").exists()


def test_checkpoint_state_dict_keys_have_no_compile_prefix(tmp_path: Path, tiny_token_bins) -> None:
    """Regression guard: torch.compile(model) prefixes state_dict keys with '_orig_mod.' if the
    COMPILED wrapper is what gets saved/loaded. Trainer must always save/load via the
    UNCOMPILED `self.model`, so checkpoints stay compatible regardless of whether a given run
    (or a later resume of it) has compile_model enabled -- otherwise resuming a compiled run's
    checkpoint into a non-compiled Trainer (or vice versa) would silently fail to load any
    weights (a state_dict with all-mismatched keys loads as a no-op under strict=False, or
    raises under strict=True -- either way it is NOT the intended resume behavior)."""
    model_config = _tiny_model_config()
    train_config = _tiny_train_config(tmp_path, tiny_token_bins, max_steps=2)
    trainer = Trainer(model_config=model_config, train_config=train_config)
    assert trainer.model is trainer.model_forward, "compile_model=False must make these the same object."
    trainer.train()

    checkpoint_path = train_config.output_dir / "checkpoint_step2.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict_keys = list(checkpoint["model_state_dict"].keys())
    assert len(state_dict_keys) > 0
    assert not any(k.startswith("_orig_mod.") for k in state_dict_keys), (
        f"Checkpoint state_dict keys are prefixed with '_orig_mod.' -- the compiled wrapper was "
        f"saved instead of the uncompiled model. Sample keys: {state_dict_keys[:3]}"
    )

    # A fresh, uncompiled Trainer must be able to resume from this checkpoint without any key
    # mismatch, regardless of whether compile_model was used for the run that produced it.
    resumed_trainer = Trainer(model_config=model_config, train_config=train_config, resume_from=checkpoint_path)
    assert resumed_trainer.start_step == 2
