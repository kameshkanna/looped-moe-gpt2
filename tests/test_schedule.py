"""Unit tests for LR and MoE bias-update-speed schedules."""

from pathlib import Path

from looped_moe_gpt2.train.config import TrainConfig
from looped_moe_gpt2.train.schedule import cosine_lr_with_warmup, moe_bias_update_speed


def _train_config(**overrides) -> TrainConfig:
    defaults = dict(
        train_bin_path=Path("dummy_train.bin"),
        val_bin_path=Path("dummy_val.bin"),
        output_dir=Path("dummy_out"),
        max_steps=1000,
        warmup_steps=100,
        learning_rate=1e-3,
        min_learning_rate=1e-4,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def test_lr_warmup_is_linear_and_increasing() -> None:
    config = _train_config()
    lr_start = cosine_lr_with_warmup(0, config)
    lr_mid_warmup = cosine_lr_with_warmup(50, config)
    lr_end_warmup = cosine_lr_with_warmup(99, config)
    assert lr_start < lr_mid_warmup < lr_end_warmup
    assert lr_end_warmup <= config.learning_rate


def test_lr_peaks_near_learning_rate_after_warmup() -> None:
    config = _train_config()
    lr_at_warmup_boundary = cosine_lr_with_warmup(config.warmup_steps, config)
    assert abs(lr_at_warmup_boundary - config.learning_rate) < 1e-6


def test_lr_decays_to_min_at_max_steps() -> None:
    config = _train_config()
    lr_at_end = cosine_lr_with_warmup(config.max_steps, config)
    assert lr_at_end == config.min_learning_rate


def test_lr_monotonically_decreasing_after_warmup() -> None:
    config = _train_config()
    lrs = [cosine_lr_with_warmup(s, config) for s in range(config.warmup_steps, config.max_steps, 50)]
    assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1))


def test_bias_speed_constant_before_decay_window() -> None:
    speed = moe_bias_update_speed(step=0, max_steps=1000, initial_speed=1e-3, decay_frac=0.1)
    assert speed == 1e-3


def test_bias_speed_zero_at_final_step() -> None:
    speed = moe_bias_update_speed(step=1000, max_steps=1000, initial_speed=1e-3, decay_frac=0.1)
    assert speed == 0.0


def test_bias_speed_linearly_decays_within_window() -> None:
    max_steps = 1000
    decay_frac = 0.1
    decay_start = max_steps * (1 - decay_frac)
    speed_at_start = moe_bias_update_speed(int(decay_start), max_steps, 1e-3, decay_frac)
    speed_at_mid = moe_bias_update_speed(int(decay_start + (max_steps - decay_start) / 2), max_steps, 1e-3, decay_frac)
    speed_at_end = moe_bias_update_speed(max_steps, max_steps, 1e-3, decay_frac)
    assert speed_at_start > speed_at_mid > speed_at_end


def test_bias_speed_disabled_with_zero_decay_frac() -> None:
    speed = moe_bias_update_speed(step=999, max_steps=1000, initial_speed=1e-3, decay_frac=0.0)
    assert speed == 1e-3
