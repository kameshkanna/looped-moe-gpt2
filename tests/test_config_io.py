"""Unit tests for YAML config loading, exercised against the actual repo config files.

Loading the real configs/*.yaml files here means a YAML typo or schema drift breaks CI
immediately, rather than being discovered only when a training run is launched.
"""

from pathlib import Path

import pytest

from looped_moe_gpt2.model.config import PositionEncodingType, SharingPattern
from looped_moe_gpt2.utils.config_io import load_model_config, load_train_config

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.mark.parametrize(
    "config_name",
    [
        "00_dense_baseline.yaml",
        "01_dense_rope.yaml",
        "02_looped_dense.yaml",
        "03_looped_moe_full.yaml",
        "04_looped_moe_partial.yaml",
    ],
)
def test_all_repo_configs_load_as_valid_model_config(config_name: str) -> None:
    config_path = CONFIGS_DIR / config_name
    model_config = load_model_config(config_path)
    assert model_config.hidden_size % model_config.num_heads == 0
    assert model_config.effective_depth >= model_config.num_layers


@pytest.mark.parametrize(
    "config_name",
    [
        "00_dense_baseline.yaml",
        "01_dense_rope.yaml",
        "02_looped_dense.yaml",
        "03_looped_moe_full.yaml",
        "04_looped_moe_partial.yaml",
    ],
)
def test_all_repo_configs_load_as_valid_train_config(config_name: str) -> None:
    config_path = CONFIGS_DIR / config_name
    train_config = load_train_config(config_path)
    assert train_config.max_steps > 0
    assert train_config.warmup_steps < train_config.max_steps


def test_dense_baseline_uses_learned_positions_and_no_loop() -> None:
    config = load_model_config(CONFIGS_DIR / "00_dense_baseline.yaml")
    assert config.position_encoding == PositionEncodingType.LEARNED
    assert config.loop.enabled is False
    assert config.moe.enabled is False


def test_looped_moe_full_uses_decoupled_rope_and_middle_cycle() -> None:
    config = load_model_config(CONFIGS_DIR / "03_looped_moe_full.yaml")
    assert config.position_encoding == PositionEncodingType.DECOUPLED_ROPE
    assert config.attention.use_mla is True
    assert config.moe.enabled is True
    assert config.loop.sharing_pattern == SharingPattern.MIDDLE_CYCLE


def test_missing_config_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_model_config(CONFIGS_DIR / "does_not_exist.yaml")
