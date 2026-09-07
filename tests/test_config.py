"""Unit tests for configuration validation logic."""

import pytest

from looped_moe_gpt2.model.config import (
    AttentionConfig,
    LoopConfig,
    ModelConfig,
    MoEConfig,
    PositionEncodingType,
    SharingPattern,
)


def test_default_config_is_valid() -> None:
    """The default ModelConfig() must construct without raising."""
    config = ModelConfig()
    assert config.hidden_size % config.num_heads == 0


def test_hidden_size_not_divisible_by_heads_raises() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(hidden_size=100, num_heads=12)


def test_mla_requires_decoupled_rope() -> None:
    with pytest.raises(ValueError, match="DECOUPLED_ROPE"):
        ModelConfig(
            attention=AttentionConfig(use_mla=True),
            position_encoding=PositionEncodingType.ROPE,
        )


def test_decoupled_rope_requires_mla() -> None:
    with pytest.raises(ValueError, match="use_mla=True"):
        ModelConfig(
            attention=AttentionConfig(use_mla=False),
            position_encoding=PositionEncodingType.DECOUPLED_ROPE,
        )


def test_rope_head_dim_exceeding_head_dim_raises() -> None:
    with pytest.raises(ValueError, match="rope_head_dim"):
        ModelConfig(
            hidden_size=64,
            num_heads=8,  # head_dim = 8
            attention=AttentionConfig(use_mla=True, rope_head_dim=32),
            position_encoding=PositionEncodingType.DECOUPLED_ROPE,
        )


def test_moe_top_k_exceeding_num_experts_raises() -> None:
    with pytest.raises(ValueError, match="top_k"):
        MoEConfig(enabled=True, num_routed_experts=4, top_k=8)


def test_moe_disabled_skips_top_k_validation() -> None:
    """When MoE is disabled, an otherwise-invalid top_k/expert count should not raise."""
    config = MoEConfig(enabled=False, num_routed_experts=1, top_k=8)
    assert config.enabled is False


def test_loop_zero_loops_raises() -> None:
    with pytest.raises(ValueError, match="num_loops"):
        LoopConfig(enabled=True, num_loops=0)


def test_num_layers_too_small_for_loop_body_raises() -> None:
    with pytest.raises(ValueError, match="num_layers"):
        ModelConfig(
            num_layers=2,
            loop=LoopConfig(
                enabled=True,
                sharing_pattern=SharingPattern.MIDDLE_CYCLE,
                num_unique_prefix_layers=1,
                num_unique_suffix_layers=1,
            ),
        )


def test_effective_depth_no_loop_equals_num_layers() -> None:
    config = ModelConfig(num_layers=12, loop=LoopConfig(enabled=False))
    assert config.effective_depth == 12


def test_effective_depth_with_loop() -> None:
    config = ModelConfig(
        num_layers=6,
        loop=LoopConfig(
            enabled=True,
            sharing_pattern=SharingPattern.MIDDLE_CYCLE,
            num_unique_prefix_layers=1,
            num_unique_suffix_layers=1,
            num_loops=3,
        ),
    )
    # 1 prefix + (6 - 1 - 1) * 3 loops + 1 suffix = 1 + 12 + 1 = 14
    assert config.effective_depth == 14


def test_head_dim_property() -> None:
    config = ModelConfig(hidden_size=768, num_heads=12)
    assert config.head_dim == 64
