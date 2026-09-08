"""Tests for the Mamba-2 sequence mixer and its integration into TransformerBlock/LoopedMoEGPT.

Requires the optional ``mamba_ssm`` dependency (CUDA-kernel-compiled; install via WSL, see
``docs/mamba_investigation.md``) AND a CUDA device -- ``mamba_ssm.Mamba2``'s fused Triton path
does not run meaningfully on CPU. Every test in this module is skipped, not failed, when either
is unavailable, so the rest of the suite stays fully runnable on CPU-only/native-Windows setups
where ``mamba_ssm`` cannot be installed at all.
"""

import pytest
import torch

from looped_moe_gpt2.model.config import (
    AttentionConfig,
    LoopConfig,
    MambaConfig,
    MixerType,
    ModelConfig,
    MoEConfig,
    PositionEncodingType,
    SharingPattern,
)
from looped_moe_gpt2.model.gpt import LoopedMoEGPT

mamba_ssm = pytest.importorskip("mamba_ssm", reason="mamba_ssm is an optional, WSL-only dependency")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="mamba_ssm requires a CUDA device")

TINY_VOCAB = 100
# Chosen so (expand * hidden_size) // headdim == 8, mamba_ssm's Triton-kernel stride requirement.
TINY_HIDDEN = 64
TINY_HEADS = 4
TINY_SEQ_LEN = 16
TINY_LAYERS = 4
TINY_MAMBA = MambaConfig(d_state=16, d_conv=4, expand=2, headdim=16, ngroups=1)


def _tiny_config(**overrides) -> ModelConfig:
    """Build a small Mamba-mixer ModelConfig suitable for fast GPU tests, with overrides."""
    defaults = dict(
        vocab_size=TINY_VOCAB,
        hidden_size=TINY_HIDDEN,
        num_layers=TINY_LAYERS,
        num_heads=TINY_HEADS,
        max_seq_len=TINY_SEQ_LEN,
        dropout=0.0,
        position_encoding=PositionEncodingType.NONE,
        attention=AttentionConfig(use_mla=False),
        moe=MoEConfig(enabled=True, num_routed_experts=4, num_shared_experts=1, top_k=2, expert_intermediate_size=16),
        loop=LoopConfig(enabled=True, sharing_pattern=SharingPattern.MIDDLE_CYCLE, num_loops=2, num_unique_prefix_layers=1, num_unique_suffix_layers=1),
        mixer_type=MixerType.MAMBA2,
        mamba=TINY_MAMBA,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def test_mamba_config_rejects_non_multiple_of_8_heads() -> None:
    with pytest.raises(ValueError, match="multiple of 8"):
        ModelConfig(
            hidden_size=32,
            num_heads=4,
            position_encoding=PositionEncodingType.NONE,
            attention=AttentionConfig(use_mla=False),
            mixer_type=MixerType.MAMBA2,
            mamba=MambaConfig(expand=2, headdim=64),  # inner_dim=64, 64/64=1 head, not a multiple of 8
        )


def test_mamba2_requires_mamba_config() -> None:
    with pytest.raises(ValueError, match="mamba config must be provided"):
        ModelConfig(
            hidden_size=64,
            num_heads=4,
            position_encoding=PositionEncodingType.NONE,
            attention=AttentionConfig(use_mla=False),
            mixer_type=MixerType.MAMBA2,
            mamba=None,
        )


def test_mamba2_requires_none_position_encoding() -> None:
    with pytest.raises(ValueError, match="position_encoding=PositionEncodingType.NONE"):
        ModelConfig(
            hidden_size=64,
            num_heads=4,
            position_encoding=PositionEncodingType.LEARNED,
            attention=AttentionConfig(use_mla=False),
            mixer_type=MixerType.MAMBA2,
            mamba=TINY_MAMBA,
        )


def test_pure_mamba2_forward_shapes() -> None:
    config = _tiny_config(mixer_type=MixerType.MAMBA2)
    model = LoopedMoEGPT(config).cuda()
    input_ids = torch.randint(0, TINY_VOCAB, (2, TINY_SEQ_LEN)).cuda()
    logits, loss = model(input_ids)
    assert logits.shape == (2, TINY_SEQ_LEN, TINY_VOCAB)
    assert loss is None


def test_pure_mamba2_backward_pass_populates_gradients() -> None:
    config = _tiny_config(mixer_type=MixerType.MAMBA2)
    model = LoopedMoEGPT(config).cuda()
    input_ids = torch.randint(0, TINY_VOCAB, (2, TINY_SEQ_LEN)).cuda()
    targets = torch.randint(0, TINY_VOCAB, (2, TINY_SEQ_LEN)).cuda()
    _, loss = model(input_ids, targets=targets)
    assert loss is not None
    loss.backward()
    grad_found = any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())
    assert grad_found


def test_hybrid_mamba_attention_forward_shapes() -> None:
    config = _tiny_config(
        mixer_type=MixerType.HYBRID_MAMBA_ATTENTION,
        position_encoding=PositionEncodingType.DECOUPLED_ROPE,
        attention=AttentionConfig(use_mla=True, kv_compression_dim=8, q_compression_dim=12, rope_head_dim=4),
    )
    model = LoopedMoEGPT(config).cuda()
    input_ids = torch.randint(0, TINY_VOCAB, (2, TINY_SEQ_LEN)).cuda()
    logits, loss = model(input_ids)
    assert logits.shape == (2, TINY_SEQ_LEN, TINY_VOCAB)
    assert loss is None


def test_hybrid_mamba_attention_backward_pass_populates_gradients() -> None:
    config = _tiny_config(
        mixer_type=MixerType.HYBRID_MAMBA_ATTENTION,
        position_encoding=PositionEncodingType.DECOUPLED_ROPE,
        attention=AttentionConfig(use_mla=True, kv_compression_dim=8, q_compression_dim=12, rope_head_dim=4),
    )
    model = LoopedMoEGPT(config).cuda()
    input_ids = torch.randint(0, TINY_VOCAB, (2, TINY_SEQ_LEN)).cuda()
    targets = torch.randint(0, TINY_VOCAB, (2, TINY_SEQ_LEN)).cuda()
    _, loss = model(input_ids, targets=targets)
    assert loss is not None
    loss.backward()
    grad_found = any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())
    assert grad_found
