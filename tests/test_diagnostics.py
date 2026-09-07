"""Unit tests for loop diagnostics: per-iteration hidden-state attribution and active-ratio.

The core regression this file guards against: a naive forward-hook implementation that keys a
closure by a fixed iteration index will misattribute hidden states when the SAME block module is
called at multiple loop iterations (as Middle-Cycle sharing does), because
``register_forward_hook`` stacks hooks rather than replacing them -- every stacked hook then
fires on every call, corrupting the recorded per-iteration snapshots. See the fix history in
``src/looped_moe_gpt2/train/diagnostics.py`` for the incident this test suite was written after.
"""

import torch

from looped_moe_gpt2.model.config import (
    AttentionConfig,
    LoopConfig,
    ModelConfig,
    MoEConfig,
    PositionEncodingType,
    SharingPattern,
)
from looped_moe_gpt2.model.gpt import LoopedMoEGPT
from looped_moe_gpt2.train.diagnostics import attach_loop_diagnostics, compute_active_ratio


def _looped_config(num_loops: int = 3) -> ModelConfig:
    return ModelConfig(
        vocab_size=100,
        hidden_size=32,
        num_layers=5,
        num_heads=4,
        max_seq_len=16,
        dropout=0.0,
        position_encoding=PositionEncodingType.DECOUPLED_ROPE,
        attention=AttentionConfig(use_mla=True, kv_compression_dim=8, q_compression_dim=12, rope_head_dim=4),
        moe=MoEConfig(enabled=True, num_routed_experts=4, num_shared_experts=1, top_k=2, expert_intermediate_size=16),
        loop=LoopConfig(
            enabled=True,
            sharing_pattern=SharingPattern.MIDDLE_CYCLE,
            num_loops=num_loops,
            num_unique_prefix_layers=1,
            num_unique_suffix_layers=1,
        ),
    )


def test_diagnostics_record_one_entry_per_iteration() -> None:
    config = _looped_config(num_loops=3)
    model = LoopedMoEGPT(config)
    diag = attach_loop_diagnostics(model)

    x = torch.randint(0, config.vocab_size, (2, 16))
    diag.reset()
    model(x)

    assert sorted(diag.hidden_states_by_iteration.keys()) == [0, 1, 2]


def test_diagnostics_iterations_are_distinguishable_with_perturbed_adaln() -> None:
    """With IterAdaLN's embeddings perturbed away from zero-init, recorded iterations must
    actually differ from each other -- guards against the hook-aliasing bug where every
    iteration slot silently received the same (last-called) block output."""
    config = _looped_config(num_loops=3)
    model = LoopedMoEGPT(config)
    for block in model.blocks:
        if block.use_iter_adaln:
            torch.nn.init.normal_(block.norm1.iter_embedding.weight, std=2.0)
            torch.nn.init.normal_(block.norm2.iter_embedding.weight, std=2.0)

    diag = attach_loop_diagnostics(model)
    x = torch.randint(0, config.vocab_size, (2, 16))
    diag.reset()
    model(x)

    h0 = diag.hidden_states_by_iteration[0]
    h1 = diag.hidden_states_by_iteration[1]
    h2 = diag.hidden_states_by_iteration[2]
    assert not torch.allclose(h0, h1)
    assert not torch.allclose(h1, h2)
    assert not torch.allclose(h0, h2)


def test_diagnostics_survive_unrelated_forward_passes_between_resets() -> None:
    """The hook must not accumulate stale state across forward passes that occur between
    reset() calls (e.g. many training steps run before the next diagnostic snapshot)."""
    config = _looped_config(num_loops=2)
    model = LoopedMoEGPT(config)
    diag = attach_loop_diagnostics(model)

    x = torch.randint(0, config.vocab_size, (2, 16))
    for _ in range(20):
        model(x)  # unrelated forward passes, no reset() in between

    diag.reset()
    model(x)
    assert sorted(diag.hidden_states_by_iteration.keys()) == [0, 1]


def test_cosine_similarity_requires_at_least_two_iterations() -> None:
    from looped_moe_gpt2.train.diagnostics import LoopDiagnostics

    diag = LoopDiagnostics()
    diag.hidden_states_by_iteration[0] = torch.randn(2, 16, 32)
    try:
        diag.consecutive_iteration_cosine_similarities()
        assert False, "Expected ValueError with fewer than two recorded iterations."
    except ValueError:
        pass


def test_compute_active_ratio_is_positive_finite() -> None:
    config = _looped_config(num_loops=2)
    model = LoopedMoEGPT(config)
    x = torch.randint(0, config.vocab_size, (2, 16))
    model(x)
    ratio = compute_active_ratio(model)
    assert ratio > 0
    assert ratio != float("inf")
