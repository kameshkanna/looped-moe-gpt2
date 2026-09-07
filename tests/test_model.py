"""Integration tests for the full LoopedMoEGPT model: shapes, gradients, and looping semantics."""

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

TINY_VOCAB = 100
TINY_HIDDEN = 32
TINY_HEADS = 4
TINY_SEQ_LEN = 16
TINY_LAYERS = 4


def _tiny_config(**overrides) -> ModelConfig:
    """Build a small ModelConfig suitable for fast CPU tests, with optional field overrides."""
    defaults = dict(
        vocab_size=TINY_VOCAB,
        hidden_size=TINY_HIDDEN,
        num_layers=TINY_LAYERS,
        num_heads=TINY_HEADS,
        max_seq_len=TINY_SEQ_LEN,
        dropout=0.0,
        position_encoding=PositionEncodingType.DECOUPLED_ROPE,
        attention=AttentionConfig(use_mla=True, kv_compression_dim=8, q_compression_dim=12, rope_head_dim=4),
        moe=MoEConfig(enabled=True, num_routed_experts=4, num_shared_experts=1, top_k=2, expert_intermediate_size=16),
        loop=LoopConfig(enabled=True, sharing_pattern=SharingPattern.MIDDLE_CYCLE, num_loops=2, num_unique_prefix_layers=1, num_unique_suffix_layers=1),
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def test_forward_shapes_looped_moe() -> None:
    config = _tiny_config()
    model = LoopedMoEGPT(config)
    input_ids = torch.randint(0, TINY_VOCAB, (2, TINY_SEQ_LEN))
    logits, loss = model(input_ids)
    assert logits.shape == (2, TINY_SEQ_LEN, TINY_VOCAB)
    assert loss is None


def test_forward_with_targets_returns_scalar_loss() -> None:
    config = _tiny_config()
    model = LoopedMoEGPT(config)
    input_ids = torch.randint(0, TINY_VOCAB, (2, TINY_SEQ_LEN))
    targets = torch.randint(0, TINY_VOCAB, (2, TINY_SEQ_LEN))
    logits, loss = model(input_ids, targets=targets)
    assert loss is not None
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_backward_pass_populates_gradients() -> None:
    config = _tiny_config()
    model = LoopedMoEGPT(config)
    input_ids = torch.randint(0, TINY_VOCAB, (2, TINY_SEQ_LEN))
    targets = torch.randint(0, TINY_VOCAB, (2, TINY_SEQ_LEN))
    _, loss = model(input_ids, targets=targets)
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert any(g > 0 for g in grad_norms)


def test_dense_baseline_no_loop_no_moe() -> None:
    """The SharingPattern.NONE + MoE-disabled config should behave as a plain GPT-2-style model."""
    config = _tiny_config(
        loop=LoopConfig(enabled=False, sharing_pattern=SharingPattern.NONE),
        moe=MoEConfig(enabled=False),
        attention=AttentionConfig(use_mla=False),
        position_encoding=PositionEncodingType.LEARNED,
    )
    model = LoopedMoEGPT(config)
    assert len(model.blocks) == TINY_LAYERS
    assert model.loop_plan == [(i, None) for i in range(TINY_LAYERS)]
    input_ids = torch.randint(0, TINY_VOCAB, (2, TINY_SEQ_LEN))
    logits, _ = model(input_ids)
    assert logits.shape == (2, TINY_SEQ_LEN, TINY_VOCAB)


def test_middle_cycle_shares_weights_across_iterations() -> None:
    """Middle-Cycle sharing must reuse the SAME parameter tensors across loop iterations."""
    config = _tiny_config()
    model = LoopedMoEGPT(config)
    num_loop_body = TINY_LAYERS - config.loop.num_unique_prefix_layers - config.loop.num_unique_suffix_layers
    expected_unique_blocks = config.loop.num_unique_prefix_layers + num_loop_body + config.loop.num_unique_suffix_layers
    assert len(model.blocks) == expected_unique_blocks

    expected_plan_length = config.loop.num_unique_prefix_layers + num_loop_body * config.loop.num_loops + config.loop.num_unique_suffix_layers
    assert len(model.loop_plan) == expected_plan_length
    assert config.effective_depth == expected_plan_length

    block_indices_used = {idx for idx, _ in model.loop_plan}
    assert block_indices_used == set(range(expected_unique_blocks))


def test_iteration_index_changes_output_via_iter_adaln() -> None:
    """With IterAdaLN's embedding perturbed away from zero-init, distinct iterations must diverge."""
    config = _tiny_config()
    model = LoopedMoEGPT(config)
    for block in model.blocks:
        if block.use_iter_adaln:
            torch.nn.init.normal_(block.norm1.iter_embedding.weight, std=1.0)
            torch.nn.init.normal_(block.norm2.iter_embedding.weight, std=1.0)

    x = torch.randn(2, TINY_SEQ_LEN, TINY_HIDDEN)
    looped_block = next(b for b in model.blocks if b.use_iter_adaln)
    out_iter0 = looped_block(x, iteration_idx=0)
    out_iter1 = looped_block(x, iteration_idx=1)
    assert not torch.allclose(out_iter0, out_iter1)


def test_moe_expert_load_recorded_after_forward() -> None:
    config = _tiny_config()
    model = LoopedMoEGPT(config)
    input_ids = torch.randint(0, TINY_VOCAB, (4, TINY_SEQ_LEN))
    model(input_ids)
    load_summary = model.expert_load_summary()
    assert len(load_summary) > 0
    for load in load_summary.values():
        assert load.shape == (config.moe.num_routed_experts,)
        assert torch.isclose(load.sum(), torch.tensor(float(config.moe.top_k)), atol=1e-4)


def test_routing_bias_update_shifts_underloaded_experts_up() -> None:
    config = _tiny_config()
    model = LoopedMoEGPT(config)
    input_ids = torch.randint(0, TINY_VOCAB, (4, TINY_SEQ_LEN))
    model(input_ids)

    moe_block = next(b for b in model.blocks if b.feed_forward.__class__.__name__ == "SparseMoE")
    bias_before = moe_block.feed_forward.routing_bias.clone()
    load = moe_block.feed_forward.last_expert_load
    mean_load = load.mean()

    model.update_all_routing_biases()
    bias_after = moe_block.feed_forward.routing_bias

    for i in range(config.moe.num_routed_experts):
        if load[i] > mean_load:
            assert bias_after[i] < bias_before[i]
        elif load[i] < mean_load:
            assert bias_after[i] > bias_before[i]


def test_generate_produces_correct_length() -> None:
    config = _tiny_config()
    model = LoopedMoEGPT(config)
    prompt = torch.randint(0, TINY_VOCAB, (1, 4))
    generated = model.generate(prompt, max_new_tokens=5, temperature=1.0)
    assert generated.shape == (1, 9)


def test_generate_without_eos_token_id_runs_full_length() -> None:
    """Backward-compatibility guard: omitting eos_token_id (the default, None) must behave
    exactly as before this parameter was added -- always generate the full max_new_tokens."""
    config = _tiny_config()
    model = LoopedMoEGPT(config)
    prompt = torch.randint(0, TINY_VOCAB, (2, 4))
    generated = model.generate(prompt, max_new_tokens=6, temperature=1.0, eos_token_id=None)
    assert generated.shape == (2, 10)


def test_generate_stops_early_when_eos_is_forced() -> None:
    """If the model always samples the eos token (forced here via a near-zero-entropy logit
    distribution on a single-vocab-item model), generation must stop as soon as it's sampled,
    producing a SHORTER sequence than max_new_tokens rather than continuing to pad/sample."""
    config = _tiny_config(vocab_size=2)  # tiny vocab: token 0 or token 1 only
    model = LoopedMoEGPT(config)
    eos_id = 1

    # Force the LM head to always emit an overwhelming logit for `eos_id`, regardless of its
    # input -- a monkeypatched forward rather than trying to hand-craft weight values, since
    # final_norm's LayerNorm before lm_head makes the actual logit magnitude from a given weight
    # value depend on activation statistics that are awkward to control directly in a test.
    def forced_eos_logits(x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        logits = torch.full((batch, seq_len, 2), -100.0)
        logits[..., eos_id] = 100.0
        return logits

    model.lm_head.forward = forced_eos_logits

    prompt = torch.randint(0, 2, (1, 4))
    generated = model.generate(prompt, max_new_tokens=20, temperature=1.0, eos_token_id=eos_id)
    # Should stop at 1 new token (the forced EOS), not run all 20.
    assert generated.shape[1] < 4 + 20
    assert generated[0, 4].item() == eos_id


def test_generate_batched_stops_per_sequence_independently() -> None:
    """When eos_token_id is set and different sequences in a batch would naturally stop at
    different points, the whole batch must run until every sequence has stopped (not just the
    first), and stopped sequences must be padded with eos_token_id rather than keep sampling."""
    config = _tiny_config()
    model = LoopedMoEGPT(config)
    prompt = torch.randint(0, TINY_VOCAB, (3, 4))
    eos_id = 0

    generated = model.generate(prompt, max_new_tokens=10, temperature=1.0, eos_token_id=eos_id)
    # Every sequence must be padded to the SAME final length (the longest-running sequence's
    # stopping point, or max_new_tokens if none stopped naturally).
    assert generated.shape[0] == 3
    assert generated.shape[1] <= 4 + 10

    # For any sequence, once eos_id appears, every subsequent position must also be eos_id
    # (no sampling resumes after a sequence has "finished").
    for row in range(3):
        seq = generated[row, 4:].tolist()
        if eos_id in seq:
            first_eos = seq.index(eos_id)
            assert all(tok == eos_id for tok in seq[first_eos:]), (
                f"Row {row} resumed sampling after EOS: {seq}"
            )


def test_sequence_length_exceeding_max_raises() -> None:
    config = _tiny_config()
    model = LoopedMoEGPT(config)
    input_ids = torch.randint(0, TINY_VOCAB, (1, TINY_SEQ_LEN + 1))
    try:
        model(input_ids)
        assert False, "Expected ValueError for oversized sequence length."
    except ValueError:
        pass
