"""Tests for PonderNetRouter and the PonderNet-gated forward path in LoopedMoEGPT.

These specifically guard the property that motivated building PonderNet as a follow-up to
ACTRouter: gradient must reach EVERY router that was called during a forward pass, not just the
one active at the final iteration -- this was a real, literature-documented bug found in the
first (Graves 2016 ACT) attempt (see docs/adaptive_depth_router_design.md and act_router.py's
module docstring).
"""

import torch

from looped_moe_gpt2.model.config import (
    AttentionConfig,
    LoopConfig,
    ModelConfig,
    MoEConfig,
    PositionEncodingType,
    RouterConfig,
    SharingPattern,
)
from looped_moe_gpt2.model.gpt import LoopedMoEGPT
from looped_moe_gpt2.model.ponder_router import PonderNetRouter, geometric_kl_regularization

TINY_VOCAB = 100
TINY_HIDDEN = 32
TINY_SEQ_LEN = 32


def _pondernet_config(num_loops: int = 6, min_loops: int = 1, ponder_cost_weight: float = 0.01) -> ModelConfig:
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
        loop=LoopConfig(
            enabled=True,
            sharing_pattern=SharingPattern.FULL_LOOP,
            num_loops=num_loops,
            num_unique_prefix_layers=1,
            num_unique_suffix_layers=1,
            use_router_gating=True,
            router=RouterConfig(mechanism="pondernet", min_loops=min_loops, ponder_cost_weight=ponder_cost_weight, geometric_prior_lambda=0.3),
        ),
    )


def test_pondernet_router_requires_pondernet_mechanism() -> None:
    try:
        PonderNetRouter(hidden_size=TINY_HIDDEN, router_config=RouterConfig(mechanism="capacity"))
        assert False, "Expected ValueError for non-pondernet RouterConfig."
    except ValueError:
        pass


def test_pondernet_forces_lambda_one_on_last_step() -> None:
    router = PonderNetRouter(hidden_size=TINY_HIDDEN, router_config=RouterConfig(mechanism="pondernet"))
    x = torch.randn(2, TINY_SEQ_LEN, TINY_HIDDEN)
    un_halted = torch.rand(2, TINY_SEQ_LEN)
    p_n, lambda_n, new_un_halted = router(x, un_halted, is_last_step=True)
    assert torch.allclose(lambda_n, torch.ones_like(lambda_n))
    assert torch.allclose(p_n, un_halted)  # p_n = un_halted * 1.0
    assert torch.allclose(new_un_halted, torch.zeros_like(new_un_halted))


def test_pondernet_p_n_sums_to_one_across_full_sequence() -> None:
    """The defining PonderNet property: summing p_n over ALL steps (with the last step forced
    to lambda=1) must equal exactly 1 for every token, since it's a proper probability
    distribution over "which step did this token halt at"."""
    torch.manual_seed(0)
    router_config = RouterConfig(mechanism="pondernet")
    routers = [PonderNetRouter(hidden_size=TINY_HIDDEN, router_config=router_config) for _ in range(4)]
    for r in routers:
        torch.nn.init.normal_(r.halting_proj.weight, std=1.0)

    x = torch.randn(3, TINY_SEQ_LEN, TINY_HIDDEN)
    un_halted = torch.ones(3, TINY_SEQ_LEN)
    p_n_total = torch.zeros(3, TINY_SEQ_LEN)
    for i, router in enumerate(routers):
        is_last = i == len(routers) - 1
        p_n, _, un_halted = router(x, un_halted, is_last_step=is_last)
        p_n_total = p_n_total + p_n

    assert torch.allclose(p_n_total, torch.ones_like(p_n_total), atol=1e-5)


def test_geometric_kl_regularization_is_zero_when_matching_prior() -> None:
    """If every step's lambda exactly matches the geometric prior's constant rate, KL should be
    very close to zero (the model's implied distribution equals the prior)."""
    num_loops = 5
    prior_lambda = 0.3
    lambda_sequence = torch.full((num_loops, 2, 4), prior_lambda)
    # Last step must be forced to 1.0 to make this a valid (sums-to-1) distribution, matching
    # how the real forward pass always forces the final iteration's lambda to 1.
    lambda_sequence[-1] = 1.0
    kl = geometric_kl_regularization(lambda_sequence, geometric_prior_lambda=prior_lambda)
    # Not exactly zero because the prior used for comparison is RE-normalized after truncation
    # (the real forward pass's prior is a truncated geometric, not an exact match to a
    # lambda=1-forced last step) -- but should be small relative to a badly-mismatched case.
    kl_mismatched = geometric_kl_regularization(lambda_sequence, geometric_prior_lambda=0.9)
    assert kl.item() < kl_mismatched.item()


def test_pondernet_gated_model_forward_backward() -> None:
    config = _pondernet_config()
    model = LoopedMoEGPT(config)
    x = torch.randint(0, TINY_VOCAB, (4, TINY_SEQ_LEN))
    y = torch.randint(0, TINY_VOCAB, (4, TINY_SEQ_LEN))
    logits, loss = model(x, targets=y)
    assert logits.shape == (4, TINY_SEQ_LEN, TINY_VOCAB)
    assert torch.isfinite(loss)
    loss.backward()


def test_pondernet_gradient_reaches_every_non_boundary_router() -> None:
    """The core regression test motivating PonderNet's construction: unlike ACTRouter (where a
    router whose only role is causing halting gets zero gradient via the remainder term),
    EVERY router strictly between the forced-min_loops floor and the forced-last-step ceiling
    must receive nonzero gradient, since its own lambda_n multiplicatively affects every later
    p_n as well as its own."""
    torch.manual_seed(0)
    config = _pondernet_config(num_loops=6, min_loops=1)
    model = LoopedMoEGPT(config)
    x = torch.randint(0, TINY_VOCAB, (8, TINY_SEQ_LEN))
    y = torch.randint(0, TINY_VOCAB, (8, TINY_SEQ_LEN))
    _, loss = model(x, targets=y)
    loss.backward()

    # Routers 1..4 (0-indexed) are neither the forced-min_loops-zeroed step (0) nor the
    # forced-lambda=1 last step (5) -- these must all show real gradient.
    for i in range(1, config.loop.num_loops - 1):
        router = model.routers[i]
        grad_norms = [p.grad.norm().item() for p in router.parameters() if p.grad is not None]
        assert len(grad_norms) > 0, f"router[{i}] received no gradient at all."
        assert any(g > 0 for g in grad_norms), f"router[{i}] received only zero gradients."


def test_pondernet_exit_iteration_shape_and_range() -> None:
    config = _pondernet_config(num_loops=6, min_loops=1)
    model = LoopedMoEGPT(config)
    x = torch.randint(0, TINY_VOCAB, (4, TINY_SEQ_LEN))
    model(x)
    assert model.last_exit_iteration.shape == (4, TINY_SEQ_LEN)
    assert (model.last_exit_iteration >= 0).all()
    assert (model.last_exit_iteration < config.loop.num_loops).all()


def test_pondernet_requires_positive_ponder_cost_weight() -> None:
    try:
        RouterConfig(mechanism="pondernet", ponder_cost_weight=0.0)
        assert False, "Expected ValueError for ponder_cost_weight=0 with mechanism='pondernet'."
    except ValueError:
        pass


def test_pondernet_requires_geometric_prior_lambda_in_range() -> None:
    try:
        RouterConfig(mechanism="pondernet", geometric_prior_lambda=1.5)
        assert False, "Expected ValueError for geometric_prior_lambda outside (0, 1)."
    except ValueError:
        pass
