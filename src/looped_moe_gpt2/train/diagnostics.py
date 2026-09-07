"""Ablation diagnostics: representational-collapse and MoE active-ratio tracking.

Implements the two measurements ``docs/literature_survey.md`` §4 calls for: (1) cosine
similarity of hidden states across consecutive loop iterations, to directly detect the
fixed-point collapse that motivates IterAdaLN; (2) the attention-to-FFN active-parameter ratio
drift that motivates LoopMoE's balancing correction. Both are read-only instrumentation attached
via forward hooks so they add no overhead to gradient computation and can be disabled entirely
by simply not registering them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn

from looped_moe_gpt2.model.gpt import LoopedMoEGPT


@dataclass
class LoopDiagnostics:
    """Accumulates per-iteration hidden-state snapshots for a single forward pass.

    Attributes:
        hidden_states_by_iteration: Maps loop iteration index to the LAST block output observed
            at that iteration during the most recent forward pass (overwritten each call).
    """

    hidden_states_by_iteration: dict[int, torch.Tensor] = field(default_factory=dict)

    def reset(self) -> None:
        """Clear accumulated snapshots before the next forward pass."""
        self.hidden_states_by_iteration.clear()

    def consecutive_iteration_cosine_similarities(self) -> list[float]:
        """Compute cosine similarity between each pair of consecutive recorded iterations.

        Returns:
            A list of length ``len(hidden_states_by_iteration) - 1``, where entry ``i`` is the
            mean cosine similarity between iteration ``i`` and iteration ``i + 1``'s hidden
            states. A value approaching 1.0 across iterations indicates the representational
            collapse (near-fixed-point) that IterAdaLN is meant to prevent; values should stay
            meaningfully below 1.0 across successive loops if IterAdaLN is working.

        Raises:
            ValueError: If fewer than two iterations were recorded.
        """
        iterations = sorted(self.hidden_states_by_iteration.keys())
        if len(iterations) < 2:
            raise ValueError("Need at least two recorded iterations to compute cosine similarity.")
        similarities = []
        for earlier, later in zip(iterations[:-1], iterations[1:]):
            h_earlier = self.hidden_states_by_iteration[earlier].flatten(end_dim=-2)
            h_later = self.hidden_states_by_iteration[later].flatten(end_dim=-2)
            cos_sim = torch.nn.functional.cosine_similarity(h_earlier, h_later, dim=-1)
            similarities.append(cos_sim.mean().item())
        return similarities


def attach_loop_diagnostics(model: LoopedMoEGPT) -> LoopDiagnostics:
    """Register forward hooks on looped blocks to record per-iteration hidden states.

    A shared block module (e.g. Middle-Cycle's loop body) is called once per loop iteration with
    its ``iteration_idx`` passed explicitly as a forward keyword argument (see
    :meth:`looped_moe_gpt2.model.block.TransformerBlock.forward`). This function reads that same
    argument directly out of the hook's ``inputs``/``kwargs``, so each call is attributed to the
    correct iteration by construction -- no external call-order bookkeeping is needed, and the
    hook is safe to leave attached across arbitrarily many unrelated forward passes (e.g. many
    training steps) between calls to :meth:`LoopDiagnostics.reset`.

    Args:
        model: The model to instrument. Only blocks associated with a non-None iteration index
            in ``model.loop_plan`` are hooked.

    Returns:
        A :class:`LoopDiagnostics` instance that will be populated on every subsequent forward
        pass until the caller discards the returned hook handles (this function does not return
        handles; call :meth:`LoopDiagnostics.reset` before each forward pass you want isolated
        measurements for, and read the dict immediately after that forward pass).
    """
    diagnostics = LoopDiagnostics()
    looped_block_indices = {idx for idx, iteration_idx in model.loop_plan if iteration_idx is not None}

    def _hook(module: nn.Module, args: tuple, kwargs: dict, output: torch.Tensor) -> None:
        iteration_idx = kwargs.get("iteration_idx")
        if iteration_idx is None and len(args) > 1:
            iteration_idx = args[1]
        if iteration_idx is None:
            raise RuntimeError(
                "Hooked block did not receive an iteration_idx argument; "
                "attach_loop_diagnostics should only be used on blocks with use_iter_adaln=True."
            )
        diagnostics.hidden_states_by_iteration[iteration_idx] = output.detach()

    for block_idx in looped_block_indices:
        model.blocks[block_idx].register_forward_hook(_hook, with_kwargs=True)

    return diagnostics


def compute_active_ratio(model: LoopedMoEGPT) -> float:
    """Compute the active attention-FLOPs-to-FFN-FLOPs ratio drift diagnostic from LoopMoE.

    Approximates the ratio using parameter counts actually touched in the last forward pass
    (all attention parameters, since attention is always fully active; only the routed-expert
    parameters actually selected via top-k, plus always-active shared experts, for MoE blocks).

    Args:
        model: The model to inspect; must have just completed a forward pass so
            ``expert_load_summary`` reflects live routing statistics.

    Returns:
        The ratio of active attention parameters to active feed-forward parameters. LoopMoE's
        diagnostic tracks whether this ratio drifts from a well-tuned non-looped baseline's
        ratio as the number of loop iterations increases.
    """
    total_attn_params = 0
    total_active_ffn_params = 0
    expert_loads = model.expert_load_summary()

    for block_idx, block in enumerate(model.blocks):
        total_attn_params += sum(p.numel() for p in block.attention.parameters())
        if block_idx in expert_loads:
            moe = block.feed_forward
            load = expert_loads[block_idx]
            active_experts = (load > 0).sum().item()
            total_routed_params = sum(p.numel() for p in moe.routed_experts.parameters())
            params_per_expert = total_routed_params / moe.num_routed_experts
            shared_params = sum(p.numel() for e in moe.shared_experts for p in e.parameters())
            total_active_ffn_params += active_experts * params_per_expert + shared_params
        else:
            total_active_ffn_params += sum(p.numel() for p in block.feed_forward.parameters())

    if total_active_ffn_params == 0:
        return float("inf")
    return total_attn_params / total_active_ffn_params


def exit_depth_histogram(model: LoopedMoEGPT) -> Optional[dict[int, float]]:
    """Summarize how many tokens exited the adaptive-depth loop at each iteration.

    This is the direct measurement of whether the router-gated loop
    (:class:`looped_moe_gpt2.model.router.AdaptiveDepthRouter`) is actually behaving
    adaptively -- i.e. whether depth correlates with anything meaningful (token difficulty,
    position, content) rather than every token exiting at the same iteration (which would mean
    the router has collapsed to a fixed-depth model in practice, just with extra unused
    parameters). Call this immediately after a forward pass on the input you want to inspect.

    Args:
        model: The model to inspect; must have ``router_gated=True`` and have just completed a
            forward pass so ``last_exit_iteration`` is populated.

    Returns:
        A dict mapping iteration index to the fraction of tokens that exited at that iteration,
        or None if the model is not router-gated (nothing to report).
    """
    if not model.router_gated or model.last_exit_iteration is None:
        return None
    exit_iters = model.last_exit_iteration
    total = exit_iters.numel()
    return {
        iteration: (exit_iters == iteration).sum().item() / total
        for iteration in range(model.config.loop.num_loops)
    }
