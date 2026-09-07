"""Sparse Mixture-of-Experts feed-forward layer with auxiliary-loss-free load balancing.

Implements the Kimi K2 / DeepSeek-V3 mechanism (arXiv:2507.20534, arXiv:2412.19437): a bank of
fine-grained routed experts selected via top-k on a bias-adjusted routing score, plus one
always-on shared expert, with per-expert bias terms updated by a simple non-gradient rule
(overloaded experts' bias decreases, underloaded experts' bias increases) instead of an
auxiliary load-balancing loss term. See ``docs/literature_survey.md`` §3.2 for the exact
formulas this reimplements and the small-scale sizing rationale.

Routing is implemented as a DENSE batched computation (every token passes through every routed
expert, masked and summed by routing weight afterward) rather than a Python loop that gathers a
per-expert token subset. At this model's scale (small hidden size, few experts), the wasted FLOPs
from dense computation are cheap on a GPU, while the Python-loop version was measured to bottleneck
on CPU kernel-launch/dispatch overhead rather than actual compute -- each loop iteration's
boolean-mask gather (`x_flat[token_mask]`) is a small, shape-dependent op that cannot be batched
or overlapped by the CUDA scheduler. The dense formulation below issues one batched matmul per
weight matrix regardless of ``num_routed_experts``, which is both simpler and faster here.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from looped_moe_gpt2.model.config import MoEConfig

logger = logging.getLogger(__name__)


class Expert(nn.Module):
    """A single feed-forward expert: standard GPT-2-style GELU MLP.

    Args:
        hidden_size: Model (residual stream) width, i.e. this expert's input/output dimension.
        intermediate_size: This expert's hidden width (not the aggregate MoE layer width).
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.fc_in = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.activation = nn.GELU()
        self.fc_out = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the expert FFN.

        Args:
            x: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of the same shape as ``x``.
        """
        return self.fc_out(self.activation(self.fc_in(x)))


class BatchedExperts(nn.Module):
    """A bank of identically-shaped expert FFNs evaluated densely as one batched matmul.

    Functionally equivalent to ``nn.ModuleList([Expert(...) for _ in range(num_experts)])``
    called one-by-one, but stores every expert's weights stacked along a leading "expert"
    dimension so the whole bank can be applied to all tokens in a single pair of batched matrix
    multiplies (via ``torch.einsum``) instead of ``num_experts`` sequential Python-level calls.
    See this module's file docstring for why this matters at small model scale.

    Args:
        num_experts: Number of experts in this bank.
        hidden_size: Model (residual stream) width.
        intermediate_size: Hidden width of each individual expert's FFN.
    """

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        # Match nn.Linear's default init (Kaiming-uniform-derived) by constructing per-expert
        # nn.Linear modules once, then copying their initialized weights into the stacked
        # parameter tensors -- keeps initialization statistics identical to the un-batched form.
        fc_in_refs = [nn.Linear(hidden_size, intermediate_size) for _ in range(num_experts)]
        fc_out_refs = [nn.Linear(intermediate_size, hidden_size) for _ in range(num_experts)]

        self.fc_in_weight = nn.Parameter(torch.stack([m.weight.T for m in fc_in_refs]))  # (E, H, I)
        self.fc_in_bias = nn.Parameter(torch.stack([m.bias for m in fc_in_refs]))  # (E, I)
        self.fc_out_weight = nn.Parameter(torch.stack([m.weight.T for m in fc_out_refs]))  # (E, I, H)
        self.fc_out_bias = nn.Parameter(torch.stack([m.bias for m in fc_out_refs]))  # (E, H)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply every expert in the bank to every token.

        Args:
            x: Input tensor of shape ``(N, hidden_size)``, shared across all experts.

        Returns:
            Output tensor of shape ``(num_experts, N, hidden_size)``: entry ``[e]`` is expert
            ``e``'s output for every token in ``x``.
        """
        hidden = torch.einsum("nh,ehi->eni", x, self.fc_in_weight) + self.fc_in_bias.unsqueeze(1)
        hidden = self.activation(hidden)
        out = torch.einsum("eni,eih->enh", hidden, self.fc_out_weight) + self.fc_out_bias.unsqueeze(1)
        return out


class SparseMoE(nn.Module):
    """Sparse MoE feed-forward layer with a shared expert and auxiliary-loss-free routing.

    The per-expert routing bias (``self.routing_bias``) is a non-trainable buffer updated
    in-place by :meth:`update_routing_bias`, which the training loop must call after each
    optimizer step (it deliberately sits outside the autograd graph, per DeepSeek-V3's design,
    to avoid the gradient interference that an auxiliary balancing loss introduces).

    Args:
        hidden_size: Model (residual stream) width.
        moe_config: Expert count, top-k, and bias update hyperparameters.
    """

    def __init__(self, hidden_size: int, moe_config: MoEConfig) -> None:
        super().__init__()
        self.num_routed_experts = moe_config.num_routed_experts
        self.top_k = moe_config.top_k
        self.bias_update_speed = moe_config.bias_update_speed

        self.routed_experts = BatchedExperts(
            moe_config.num_routed_experts, hidden_size, moe_config.expert_intermediate_size
        )
        self.shared_experts = nn.ModuleList(
            [
                Expert(hidden_size, moe_config.expert_intermediate_size)
                for _ in range(moe_config.num_shared_experts)
            ]
        )

        self.expert_centroids = nn.Linear(hidden_size, moe_config.num_routed_experts, bias=False)
        self.register_buffer(
            "routing_bias", torch.zeros(moe_config.num_routed_experts), persistent=True
        )
        self.register_buffer(
            "last_expert_load", torch.zeros(moe_config.num_routed_experts), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route tokens to experts and combine outputs.

        Args:
            x: Input tensor of shape ``(batch, seq_len, hidden_size)``.

        Returns:
            Output tensor of the same shape as ``x``.
        """
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)  # (N, hidden_size), N = batch_size * seq_len

        scores = self.expert_centroids(x_flat)  # (N, num_routed_experts), raw s_i
        biased_scores = scores + self.routing_bias  # bias affects selection only, not weighting
        _, top_k_indices = torch.topk(biased_scores, self.top_k, dim=-1)  # (N, top_k)
        # Combination weights use the *unbiased* scores at the selected indices, softmax-normalized
        # over just the selected top-k (standard DeepSeek-V3 / Kimi K2 practice).
        top_k_scores = torch.gather(scores, dim=-1, index=top_k_indices)
        top_k_weights = torch.softmax(top_k_scores, dim=-1)  # (N, top_k)

        # Dense combination weight per (token, expert): 0 unless that expert was in the token's
        # top-k, in which case it's that slot's softmax weight. Built via scatter rather than a
        # per-expert Python loop, so this is a single batched op regardless of expert count.
        dense_weights = torch.zeros(x_flat.shape[0], self.num_routed_experts, device=x.device, dtype=top_k_weights.dtype)
        dense_weights.scatter_(dim=1, index=top_k_indices, src=top_k_weights)  # (N, num_routed_experts)

        expert_outputs = self.routed_experts(x_flat)  # (num_routed_experts, N, hidden_size)
        # Weighted sum over the expert dimension: for each token, only its top-k experts have
        # nonzero weight, so this reproduces sparse routing's output exactly while computing (and
        # discarding) every expert's output densely -- see module docstring for why that trade is
        # worthwhile at this model's scale.
        output = torch.einsum("en,enh->nh", dense_weights.T, expert_outputs)

        for shared_expert in self.shared_experts:
            output = output + shared_expert(x_flat)

        with torch.no_grad():
            self.last_expert_load = (dense_weights > 0).sum(dim=0).float() / max(x_flat.shape[0], 1)

        return output.view(batch_size, seq_len, hidden_size)

    @torch.no_grad()
    def update_routing_bias(self) -> None:
        """Apply the DeepSeek-V3 auxiliary-loss-free bias update using the last forward's load.

        Overloaded experts (load above the mean) have their bias decreased by
        ``bias_update_speed``; underloaded experts (load below the mean) have their bias
        increased by the same amount. Must be called once per training step, after the forward
        pass that populated ``self.last_expert_load``, and is a no-op if ``bias_update_speed``
        has been annealed to zero.
        """
        if self.bias_update_speed == 0.0:
            return
        mean_load = self.last_expert_load.mean()
        overloaded = self.last_expert_load > mean_load
        underloaded = self.last_expert_load < mean_load
        self.routing_bias[overloaded] -= self.bias_update_speed
        self.routing_bias[underloaded] += self.bias_update_speed

    def set_bias_update_speed(self, speed: float) -> None:
        """Update the bias update speed (used by the training loop to anneal it to zero).

        Args:
            speed: New value for ``bias_update_speed``, typically annealed linearly to 0 over
                the final ``MoEConfig.bias_update_speed_decay_frac`` fraction of training.
        """
        self.bias_update_speed = speed
