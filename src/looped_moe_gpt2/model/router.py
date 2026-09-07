"""Per-token adaptive loop-depth router (Mixture-of-Recursions style, masked-dense variant).

Implements the design in ``docs/adaptive_depth_router_design.md``: at each loop iteration, a
per-iteration router scores every still-active token, and only the top ``capacity_ratio``
fraction remain active for the next iteration -- the rest "exit," carrying their current hidden
state forward unchanged through the remaining iterations. This follows Mixture-of-Recursions'
expert-choice routing (arXiv:2507.10524), adapted to this repo's Middle-Cycle sharing pattern.

**Masked-dense, not gather-scatter.** Every token is computed by the shared block at every
iteration regardless of exit status (wasting some FLOPs on already-exited tokens), and only the
WRITE-BACK of each iteration's update is masked by which tokens are still active. This keeps
every tensor's shape constant across iterations and across the whole batch, which avoids the
dynamic-shape recompilation problems already hit once in this repo (see
:class:`looped_moe_gpt2.model.moe.SparseMoE`'s vectorization fix). A real inference-time FLOPs
saving requires the gather/scatter-dense variant described as a later step in the design doc;
this module intentionally does not implement that yet.

**Training-time gradient path.** A hard top-k "which tokens exit" decision has no gradient, so
the router is trained with a soft, differentiable relaxation: each token's contribution to the
loop's output is scaled by a continuous keep-probability (a sigmoid of the router's score
relative to the current iteration's threshold) rather than a hard 0/1 mask. At inference, the
hard top-k mask is used directly. An auxiliary loss encourages the soft and hard decisions to
agree, so the soft path at the end of training approximates the hard path closely.
"""

from __future__ import annotations

import torch
from torch import nn

from looped_moe_gpt2.model.config import RouterConfig


class AdaptiveDepthRouter(nn.Module):
    """Per-iteration router deciding which tokens continue looping vs. exit early.

    One instance is used per loop iteration (MoR found per-step routers outperform a single
    router shared across all iterations), so :class:`~looped_moe_gpt2.model.gpt.LoopedMoEGPT`
    constructs ``num_loops`` instances of this module for a router-gated loop body.

    Args:
        hidden_size: Model (residual stream) width.
        router_config: Capacity ratio, minimum loop count, and auxiliary loss weight.
    """

    def __init__(self, hidden_size: int, router_config: RouterConfig) -> None:
        super().__init__()
        self.capacity_ratio = router_config.capacity_ratio
        self.aux_loss_weight = router_config.aux_loss_weight
        self.score_proj = nn.Linear(hidden_size, 1, bias=True)

    def forward(
        self, x: torch.Tensor, active_mask: torch.Tensor, force_keep_all: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score currently-active tokens and decide which remain active for the next iteration.

        Args:
            x: Hidden states of shape ``(batch, seq_len, hidden_size)`` at the current iteration.
            active_mask: Boolean tensor of shape ``(batch, seq_len)``; True where the token is
                still active entering this iteration (has not yet exited in an earlier one).
            force_keep_all: If True (used for iterations below ``RouterConfig.min_loops``),
                every currently-active token is kept regardless of its router score -- enforces
                the minimum-loops-per-token floor.

        Returns:
            A tuple ``(next_active_mask, soft_keep_prob, aux_loss)``:

            - ``next_active_mask``: boolean ``(batch, seq_len)``, the hard top-``capacity_ratio``
              decision -- which tokens remain active for the NEXT iteration. Tokens already
              inactive entering this call stay inactive (an exited token never re-enters).
            - ``soft_keep_prob``: float ``(batch, seq_len)`` in [0, 1], the differentiable
              keep-probability used to scale this iteration's contribution during training. Is
              exactly ``next_active_mask.float()`` when ``force_keep_all`` is True.
            - ``aux_loss``: scalar tensor, the soft/hard agreement penalty (already multiplied
              by ``aux_loss_weight``) to add to the training loss.
        """
        scores = self.score_proj(x).squeeze(-1)  # (batch, seq_len)
        # Inactive tokens must never be selected; push their score to -inf for the top-k ranking.
        masked_scores = scores.masked_fill(~active_mask, float("-inf"))

        if force_keep_all:
            next_active_mask = active_mask.clone()
            soft_keep_prob = active_mask.float()
            return next_active_mask, soft_keep_prob, torch.zeros((), device=x.device, dtype=x.dtype)

        num_active_per_batch = active_mask.sum(dim=-1)  # (batch,)
        keep_count_per_batch = (num_active_per_batch.float() * self.capacity_ratio).ceil().long()
        keep_count_per_batch = keep_count_per_batch.clamp(min=0)

        # Per-sequence keep-count varies (active-token count shrinks as earlier tokens exit), so
        # a single torch.topk(k=...) can't be used directly -- PyTorch requires one static k for
        # the whole batch. Rather than loop over the batch calling topk with each row's own k
        # (which requires a GPU->CPU .item() sync per batch element per iteration -- the same
        # anti-pattern fixed in SparseMoE's vectorization, measured here to cost ~7% overhead at
        # batch_size=8 and worse at larger batch sizes), rank ALL tokens by score via a single
        # batched argsort and keep the ones whose rank is below that row's own keep_count. This
        # is one vectorized op regardless of batch size, with no host-device sync in the loop.
        sorted_indices = masked_scores.argsort(dim=-1, descending=True)  # (batch, seq_len)
        rank = torch.empty_like(sorted_indices)
        rank.scatter_(
            dim=1, index=sorted_indices, src=torch.arange(x.shape[1], device=x.device).expand(x.shape[0], -1)
        )
        next_active_mask = rank < keep_count_per_batch.unsqueeze(-1)
        next_active_mask = next_active_mask & active_mask  # never select an already-inactive token

        # Soft, differentiable relaxation for the training-time gradient path: keep-probability
        # is a sigmoid of each token's score relative to its batch's threshold score (the k-th
        # highest active score), so tokens near the cutoff get a smooth, learnable gradient
        # signal instead of a hard step function.
        threshold = torch.where(
            keep_count_per_batch > 0,
            torch.gather(
                masked_scores.sort(dim=-1, descending=True).values,
                1,
                (keep_count_per_batch - 1).clamp(min=0).unsqueeze(-1),
            ).squeeze(-1),
            torch.full_like(keep_count_per_batch, float("inf"), dtype=masked_scores.dtype),
        )
        soft_keep_prob = torch.sigmoid(scores - threshold.unsqueeze(-1))
        soft_keep_prob = soft_keep_prob * active_mask.float()  # inactive tokens stay at 0

        aux_loss = self.aux_loss_weight * torch.mean(
            (soft_keep_prob - next_active_mask.float()) ** 2
        )

        return next_active_mask, soft_keep_prob, aux_loss
