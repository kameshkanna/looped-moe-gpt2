"""Per-token Adaptive Computation Time (ACT) halting, following Graves (2016) / Universal
Transformer's ACT mechanism -- built as a follow-up to the ``AdaptiveDepthRouter`` (capacity-based)
mechanism in ``router.py``, after an empirical finding (see ``docs/adaptive_depth_router_design.md``)
that capacity-based routing cannot express genuine difficulty-driven depth scaling: its fixed
survivor-fraction cutoff bounds the exit-depth DISTRIBUTION independent of training, so the model
can only learn which tokens occupy a fixed-size pool at each depth, never how much total compute
a given token receives.

**The key structural difference from capacity-based routing**: there is no fixed fraction of
tokens that must exit at each iteration. Each token independently accumulates a per-iteration
halting probability and stops once ITS OWN cumulative probability crosses a threshold -- so, in
principle, a genuinely hard token could ride every iteration while an easy token halts after one,
with no artificial cap forcing some fixed proportion to exit each round. Whether this actually
produces difficulty-correlated depth in practice (rather than some other learned pattern) is an
open empirical question this class exists to let the project test -- it is not assumed here.

**Ponder cost is essential, not optional.** Without a loss term penalizing how many iterations a
token uses, there is no reason for a loss-minimizing model to ever halt early: additional
iterations through the shared block can only help or be neutral for prediction quality, never
hurt it, so a model with no ponder-cost pressure would learn to always use every iteration
(collapsing back to fixed-depth behavior, just with extra unused halting-probability parameters).
"""

from __future__ import annotations

import torch
from torch import nn

from looped_moe_gpt2.model.config import RouterConfig


class ACTRouter(nn.Module):
    """Per-iteration ACT halting-probability head.

    One instance is used per loop iteration (matching :class:`~looped_moe_gpt2.model.router.
    AdaptiveDepthRouter`'s per-iteration design), constructed by
    :class:`~looped_moe_gpt2.model.gpt.LoopedMoEGPT` when ``RouterConfig.mechanism="act"``.

    Args:
        hidden_size: Model (residual stream) width.
        router_config: Must have ``mechanism="act"``; only ``min_loops`` and
            ``ponder_cost_weight`` are read from it (``capacity_ratio``/``aux_loss_weight`` are
            unused by this mechanism).
    """

    def __init__(self, hidden_size: int, router_config: RouterConfig) -> None:
        super().__init__()
        if router_config.mechanism != "act":
            raise ValueError(f"ACTRouter requires RouterConfig.mechanism='act', got '{router_config.mechanism}'.")
        self.halting_proj = nn.Linear(hidden_size, 1, bias=True)
        # Bias the halting unit toward NOT halting at init (small initial halting probability),
        # so early training doesn't immediately collapse to halting everything at iteration 1
        # before the model has learned anything -- mirrors standard ACT initialization practice.
        nn.init.zeros_(self.halting_proj.weight)
        nn.init.constant_(self.halting_proj.bias, -2.0)

    def forward(
        self,
        x: torch.Tensor,
        cumulative_halting_prob: torch.Tensor,
        still_running: torch.Tensor,
        force_continue: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute this iteration's halting probability and update per-token running totals.

        Args:
            x: Hidden states of shape ``(batch, seq_len, hidden_size)`` at the current iteration.
            cumulative_halting_prob: Running sum of halting probabilities accumulated over all
                PRIOR iterations for each token, shape ``(batch, seq_len)``. Tokens that have
                already halted keep their cumulative probability frozen at whatever value it had
                when they halted (this method does not mutate it for already-halted tokens).
            still_running: Boolean ``(batch, seq_len)``; True where the token has not yet halted
                entering this iteration.
            force_continue: If True (used for iterations below ``RouterConfig.min_loops``), this
                iteration's halting probability is computed and returned for bookkeeping, but no
                token is allowed to halt yet regardless of its accumulated probability --
                enforces the minimum-loops-per-token floor.

        Returns:
            A tuple ``(update_weight, new_cumulative_prob, newly_halted, step_ponder_cost)``:

            - ``update_weight``: float ``(batch, seq_len)`` in [0, 1], how much this iteration's
              block output should contribute to the token's final representation -- for a
              continuing token this is its halting probability at this step; for a token halting
              AT this step, it is the "remainder" (``1 - cumulative_prob_before_this_step``, per
              Graves 2016) so all weights across a token's iterations sum to exactly 1; for an
              already-halted token (from a prior iteration) it is 0.
            - ``new_cumulative_prob``: updated running total, shape ``(batch, seq_len)``.
            - ``newly_halted``: boolean ``(batch, seq_len)``, True for tokens that cross the
              halting threshold AT this iteration (used by the caller to update ``still_running``
              and record the exit iteration for diagnostics).
            - ``step_ponder_cost``: float ``(batch, seq_len)``, this iteration's contribution to
              the ACT ponder cost -- ``update_weight`` itself for continuing tokens (so the total
              summed across iterations approaches the number of steps taken, per Graves 2016's
              "N(t) + R(t)" ponder cost), 0 for already-halted tokens. Built from
              ``update_weight`` (a differentiable quantity depending on THIS iteration's
              ``halting_proj`` output) rather than a boolean mask, so summing this across
              iterations and backpropagating actually reaches every router that was called,
              including the one that caused a token to halt -- unlike naively using the boolean
              "still running" mask, which carries no gradient at all.
        """
        halting_prob = torch.sigmoid(self.halting_proj(x).squeeze(-1))  # (batch, seq_len)

        if force_continue:
            update_weight = halting_prob * still_running.float()
            new_cumulative_prob = cumulative_halting_prob + update_weight
            newly_halted = torch.zeros_like(still_running)
            return update_weight, new_cumulative_prob, newly_halted, update_weight

        prospective_cumulative = cumulative_halting_prob + halting_prob
        # Halt if adding this step's probability would reach/exceed 1 -- the epsilon slack
        # (Graves 2016 uses a small epsilon so near-1.0 accumulation halts rather than needing
        # to hit exactly 1.0, which floating point may never do) is folded into using >= 1.0
        # directly here since sigmoid output is continuous and exact 1.0 is not reachable, making
        # a separate epsilon unnecessary in practice.
        would_halt = (prospective_cumulative >= 1.0) & still_running
        remainder = (1.0 - cumulative_halting_prob).clamp(min=0.0, max=1.0)

        update_weight = torch.where(would_halt, remainder, halting_prob) * still_running.float()
        new_cumulative_prob = torch.where(
            would_halt, cumulative_halting_prob + remainder, prospective_cumulative
        )
        # Tokens that already halted in a prior iteration keep their frozen cumulative value.
        new_cumulative_prob = torch.where(still_running, new_cumulative_prob, cumulative_halting_prob)

        newly_halted = would_halt

        return update_weight, new_cumulative_prob, newly_halted, update_weight
