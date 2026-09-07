"""Per-token PonderNet-style adaptive halting (Banino et al. 2021, arXiv:2107.05407) -- the
project's second follow-up to the capacity-based ``AdaptiveDepthRouter`` (see ``router.py``),
built after finding that a first attempt at Graves (2016) ACT (``act_router.py``) reproduces a
well-documented, literature-acknowledged limitation of vanilla ACT: the "remainder" term used
to normalize per-step weights to sum to 1 has NO gradient dependency on the halting step's own
halting-probability output, so the router responsible for a token's actual halting decision
receives no direct training signal from that decision -- confirmed empirically in this project
(a later-iteration router that only ever fires on the "halting" branch across an entire batch
received exactly zero gradient).

**PonderNet's fix**: instead of a remainder correction, define the probability of halting at
EXACTLY step n as a proper (truncated) geometric-style chain:

    p_n = lambda_n * prod_{j=1}^{n-1} (1 - lambda_j)

where lambda_n is step n's own halting probability. Because p_n multiplies together every
step's own lambda (not just the halting step's), gradients flow to EVERY step's halting head
through every p_n that follows it -- not just the last one. The final output is a weighted sum
over ALL steps' outputs, weighted by p_n (computed for every step during training, not just up
to the sampled halting step), and the loss adds a KL-divergence regularizer pulling the p_n
distribution toward a geometric prior (parameterized by lambda_p) -- this is what supplies the
"pressure to halt early" that Graves' ponder cost provided, in a differentiable, less biased form.
"""

from __future__ import annotations

import torch
from torch import nn

from looped_moe_gpt2.model.config import RouterConfig


class PonderNetRouter(nn.Module):
    """Per-iteration PonderNet halting-probability head.

    One instance is used per loop iteration, matching :class:`~looped_moe_gpt2.model.router.
    AdaptiveDepthRouter` and :class:`~looped_moe_gpt2.model.act_router.ACTRouter`'s per-iteration
    design. Constructed by :class:`~looped_moe_gpt2.model.gpt.LoopedMoEGPT` when
    ``RouterConfig.mechanism="pondernet"``.

    Args:
        hidden_size: Model (residual stream) width.
        router_config: Must have ``mechanism="pondernet"``; reads ``min_loops`` and
            ``ponder_cost_weight`` (used here as the KL-regularization weight ``beta``, and
            ``geometric_prior_lambda`` for the prior's own halting rate).
    """

    def __init__(self, hidden_size: int, router_config: RouterConfig) -> None:
        super().__init__()
        if router_config.mechanism != "pondernet":
            raise ValueError(
                f"PonderNetRouter requires RouterConfig.mechanism='pondernet', got "
                f"'{router_config.mechanism}'."
            )
        self.halting_proj = nn.Linear(hidden_size, 1, bias=True)
        # Same rationale as ACTRouter: bias toward low initial halting probability so early
        # training doesn't collapse to halting-everything-immediately before anything is learned.
        nn.init.zeros_(self.halting_proj.weight)
        nn.init.constant_(self.halting_proj.bias, -2.0)

    def forward(
        self, x: torch.Tensor, un_halted_prob: torch.Tensor, is_last_step: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute this step's halting probability and this step's contribution to p_n.

        Args:
            x: Hidden states of shape ``(batch, seq_len, hidden_size)`` at the current iteration.
            un_halted_prob: Running product of ``(1 - lambda_j)`` over all PRIOR iterations for
                each token, shape ``(batch, seq_len)`` -- i.e. the probability of having survived
                (not halted) up to but not including this step. Starts at 1.0 before iteration 0.
            is_last_step: If True (the final loop iteration, ``num_loops - 1``), this step's
                lambda is forced to 1.0 regardless of the projection's output -- guarantees every
                token halts by the last iteration (``p_n`` values across all iterations still sum
                to exactly 1 for every token, matching PonderNet's own convention of forcing
                halting at the maximum step count).

        Returns:
            A tuple ``(p_n, lambda_n, new_un_halted_prob)``:

            - ``p_n``: float ``(batch, seq_len)``, this step's probability-of-halting-exactly-here,
              used both to weight this step's block output in the final accumulated output and
              as this step's contribution to the KL regularization term.
            - ``lambda_n``: this step's own halting probability (pre-multiplication by
              ``un_halted_prob``) -- returned so the caller can track the full lambda sequence if
              needed for diagnostics.
            - ``new_un_halted_prob``: updated running survival product, to pass into the next
              iteration's call.
        """
        if is_last_step:
            lambda_n = torch.ones_like(un_halted_prob)
        else:
            lambda_n = torch.sigmoid(self.halting_proj(x).squeeze(-1))

        p_n = un_halted_prob * lambda_n
        new_un_halted_prob = un_halted_prob * (1.0 - lambda_n)

        return p_n, lambda_n, new_un_halted_prob


def geometric_kl_regularization(
    lambda_sequence: torch.Tensor, geometric_prior_lambda: float
) -> torch.Tensor:
    """Compute the KL divergence between the model's halting distribution and a geometric prior.

    Args:
        lambda_sequence: Stacked per-step halting probabilities, shape
            ``(num_loops, batch, seq_len)`` -- ``lambda_sequence[n]`` is step n's ``lambda_n``
            for every token (the SAME quantity :meth:`PonderNetRouter.forward` returns as
            ``lambda_n``, stacked across all iterations by the caller).
        geometric_prior_lambda: The prior's constant per-step halting rate (``lambda_p`` in
            PonderNet's notation) -- e.g. 0.1 encodes a prior belief that ~10% of tokens should
            halt at each step, giving an expected ponder depth of ``1 / lambda_p`` steps.

    Returns:
        A scalar tensor: the mean (over batch and sequence position) KL divergence between the
        per-token halting distribution implied by ``lambda_sequence`` and the geometric prior,
        truncated at ``num_loops`` steps.
    """
    num_loops = lambda_sequence.shape[0]
    device = lambda_sequence.device
    dtype = lambda_sequence.dtype

    # Reconstruct p_n (probability of halting at exactly step n) from the lambda sequence, the
    # same way PonderNetRouter.forward does internally -- done here as a second pass so this
    # function can be called independently of the main forward loop's per-step un_halted_prob
    # bookkeeping, using only the stacked lambdas the caller already collected.
    un_halted_prob = torch.ones_like(lambda_sequence[0])
    p_n_list = []
    for n in range(num_loops):
        p_n = un_halted_prob * lambda_sequence[n]
        p_n_list.append(p_n)
        un_halted_prob = un_halted_prob * (1.0 - lambda_sequence[n])
    p_n_stacked = torch.stack(p_n_list, dim=0)  # (num_loops, batch, seq_len)

    # Geometric prior probability mass at each step 1..num_loops, per Pr[X=k] = (1-lambda_p)^k * lambda_p.
    steps = torch.arange(num_loops, device=device, dtype=dtype)
    prior_p_n = (1.0 - geometric_prior_lambda) ** steps * geometric_prior_lambda
    prior_p_n = prior_p_n / prior_p_n.sum()  # renormalize since it's truncated at num_loops
    prior_p_n = prior_p_n.view(num_loops, 1, 1)

    # KL(p_n || prior_p_n) = sum_n p_n * log(p_n / prior_p_n), computed with a small epsilon to
    # avoid log(0) for steps where a token's actual halting probability is numerically zero.
    eps = 1e-8
    kl = (p_n_stacked * (torch.log(p_n_stacked + eps) - torch.log(prior_p_n + eps))).sum(dim=0)
    return kl.mean()
