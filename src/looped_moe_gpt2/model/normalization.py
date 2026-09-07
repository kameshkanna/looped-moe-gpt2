"""Iteration-conditioned adaptive layer normalization (IterAdaLN).

Follows LoopMoE (arXiv:2606.04438): under weight sharing across loop iterations, a plain
pre-norm (LayerNorm/RMSNorm) gives the model no way to distinguish iteration ``t`` from
iteration ``t-1``, since every iteration applies literally the same normalization affine
parameters to structurally similar inputs. This drives fast convergence to a near-fixed-point
(see ``docs/literature_survey.md`` §2.1) that wastes additional loop iterations. IterAdaLN fixes
this by generating the LayerNorm scale/shift from the loop iteration index (via a learned
per-iteration embedding), so each pass through the shared block is a distinguishable affine
transform even though the linear/attention weights themselves are identical.
"""

from __future__ import annotations

import torch
from torch import nn


class IterAdaLN(nn.Module):
    """LayerNorm with scale/shift modulated by the current loop iteration index.

    Args:
        hidden_size: Model (residual stream) width.
        max_iterations: Maximum number of loop iterations this module will ever be called with;
            sizes the per-iteration embedding table.
        eps: LayerNorm epsilon.
    """

    def __init__(self, hidden_size: int, max_iterations: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=False)
        self.iter_embedding = nn.Embedding(max_iterations, hidden_size * 2)
        # Zero-init so IterAdaLN starts as a no-op (identity scale=1, shift=0) at iteration 0,
        # matching the standard AdaLN-zero initialization used in conditional-generation models.
        nn.init.zeros_(self.iter_embedding.weight)

    def forward(self, x: torch.Tensor, iteration_idx: int) -> torch.Tensor:
        """Apply iteration-conditioned normalization.

        Args:
            x: Input tensor of shape ``(batch, seq_len, hidden_size)``.
            iteration_idx: Current loop iteration (0-indexed), shared across the whole batch.

        Returns:
            Normalized tensor of the same shape as ``x``.

        Raises:
            ValueError: If ``iteration_idx`` is out of range for the configured embedding table.
        """
        if not 0 <= iteration_idx < self.iter_embedding.num_embeddings:
            raise ValueError(
                f"iteration_idx ({iteration_idx}) out of range for max_iterations "
                f"({self.iter_embedding.num_embeddings})."
            )
        device = x.device
        idx_tensor = torch.tensor(iteration_idx, device=device, dtype=torch.long)
        scale_shift = self.iter_embedding(idx_tensor)  # (hidden_size * 2,)
        scale, shift = scale_shift.chunk(2, dim=-1)
        normalized = self.norm(x)
        return normalized * (1.0 + scale) + shift
