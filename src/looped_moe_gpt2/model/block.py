"""A single transformer block: attention + feed-forward, with pre-norm and optional IterAdaLN.

Composes :mod:`looped_moe_gpt2.model.attention` (MLA or standard MHA) and either
:mod:`looped_moe_gpt2.model.moe` (sparse MoE FFN) or a standard dense GPT-2-style MLP,
following the pre-norm residual pattern. When used inside a loop, normalization is delegated to
:class:`looped_moe_gpt2.model.normalization.IterAdaLN` and conditioned on the loop iteration
index; otherwise standard ``nn.LayerNorm`` is used.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import nn

from looped_moe_gpt2.model.attention import build_attention_module
from looped_moe_gpt2.model.config import AttentionConfig, MoEConfig, PositionEncodingType
from looped_moe_gpt2.model.moe import SparseMoE
from looped_moe_gpt2.model.normalization import IterAdaLN


class DenseFeedForward(nn.Module):
    """Standard GPT-2-style dense feed-forward (GELU MLP), used when MoE is disabled.

    Args:
        hidden_size: Model (residual stream) width.
        intermediate_size: MLP hidden width (GPT-2 convention: 4x hidden_size).
        dropout: Dropout probability applied to the output.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc_in = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.activation = nn.GELU()
        self.fc_out = nn.Linear(intermediate_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the dense FFN.

        Args:
            x: Input tensor of shape ``(..., hidden_size)``.

        Returns:
            Output tensor of the same shape as ``x``.
        """
        return self.dropout(self.fc_out(self.activation(self.fc_in(x))))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with MLA/MHA attention and MoE/dense feed-forward.

    Args:
        hidden_size: Model (residual stream) width.
        num_heads: Number of attention heads.
        attention_config: Attention hyperparameters.
        position_encoding: Positional encoding scheme.
        moe_config: MoE hyperparameters; if ``moe_config.enabled`` is False, a dense FFN of
            width ``4 * hidden_size`` is used instead.
        max_seq_len: Maximum sequence length.
        dropout: Dropout probability used throughout the block.
        use_iter_adaln: If True, use :class:`IterAdaLN` (conditioned on a loop iteration index
            passed to :meth:`forward`) instead of standard LayerNorm.
        max_iterations: Required if ``use_iter_adaln`` is True; sizes IterAdaLN's embedding
            table.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attention_config: AttentionConfig,
        position_encoding: PositionEncodingType,
        moe_config: MoEConfig,
        max_seq_len: int,
        dropout: float = 0.1,
        use_iter_adaln: bool = False,
        max_iterations: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.use_iter_adaln = use_iter_adaln

        if use_iter_adaln:
            if max_iterations is None:
                raise ValueError("max_iterations must be provided when use_iter_adaln=True.")
            self.norm1: Union[IterAdaLN, nn.LayerNorm] = IterAdaLN(hidden_size, max_iterations)
            self.norm2: Union[IterAdaLN, nn.LayerNorm] = IterAdaLN(hidden_size, max_iterations)
        else:
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)

        self.attention = build_attention_module(
            hidden_size=hidden_size,
            num_heads=num_heads,
            attention_config=attention_config,
            position_encoding=position_encoding,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        self.moe_enabled = moe_config.enabled
        self.feed_forward: Union[SparseMoE, DenseFeedForward] = (
            SparseMoE(hidden_size, moe_config)
            if moe_config.enabled
            else DenseFeedForward(hidden_size, 4 * hidden_size, dropout)
        )

    def forward(self, x: torch.Tensor, iteration_idx: Optional[int] = None) -> torch.Tensor:
        """Apply one transformer block pass.

        Args:
            x: Input tensor of shape ``(batch, seq_len, hidden_size)``.
            iteration_idx: Current loop iteration (0-indexed). Required if this block was
                constructed with ``use_iter_adaln=True``; ignored otherwise.

        Returns:
            Output tensor of the same shape as ``x``.

        Raises:
            ValueError: If ``use_iter_adaln=True`` but ``iteration_idx`` is not provided.
        """
        if self.use_iter_adaln:
            if iteration_idx is None:
                raise ValueError("iteration_idx is required when use_iter_adaln=True.")
            x = x + self.attention(self.norm1(x, iteration_idx))
            x = x + self.feed_forward(self.norm2(x, iteration_idx))
        else:
            x = x + self.attention(self.norm1(x))
            x = x + self.feed_forward(self.norm2(x))
        return x

    def expert_load(self) -> Optional[torch.Tensor]:
        """Return the most recent forward pass's per-expert load, if this block uses MoE.

        Returns:
            A tensor of shape ``(num_routed_experts,)`` with the fraction of tokens routed to
            each expert in the last forward call, or None if this block uses a dense FFN.
        """
        if isinstance(self.feed_forward, SparseMoE):
            return self.feed_forward.last_expert_load
        return None

    def update_routing_bias(self) -> None:
        """Trigger the auxiliary-loss-free bias update on this block's MoE layer, if present.

        No-op if this block uses a dense FFN instead of MoE.
        """
        if isinstance(self.feed_forward, SparseMoE):
            self.feed_forward.update_routing_bias()
