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
from looped_moe_gpt2.model.config import (
    AttentionConfig,
    MambaConfig,
    MixerType,
    MoEConfig,
    PositionEncodingType,
)
from looped_moe_gpt2.model.mamba_mixer import Mamba2Mixer
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
    """Pre-norm transformer block with attention and/or Mamba-2 mixing, and MoE/dense FFN.

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
        mixer_type: Which sequence-mixing mechanism(s) this block uses; see :class:`MixerType`.
            Defaults to attention-only, preserving all pre-existing behavior.
        mamba_config: Required (and only meaningful) when ``mixer_type`` is
            ``MixerType.MAMBA2`` or ``MixerType.HYBRID_MAMBA_ATTENTION``; see
            :class:`~looped_moe_gpt2.model.config.MambaConfig`.
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
        mixer_type: MixerType = MixerType.ATTENTION,
        mamba_config: Optional[MambaConfig] = None,
    ) -> None:
        super().__init__()
        self.use_iter_adaln = use_iter_adaln
        self.mixer_type = mixer_type
        self.use_mamba = mixer_type in (MixerType.MAMBA2, MixerType.HYBRID_MAMBA_ATTENTION)
        self.use_attention = mixer_type in (MixerType.ATTENTION, MixerType.HYBRID_MAMBA_ATTENTION)

        def make_norm() -> Union[IterAdaLN, nn.LayerNorm]:
            if use_iter_adaln:
                if max_iterations is None:
                    raise ValueError("max_iterations must be provided when use_iter_adaln=True.")
                return IterAdaLN(hidden_size, max_iterations)
            return nn.LayerNorm(hidden_size)

        self.norm1: Union[IterAdaLN, nn.LayerNorm] = make_norm()
        self.norm2: Union[IterAdaLN, nn.LayerNorm] = make_norm()
        # A hybrid block runs Mamba-2 first, then attention, each with its own pre-norm and
        # residual add -- norm_mamba is only constructed (and only used) in the hybrid case.
        self.norm_mamba: Optional[Union[IterAdaLN, nn.LayerNorm]] = (
            make_norm() if mixer_type == MixerType.HYBRID_MAMBA_ATTENTION else None
        )

        if self.use_mamba:
            if mamba_config is None:
                raise ValueError(f"mamba_config must be provided when mixer_type={mixer_type.value!r}.")
            self.mamba: Optional[Mamba2Mixer] = Mamba2Mixer(hidden_size, mamba_config, dropout)
        else:
            self.mamba = None

        self.attention: Optional[nn.Module] = (
            build_attention_module(
                hidden_size=hidden_size,
                num_heads=num_heads,
                attention_config=attention_config,
                position_encoding=position_encoding,
                max_seq_len=max_seq_len,
                dropout=dropout,
            )
            if self.use_attention
            else None
        )

        self.moe_enabled = moe_config.enabled
        self.feed_forward: Union[SparseMoE, DenseFeedForward] = (
            SparseMoE(hidden_size, moe_config)
            if moe_config.enabled
            else DenseFeedForward(hidden_size, 4 * hidden_size, dropout)
        )

    def forward(self, x: torch.Tensor, iteration_idx: Optional[int] = None) -> torch.Tensor:
        """Apply one transformer block pass.

        For ``MixerType.HYBRID_MAMBA_ATTENTION``, the Mamba-2 mixer and attention are applied
        as two separate pre-norm sub-blocks, in that order (Mamba-2 first), each with its own
        residual connection, before the feed-forward sub-block -- i.e. the block becomes
        (norm, mamba, residual) -> (norm, attention, residual) -> (norm, ffn, residual) rather
        than the standard two-sub-block (attention, ffn) structure.

        Args:
            x: Input tensor of shape ``(batch, seq_len, hidden_size)``.
            iteration_idx: Current loop iteration (0-indexed). Required if this block was
                constructed with ``use_iter_adaln=True``; ignored otherwise.

        Returns:
            Output tensor of the same shape as ``x``.

        Raises:
            ValueError: If ``use_iter_adaln=True`` but ``iteration_idx`` is not provided.
        """
        if self.use_iter_adaln and iteration_idx is None:
            raise ValueError("iteration_idx is required when use_iter_adaln=True.")

        def norm1(t: torch.Tensor) -> torch.Tensor:
            return self.norm1(t, iteration_idx) if self.use_iter_adaln else self.norm1(t)

        def norm2(t: torch.Tensor) -> torch.Tensor:
            return self.norm2(t, iteration_idx) if self.use_iter_adaln else self.norm2(t)

        if self.mixer_type == MixerType.HYBRID_MAMBA_ATTENTION:
            assert self.mamba is not None and self.attention is not None and self.norm_mamba is not None
            norm_mamba = (
                (lambda t: self.norm_mamba(t, iteration_idx))
                if self.use_iter_adaln
                else (lambda t: self.norm_mamba(t))
            )
            x = x + self.mamba(norm_mamba(x))
            x = x + self.attention(norm1(x))
        elif self.mixer_type == MixerType.MAMBA2:
            assert self.mamba is not None
            x = x + self.mamba(norm1(x))
        else:
            assert self.attention is not None
            x = x + self.attention(norm1(x))

        x = x + self.feed_forward(norm2(x))
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
