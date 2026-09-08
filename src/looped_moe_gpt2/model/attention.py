"""Multi-head Latent Attention (MLA) and standard multi-head attention.

MLA follows DeepSeek-V3 (arXiv:2412.19437) as reused, unmodified, by Kimi K2
(arXiv:2507.20534): queries and keys/values are each computed via a low-rank down-projection to
a shared latent, then up-projected per-head, with a small decoupled slice of each head carrying
rotary position information separately from the position-agnostic ("NoPE") latent-derived slice.
Dimensions default to GPT-2-small-scale equivalents of DeepSeek-V3's compression ratios; see
:class:`looped_moe_gpt2.model.config.AttentionConfig` and ``docs/literature_survey.md`` §3.1.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from looped_moe_gpt2.model.config import AttentionConfig, PositionEncodingType
from looped_moe_gpt2.model.positional import apply_rope, build_rope_cache


class MultiHeadLatentAttention(nn.Module):
    """DeepSeek-V3 / Kimi-K2-style Multi-head Latent Attention with decoupled RoPE.

    Args:
        hidden_size: Model (residual stream) width.
        num_heads: Number of attention heads.
        attention_config: Compression dimensions and RoPE-slice width; see
            :class:`looped_moe_gpt2.model.config.AttentionConfig`.
        max_seq_len: Maximum sequence length, used to size the causal mask buffer.
        dropout: Attention dropout probability.
        rope_base: RoPE frequency base.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attention_config: AttentionConfig,
        max_seq_len: int,
        dropout: float = 0.1,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads}).")

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rope_head_dim = attention_config.rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.rope_base = rope_base
        self.dropout_p = dropout

        if self.nope_head_dim < 1:
            raise ValueError(
                f"rope_head_dim ({self.rope_head_dim}) leaves no room for the NoPE slice "
                f"within head_dim ({self.head_dim})."
            )

        d_c = attention_config.kv_compression_dim
        d_c_q = attention_config.q_compression_dim

        # KV path: down-project to shared latent, then up-project per-head for NoPE K/V content.
        self.kv_down_proj = nn.Linear(hidden_size, d_c, bias=False)
        self.kv_up_proj = nn.Linear(d_c, num_heads * (self.nope_head_dim * 2), bias=False)
        # Decoupled RoPE key slice is derived directly from the residual stream, not the latent.
        self.k_rope_proj = nn.Linear(hidden_size, self.rope_head_dim, bias=False)

        # Query path: down-project to a (separate) latent, then up-project per-head.
        self.q_down_proj = nn.Linear(hidden_size, d_c_q, bias=False)
        self.q_up_proj = nn.Linear(d_c_q, num_heads * self.head_dim, bias=False)

        self.out_proj = nn.Linear(num_heads * self.nope_head_dim, hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute causal MLA self-attention.

        Args:
            x: Input tensor of shape ``(batch, seq_len, hidden_size)``.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.

        Raises:
            ValueError: If ``x``'s sequence length exceeds ``max_seq_len``.
        """
        batch_size, seq_len, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds max_seq_len used to configure this "
                f"attention module ({self.max_seq_len})."
            )

        cos, sin = build_rope_cache(
            seq_len=seq_len,
            rope_dim=self.rope_head_dim,
            base=self.rope_base,
            device=x.device,
            dtype=x.dtype,
        )

        # --- Key/Value path ---
        kv_latent = self.kv_down_proj(x)  # (B, T, d_c)
        kv_nope = self.kv_up_proj(kv_latent)  # (B, T, H * 2 * nope_head_dim)
        kv_nope = kv_nope.view(batch_size, seq_len, self.num_heads, 2, self.nope_head_dim)
        k_nope, v = kv_nope.unbind(dim=3)  # each (B, T, H, nope_head_dim)
        k_nope = k_nope.transpose(1, 2)  # (B, H, T, nope_head_dim)
        v = v.transpose(1, 2)  # (B, H, T, nope_head_dim)

        k_rope = self.k_rope_proj(x)  # (B, T, rope_head_dim), shared across heads
        k_rope = k_rope.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (B, H, T, rope_head_dim)
        k_rope = apply_rope(k_rope, cos, sin)

        # --- Query path ---
        q_latent = self.q_down_proj(x)  # (B, T, d_c')
        q = self.q_up_proj(q_latent).view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # (B, H, T, head_dim)
        q_nope, q_rope = q.split([self.nope_head_dim, self.rope_head_dim], dim=-1)
        q_rope = apply_rope(q_rope, cos, sin)

        k = torch.cat([k_nope, k_rope], dim=-1)  # (B, H, T, head_dim)
        q = torch.cat([q_nope, q_rope], dim=-1)  # (B, H, T, head_dim)

        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            # is_causal=True (rather than an explicit boolean attn_mask tensor) lets SDPA
            # dispatch to the Flash Attention / memory-efficient backend, which never
            # materializes a full (B, H, T, T) score matrix -- an explicit mask tensor, even
            # one that is purely causal with no padding, forces a fallback to the much more
            # memory-hungry "math" backend on many PyTorch versions. Confirmed empirically: at
            # seq_len=2048 this was responsible for ~50GB of otherwise-unexplained activation
            # memory (see docs/mamba_investigation.md's H100/A100 profiling section).
            is_causal=True,
        )  # (B, H, T, nope_head_dim) -- value dim is nope_head_dim, not head_dim

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.nope_head_dim)
        out = self.out_proj(out)
        return self.resid_dropout(out)


class StandardMultiHeadAttention(nn.Module):
    """Standard causal multi-head self-attention with optional full-head RoPE.

    Used as the non-MLA baseline path (``AttentionConfig.use_mla=False``).

    Args:
        hidden_size: Model (residual stream) width.
        num_heads: Number of attention heads.
        max_seq_len: Maximum sequence length, used to size the causal mask buffer.
        dropout: Attention dropout probability.
        use_rope: If True, apply standard RoPE to the full per-head dimension.
        rope_base: RoPE frequency base.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        use_rope: bool = False,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads}).")

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.dropout_p = dropout

        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute causal self-attention.

        Args:
            x: Input tensor of shape ``(batch, seq_len, hidden_size)``.

        Returns:
            Output tensor of shape ``(batch, seq_len, hidden_size)``.

        Raises:
            ValueError: If ``x``'s sequence length exceeds ``max_seq_len``.
        """
        batch_size, seq_len, hidden_size = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds max_seq_len used to configure this "
                f"attention module ({self.max_seq_len})."
            )
        qkv = self.qkv_proj(x).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)  # each (B, H, T, head_dim)

        if self.use_rope:
            cos, sin = build_rope_cache(
                seq_len=seq_len,
                rope_dim=self.head_dim,
                base=self.rope_base,
                device=x.device,
                dtype=x.dtype,
            )
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0, is_causal=True
        )  # is_causal=True (not an explicit mask tensor) enables the memory-efficient SDPA
        # backend -- see MultiHeadLatentAttention.forward's comment for the measured impact.
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        return self.resid_dropout(self.out_proj(out))


def build_attention_module(
    hidden_size: int,
    num_heads: int,
    attention_config: AttentionConfig,
    position_encoding: PositionEncodingType,
    max_seq_len: int,
    dropout: float,
) -> nn.Module:
    """Factory selecting MLA or standard attention based on configuration.

    Args:
        hidden_size: Model (residual stream) width.
        num_heads: Number of attention heads.
        attention_config: Attention hyperparameters.
        position_encoding: Positional encoding scheme; determines RoPE usage and validates
            compatibility with ``attention_config.use_mla``.
        max_seq_len: Maximum sequence length.
        dropout: Attention dropout probability.

    Returns:
        An instantiated attention module (:class:`MultiHeadLatentAttention` or
        :class:`StandardMultiHeadAttention`).
    """
    if attention_config.use_mla:
        return MultiHeadLatentAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            attention_config=attention_config,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
    return StandardMultiHeadAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        dropout=dropout,
        use_rope=position_encoding == PositionEncodingType.ROPE,
    )
