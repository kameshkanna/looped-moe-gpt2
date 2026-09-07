"""Rotary position embedding (RoPE) utilities, including DeepSeek-V3-style decoupled RoPE.

Standard RoPE (Su et al., 2021) rotates the full per-head query/key vector by an angle
proportional to sequence position. Decoupled RoPE (DeepSeek-V3, arXiv:2412.19437; reused by
Kimi K2) instead rotates only a small slice of each head's dimension, leaving the remainder
("NoPE") position-agnostic and derived from a compressed latent — see
:mod:`looped_moe_gpt2.model.attention` and ``docs/literature_survey.md`` §3.1 for how this
composes with Multi-head Latent Attention.

All functions here are device-agnostic: they operate on whatever device/dtype the input
tensors already live on.
"""

from __future__ import annotations

import torch


def build_rope_cache(
    seq_len: int, rope_dim: int, base: float, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cosine/sine tables for a given sequence length and rotary dimension.

    Args:
        seq_len: Number of positions to precompute (typically the current batch's sequence
            length or the model's max_seq_len).
        rope_dim: Dimensionality that will be rotated (must be even). For standard RoPE this is
            the full per-head dimension; for decoupled RoPE this is ``AttentionConfig.
            rope_head_dim``.
        base: RoPE frequency base (10000.0 is the standard choice from the original paper).
        device: Device to allocate the cache on.
        dtype: Floating point dtype for the cache.

    Returns:
        A tuple ``(cos, sin)`` each of shape ``(seq_len, rope_dim)``, ready to broadcast against
        a ``(..., seq_len, rope_dim)`` query/key tensor.

    Raises:
        ValueError: If ``rope_dim`` is not even (RoPE rotates in 2D sub-planes).
    """
    if rope_dim % 2 != 0:
        raise ValueError(f"rope_dim must be even for RoPE rotation, got {rope_dim}.")

    inv_freq = 1.0 / (
        base ** (torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32) / rope_dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # (seq_len, rope_dim // 2)
    freqs = torch.cat([freqs, freqs], dim=-1)  # (seq_len, rope_dim)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension of ``x`` by splitting it in half and swapping with negation.

    Args:
        x: Tensor of shape ``(..., rope_dim)`` with an even last dimension.

    Returns:
        Tensor of the same shape, representing the 90-degree-rotated component used by RoPE's
        complex-multiplication formulation.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding to a query/key tensor.

    Args:
        x: Tensor of shape ``(batch, num_heads, seq_len, rope_dim)`` to rotate.
        cos: Cosine cache of shape ``(seq_len, rope_dim)`` from :func:`build_rope_cache`.
        sin: Sine cache of shape ``(seq_len, rope_dim)`` from :func:`build_rope_cache`.

    Returns:
        Rotated tensor of the same shape as ``x``.

    Raises:
        ValueError: If the trailing dimension of ``x`` does not match the cache's rotary
            dimension.
    """
    if x.shape[-1] != cos.shape[-1]:
        raise ValueError(
            f"Input rotary dimension ({x.shape[-1]}) does not match RoPE cache dimension "
            f"({cos.shape[-1]})."
        )
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin
