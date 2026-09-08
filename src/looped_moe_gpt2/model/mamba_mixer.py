"""Mamba-2 state-space sequence mixer wrapper (arXiv:2405.21060).

Thin adapter around ``mamba_ssm.Mamba2`` so it plugs into
:class:`~looped_moe_gpt2.model.block.TransformerBlock` with the same
``forward(x) -> tensor`` signature as :mod:`looped_moe_gpt2.model.attention`'s mixers, keeping
the block itself agnostic to which sequence-mixing mechanism it wraps. ``mamba_ssm`` is an
optional dependency (CUDA-kernel-compiled, only installable via WSL/Linux per
``docs/mamba_investigation.md``); the import is deferred into the constructor so the rest of
this package remains importable without it on platforms where it cannot be built.
"""

from __future__ import annotations

import torch
from torch import nn

from looped_moe_gpt2.model.config import MambaConfig


class Mamba2Mixer(nn.Module):
    """Wraps ``mamba_ssm.Mamba2`` as a drop-in sequence mixer for :class:`TransformerBlock`.

    Args:
        hidden_size: Model (residual stream) width; ``mamba_ssm.Mamba2``'s ``d_model``.
        mamba_config: Mamba-2 dimension hyperparameters; see :class:`MambaConfig`.
        dropout: Dropout probability applied to the mixer's output before the residual add,
            matching the convention of :mod:`looped_moe_gpt2.model.attention`'s mixers (Mamba-2
            itself has no internal dropout).

    Raises:
        ImportError: If ``mamba_ssm`` is not installed. See ``docs/mamba_investigation.md`` for
            the WSL-based install path required for its CUDA/Triton kernels.
    """

    def __init__(self, hidden_size: int, mamba_config: MambaConfig, dropout: float = 0.1) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba2
        except ImportError as exc:
            raise ImportError(
                "mamba_ssm is required for MixerType.MAMBA2 / HYBRID_MAMBA_ATTENTION but is not "
                "installed. It requires CUDA-kernel compilation and is not reliably installable "
                "on native Windows; see docs/mamba_investigation.md for the WSL install path."
            ) from exc

        self.mamba = Mamba2(
            d_model=hidden_size,
            d_state=mamba_config.d_state,
            d_conv=mamba_config.d_conv,
            expand=mamba_config.expand,
            headdim=mamba_config.headdim,
            ngroups=mamba_config.ngroups,
        )
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Mamba-2 mixer.

        Args:
            x: Input tensor of shape ``(batch, seq_len, hidden_size)``.

        Returns:
            Output tensor of the same shape as ``x``.
        """
        return self.resid_dropout(self.mamba(x))
