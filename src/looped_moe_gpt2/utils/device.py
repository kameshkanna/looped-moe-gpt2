"""Device-agnostic helpers: resolve the best available accelerator and matching AMP dtype."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def resolve_device() -> torch.device:
    """Select CUDA if available, then Apple MPS, then fall back to CPU.

    Returns:
        The resolved :class:`torch.device`.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Resolved compute device: %s", device)
    return device


def resolve_amp_dtype(device: torch.device) -> torch.dtype:
    """Select the best autocast dtype for the given device.

    Args:
        device: The compute device automatic mixed precision will run on.

    Returns:
        ``torch.bfloat16`` on CUDA devices that support it, ``torch.float16`` on other CUDA
        devices, and ``torch.float32`` (i.e. no-op autocast) on CPU/MPS.
    """
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32
