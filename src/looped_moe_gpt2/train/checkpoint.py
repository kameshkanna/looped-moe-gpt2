"""Checkpoint save/load utilities: model, optimizer, and run-state persistence."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from looped_moe_gpt2.model.config import ModelConfig
from looped_moe_gpt2.model.gpt import LoopedMoEGPT

logger = logging.getLogger(__name__)


def save_checkpoint(
    output_dir: Path,
    step: int,
    model: LoopedMoEGPT,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    best_val_loss: float,
) -> Path:
    """Persist model weights, optimizer state, and run metadata to disk.

    Args:
        output_dir: Directory to write the checkpoint into.
        step: Current optimizer step, used to name the checkpoint file.
        model: The model to checkpoint.
        optimizer: The optimizer whose state (e.g. Adam moments) should be checkpointed.
        model_config: The architecture configuration used to build ``model`` (needed to
            reconstruct the model when loading, since it is not itself an ``nn.Module``).
        best_val_loss: Best validation loss observed so far, for resuming early-stopping logic.

    Returns:
        Path to the written checkpoint file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"checkpoint_step{step}.pt"
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": asdict(model_config),
            "best_val_loss": best_val_loss,
        },
        checkpoint_path,
    )
    logger.info("Saved checkpoint to %s", checkpoint_path)
    return checkpoint_path


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    """Load a checkpoint dict from disk onto the given device.

    Args:
        checkpoint_path: Path to a checkpoint written by :func:`save_checkpoint`.
        device: Device to map the loaded tensors onto.

    Returns:
        The raw checkpoint dictionary (keys: ``step``, ``model_state_dict``,
        ``optimizer_state_dict``, ``model_config``, ``best_val_loss``).

    Raises:
        FileNotFoundError: If ``checkpoint_path`` does not exist.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    logger.info("Loaded checkpoint from %s (step %d)", checkpoint_path, checkpoint["step"])
    return checkpoint
