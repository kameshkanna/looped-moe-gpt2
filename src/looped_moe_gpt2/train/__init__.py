"""Training subpackage: config, LR/bias schedules, checkpointing, diagnostics, and the trainer."""

from looped_moe_gpt2.train.config import TrainConfig
from looped_moe_gpt2.train.trainer import Trainer

__all__ = ["TrainConfig", "Trainer"]
