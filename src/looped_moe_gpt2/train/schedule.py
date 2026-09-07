"""Learning-rate and MoE bias-update-speed schedules.

Both schedules are pure functions of the current step and the :class:`TrainConfig` /
:class:`MoEConfig`, so they are trivially unit-testable without instantiating a model or
optimizer.
"""

from __future__ import annotations

import math

from looped_moe_gpt2.train.config import TrainConfig


def cosine_lr_with_warmup(step: int, config: TrainConfig) -> float:
    """Compute the learning rate for ``step`` under linear warmup + cosine decay.

    Args:
        step: Current optimizer step (0-indexed).
        config: Training configuration providing warmup/decay hyperparameters.

    Returns:
        The learning rate to use at this step.
    """
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    if step >= config.max_steps:
        return config.min_learning_rate
    decay_ratio = (step - config.warmup_steps) / max(config.max_steps - config.warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_learning_rate + coeff * (config.learning_rate - config.min_learning_rate)


def moe_bias_update_speed(step: int, max_steps: int, initial_speed: float, decay_frac: float) -> float:
    """Linearly anneal the MoE auxiliary-loss-free bias update speed to zero near training's end.

    DeepSeek-V3 uses a constant bias update speed for most of training, then sets it to zero for
    the final portion (see ``docs/literature_survey.md`` §3.2). This is a smoother linear
    anneal over the final ``decay_frac`` fraction of steps rather than a hard cutoff.

    Args:
        step: Current optimizer step (0-indexed).
        max_steps: Total number of optimizer steps in the run.
        initial_speed: Bias update speed used before the decay window begins.
        decay_frac: Fraction of ``max_steps``, at the end of training, over which the speed is
            linearly annealed from ``initial_speed`` to 0.

    Returns:
        The bias update speed to use at this step.
    """
    decay_start_step = max_steps * (1.0 - decay_frac)
    if step < decay_start_step or decay_frac <= 0.0:
        return initial_speed
    progress = (step - decay_start_step) / max(max_steps - decay_start_step, 1)
    return initial_speed * max(0.0, 1.0 - progress)
