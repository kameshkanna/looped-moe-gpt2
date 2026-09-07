"""Training-run configuration, kept separate from model architecture configuration.

Centralizing training hyperparameters here (rather than scattering them as CLI defaults or
magic numbers in the training loop) keeps every run reproducible from a single serialized
config alongside the model checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CurriculumPhase:
    """One phase of a multi-corpus training curriculum.

    IMPORTANT: this class only describes ONE phase's own weight curve (fade-in via
    ``start_step``/``ramp_steps``, optional fade-out via ``end_step``/``fade_out_steps``) --
    it does NOT automatically reduce an EARLIER phase's weight just because a later phase in
    the same ``TrainConfig.curriculum`` list starts. If you want phase A to hand off to phase B
    (rather than both contributing forever, diluted by renormalization), set phase A's
    ``end_step``/``fade_out_steps`` explicitly to fade it out around when phase B ramps in.

    Attributes:
        name: Human-readable phase label (used in logging), e.g. ``"general"`` or ``"math"``.
        train_bin_path: Path to this phase's tokenized training token binary.
        val_bin_path: Path to this phase's tokenized validation token binary.
        start_step: The optimizer step at which this phase's corpus first contributes any
            probability mass to the training batch mix.
        ramp_steps: Number of steps over which this phase's mixing weight linearly ramps from 0
            to ``target_weight``, starting at ``start_step``. 0 means an instant switch-on
            (this phase jumps directly to ``target_weight`` at ``start_step``) rather than a
            gradual blend-in.
        end_step: Optional optimizer step at which this phase begins fading out (linearly, over
            ``fade_out_steps``) toward weight 0. None (the default) means this phase's weight
            never decreases once ramped in -- it stays at ``target_weight`` for the rest of
            training.
        fade_out_steps: Number of steps over which this phase's weight ramps from
            ``target_weight`` down to 0, starting at ``end_step``. Ignored if ``end_step`` is
            None. 0 means an instant switch-off at ``end_step``.
        target_weight: The mixing weight (relative to other concurrently-active phases) this
            phase reaches once its fade-in ramp completes. Weights across all phases active at
            a given step are renormalized to sum to 1 when sampling each micro-batch's source
            corpus.
    """

    name: str
    train_bin_path: Path
    val_bin_path: Path
    start_step: int = 0
    ramp_steps: int = 0
    end_step: Optional[int] = None
    fade_out_steps: int = 0
    target_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.start_step < 0:
            raise ValueError("start_step must be non-negative.")
        if self.ramp_steps < 0:
            raise ValueError("ramp_steps must be non-negative.")
        if self.fade_out_steps < 0:
            raise ValueError("fade_out_steps must be non-negative.")
        if self.target_weight <= 0.0:
            raise ValueError("target_weight must be positive.")
        if self.end_step is not None and self.end_step < self.start_step:
            raise ValueError(
                f"end_step ({self.end_step}) must not precede start_step ({self.start_step})."
            )

    def weight_at_step(self, step: int) -> float:
        """Compute this phase's (un-normalized) mixing weight at a given training step.

        Args:
            step: Current optimizer step.

        Returns:
            0.0 before ``start_step``; linearly ramping from 0 to ``target_weight`` over
            ``ramp_steps`` starting at ``start_step``; ``target_weight`` until ``end_step`` (if
            set); linearly ramping back down to 0 over ``fade_out_steps`` starting at
            ``end_step``; 0.0 after the fade-out completes.
        """
        if step < self.start_step:
            return 0.0

        fade_in_progress = 1.0 if self.ramp_steps == 0 else min(1.0, (step - self.start_step) / self.ramp_steps)
        weight = self.target_weight * fade_in_progress

        if self.end_step is not None and step >= self.end_step:
            if self.fade_out_steps == 0:
                return 0.0
            fade_out_progress = min(1.0, (step - self.end_step) / self.fade_out_steps)
            weight = weight * (1.0 - fade_out_progress)

        return weight


@dataclass(frozen=True)
class TrainConfig:
    """Optimization, scheduling, and logging hyperparameters for a training run.

    Attributes:
        train_bin_path: Path to the tokenized training token binary. Ignored (but must still be
            given a placeholder value, since the field has no default) when ``curriculum`` is
            set -- the curriculum's phases each carry their own paths instead.
        val_bin_path: Path to the tokenized validation token binary. Ignored under the same
            condition as ``train_bin_path``; validation during a curriculum run reports each
            active phase's loss separately (see :meth:`Trainer.evaluate`).
        curriculum: Optional list of :class:`CurriculumPhase` entries for a multi-corpus,
            step-scheduled training run (e.g. general text early, reasoning-dense data
            introduced partway through). When None (the default), the run uses the single
            ``train_bin_path``/``val_bin_path`` corpus for its entire duration, unchanged from
            this class's original behavior.
        output_dir: Directory for checkpoints and run logs.
        batch_size: Micro-batch size (per optimizer step, before gradient accumulation).
        gradient_accumulation_steps: Number of micro-batches accumulated before an optimizer
            step, for approximating a larger effective batch size under limited GPU memory.
        max_steps: Total number of optimizer steps to run.
        learning_rate: Peak learning rate for the cosine schedule.
        min_learning_rate: Final learning rate at the end of the cosine schedule.
        warmup_steps: Number of linear warmup steps at the start of training.
        weight_decay: AdamW weight decay coefficient.
        beta1: AdamW beta1.
        beta2: AdamW beta2.
        grad_clip_norm: Maximum gradient norm for clipping (0 disables clipping).
        eval_interval: Run validation every this many optimizer steps.
        eval_iters: Number of validation batches to average per evaluation.
        log_interval: Log training metrics every this many optimizer steps.
        checkpoint_interval: Save a checkpoint every this many optimizer steps.
        use_amp: If True, use automatic mixed precision (bfloat16 if supported, else float16).
        seed: Random seed for the training loop's own RNG usage (data shuffling order).
        compile_model: If True, wrap the model with ``torch.compile`` for faster execution.
    """

    train_bin_path: Path
    val_bin_path: Path
    output_dir: Path
    curriculum: Optional[list[CurriculumPhase]] = None
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_steps: int = 5000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip_norm: float = 1.0
    eval_interval: int = 250
    eval_iters: int = 50
    log_interval: int = 20
    checkpoint_interval: int = 500
    use_amp: bool = True
    seed: int = 1337
    compile_model: bool = False

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("batch_size and gradient_accumulation_steps must be >= 1.")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1.")
        if self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be < max_steps.")
        if not 0.0 < self.learning_rate:
            raise ValueError("learning_rate must be positive.")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must not exceed learning_rate.")
        if self.curriculum is not None:
            if len(self.curriculum) == 0:
                raise ValueError("curriculum, if given, must contain at least one phase.")
            if self.curriculum[0].start_step != 0:
                raise ValueError(
                    "The first curriculum phase must have start_step=0, or no corpus would be "
                    "active at the start of training."
                )
