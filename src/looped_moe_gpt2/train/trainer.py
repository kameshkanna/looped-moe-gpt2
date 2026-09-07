"""Main training loop: AMP, gradient accumulation, cosine LR + MoE bias annealing, checkpointing.

Follows the CLAUDE.md engineering standards this project is held to: `logging` (not `print`)
for execution-flow tracking, `tqdm` for in-place progress on the long-running training loop,
device-agnostic tensor placement, explicit `torch.cuda.empty_cache()` / `gc.collect()` at
checkpoint boundaries, and fail-fast input validation.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np

from looped_moe_gpt2.data.dataset import MemmapTokenDataset, RandomOffsetSampler
from looped_moe_gpt2.model.config import ModelConfig
from looped_moe_gpt2.model.gpt import LoopedMoEGPT
from looped_moe_gpt2.train.checkpoint import load_checkpoint, save_checkpoint
from looped_moe_gpt2.train.config import CurriculumPhase, TrainConfig
from looped_moe_gpt2.train.schedule import cosine_lr_with_warmup, moe_bias_update_speed
from looped_moe_gpt2.utils.device import resolve_amp_dtype, resolve_device
from looped_moe_gpt2.utils.seed import set_seed

logger = logging.getLogger(__name__)


class Trainer:
    """Orchestrates data loading, optimization, evaluation, and checkpointing for one run.

    Args:
        model_config: Architecture configuration; used both to build the model and to
            reconstruct it when resuming from a checkpoint.
        train_config: Optimization/scheduling/logging configuration.
        resume_from: Optional path to a checkpoint (as saved by
            :func:`looped_moe_gpt2.train.checkpoint.save_checkpoint`) to resume training from.
            Restores model weights, optimizer state (Adam moments), and ``best_val_loss``;
            training then continues from that checkpoint's step rather than step 0. The
            checkpoint's own ``model_config`` is NOT used for reconstruction -- the caller's
            ``model_config`` must match the architecture the checkpoint was trained with (the
            same YAML config passed originally), or loading will fail with a state_dict
            shape-mismatch error rather than silently producing a wrong model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        train_config: TrainConfig,
        resume_from: Optional[Path] = None,
    ) -> None:
        self.model_config = model_config
        self.train_config = train_config
        set_seed(train_config.seed)

        self.device = resolve_device()
        self.amp_dtype = resolve_amp_dtype(self.device)

        # `self.model` is always the UNCOMPILED module -- used for state_dict save/load
        # (checkpoint compatibility regardless of whether this or a resuming run compiles) and
        # for attribute access (`.blocks`, `update_all_routing_biases`, etc). When compilation
        # is enabled, `self.model_forward` is the compiled wrapper actually called in the
        # training/eval loops; the two share the same underlying parameters (torch.compile
        # wraps rather than copies), so this adds zero memory and no behavior difference beyond
        # which callable executes the forward pass. Keeping them separate avoids a real gotcha:
        # a compiled module's state_dict() keys are prefixed with "_orig_mod.", which silently
        # breaks --resume checkpoint compatibility between compiled and uncompiled runs if the
        # compiled wrapper is what gets saved/loaded.
        self.model = LoopedMoEGPT(model_config).to(self.device)
        self.model_forward = torch.compile(self.model) if train_config.compile_model else self.model

        self.optimizer = self._build_optimizer()
        self.scaler = torch.amp.GradScaler(enabled=train_config.use_amp and self.amp_dtype == torch.float16)

        self.curriculum_rng = np.random.default_rng(train_config.seed)
        if train_config.curriculum is not None:
            self.phases = train_config.curriculum
            self.phase_train_iters = [
                self._infinite_batches(self._build_dataloader(phase.train_bin_path, shuffle=True))
                for phase in self.phases
            ]
            self.phase_val_loaders = [
                self._build_dataloader(phase.val_bin_path, shuffle=False) for phase in self.phases
            ]
            logger.info(
                "Curriculum enabled with %d phase(s): %s",
                len(self.phases),
                ", ".join(f"{p.name} (start_step={p.start_step})" for p in self.phases),
            )
        else:
            self.phases = None
            self.train_loader = self._build_dataloader(train_config.train_bin_path, shuffle=True)
            self.val_loader = self._build_dataloader(train_config.val_bin_path, shuffle=False)

        self.best_val_loss = float("inf")
        self.start_step = 0

        if resume_from is not None:
            self._load_resume_checkpoint(resume_from)

    def _load_resume_checkpoint(self, checkpoint_path: Path) -> None:
        """Restore model/optimizer state and resume-step from a checkpoint.

        Args:
            checkpoint_path: Path to a checkpoint written by
                :func:`looped_moe_gpt2.train.checkpoint.save_checkpoint`.

        Raises:
            RuntimeError: If the checkpoint's model state_dict shape doesn't match the current
                ``model_config`` (i.e. the wrong config was passed for this checkpoint).
        """
        checkpoint = load_checkpoint(checkpoint_path, self.device)
        try:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load checkpoint '{checkpoint_path}' into a model built from the "
                f"given config -- the config passed to resume training must be the SAME one "
                f"used for the original run. Original error: {e}"
            ) from e
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.start_step = checkpoint["step"]
        logger.info(
            "Resumed from checkpoint '%s' at step %d (best_val_loss=%.4f)",
            checkpoint_path,
            self.start_step,
            self.best_val_loss,
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Construct an AdamW optimizer, excluding norm/bias/routing-bias params from weight decay.

        Returns:
            A configured :class:`torch.optim.AdamW` instance.
        """
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or "norm" in name or "routing_bias" in name or "_bias" in name or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # fused=True: a single CUDA kernel does the AdamW elementwise update instead of several
        # separate reads/writes across gradient, momentum, and variance tensors -- measured
        # ~8.4% faster (314.7ms -> 288.4ms/microbatch on this project's 97.5M router-gated
        # config) with no change to the update math, only fewer kernel launches. No-op / falls
        # back silently on devices where fused AdamW isn't supported (e.g. CPU, MPS).
        use_fused = self.device.type == "cuda"
        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.train_config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.train_config.learning_rate,
            betas=(self.train_config.beta1, self.train_config.beta2),
            fused=use_fused,
        )

    def _build_dataloader(self, bin_path: Path, shuffle: bool) -> DataLoader:
        """Construct a DataLoader over a tokenized binary file.

        Args:
            bin_path: Path to a tokenized token binary (see
                :func:`looped_moe_gpt2.data.tokenize.tokenize_text_file`).
            shuffle: Whether to draw random sample-start offsets (True for training) or iterate
                sequentially (False for validation, where sample order doesn't matter).

        Returns:
            A configured :class:`torch.utils.data.DataLoader`.

        Note:
            When ``shuffle=True`` this uses :class:`~looped_moe_gpt2.data.dataset.
            RandomOffsetSampler` rather than ``DataLoader``'s own ``shuffle=True`` path --
            ``MemmapTokenDataset``'s length is close to the full token count of the corpus (every
            offset is a valid sample start), so the default ``RandomSampler`` would materialize a
            ``torch.randperm`` over hundreds of millions of indices before the first batch could
            be fetched. See :class:`RandomOffsetSampler`'s docstring for the measured impact.
        """
        dataset = MemmapTokenDataset(bin_path, seq_len=self.model_config.max_seq_len)
        sampler = RandomOffsetSampler(dataset, num_samples=len(dataset)) if shuffle else None
        return DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            sampler=sampler,
            shuffle=False,  # shuffling is handled by `sampler` above, not DataLoader itself
            num_workers=0,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
        )

    def _infinite_batches(self, loader: DataLoader):
        """Cycle a DataLoader indefinitely, so training isn't bounded by one epoch's length.

        Args:
            loader: The DataLoader to cycle.

        Yields:
            Successive ``(input_ids, targets)`` batches, re-iterating the loader when exhausted.
        """
        while True:
            for batch in loader:
                yield batch

    def _sample_curriculum_batch(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw one training micro-batch from the curriculum phase mix active at ``step``.

        Each call independently samples ONE phase (weighted by that phase's current mixing
        weight, per :meth:`CurriculumPhase.weight_at_step`) and returns one micro-batch from
        that phase's corpus -- so across many calls, the phase composition of the overall data
        stream matches the configured weights in expectation, without needing to interleave
        multiple corpora within a single micro-batch.

        Args:
            step: Current optimizer step, determining each phase's mixing weight.

        Returns:
            One ``(input_ids, targets)`` micro-batch from the sampled phase's DataLoader.

        Raises:
            RuntimeError: If no phase has positive weight at ``step`` (should be unreachable
                given ``TrainConfig`` requires the first phase to start at step 0).
        """
        weights = np.array([phase.weight_at_step(step) for phase in self.phases])
        total_weight = weights.sum()
        if total_weight <= 0.0:
            raise RuntimeError(
                f"No curriculum phase has positive weight at step {step} -- check phase "
                f"start_step/ramp_steps configuration."
            )
        weights = weights / total_weight
        phase_idx = self.curriculum_rng.choice(len(self.phases), p=weights)
        return next(self.phase_train_iters[phase_idx])

    def _active_phase_weights(self, step: int) -> dict[str, float]:
        """Compute the normalized mixing weight of every curriculum phase at a given step.

        Args:
            step: Current optimizer step.

        Returns:
            A dict mapping phase name to its normalized (summing to 1) mixing weight at
            ``step``. Used for logging so a curriculum run's phase transitions are visible in
            the training log, not just inferred from the config.
        """
        raw_weights = {phase.name: phase.weight_at_step(step) for phase in self.phases}
        total = sum(raw_weights.values())
        if total <= 0.0:
            return {name: 0.0 for name in raw_weights}
        return {name: w / total for name, w in raw_weights.items()}

    def _move_batch_to_device(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Move a batch's tensors to the trainer's compute device.

        Args:
            batch: A ``(input_ids, targets)`` tuple of CPU tensors from the DataLoader.

        Returns:
            The same tuple with both tensors moved to ``self.device``.
        """
        input_ids, targets = batch
        return input_ids.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)

    @torch.no_grad()
    def _evaluate_loader(self, loader: DataLoader) -> float:
        """Compute mean loss over ``TrainConfig.eval_iters`` batches from a single DataLoader.

        Args:
            loader: The DataLoader to evaluate (a single-corpus val loader, or one curriculum
                phase's val loader).

        Returns:
            Mean cross-entropy loss over the sampled batches.
        """
        val_iter = self._infinite_batches(loader)
        losses = torch.zeros(self.train_config.eval_iters, device=self.device)
        for i in range(self.train_config.eval_iters):
            input_ids, targets = self._move_batch_to_device(next(val_iter))
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.train_config.use_amp):
                _, loss = self.model_forward(input_ids, targets=targets)
            losses[i] = loss.item()
        return losses.mean().item()

    @torch.no_grad()
    def evaluate(self, step: int = 0) -> float:
        """Compute validation loss, per curriculum phase if a curriculum is configured.

        Args:
            step: Current optimizer step. Only used for logging which phases are currently
                active in the training mix; every configured phase is still evaluated
                regardless of whether it is currently contributing training batches (so you can
                see, e.g., math-phase loss before the math phase has even started ramping in).

        Returns:
            The single-corpus mean loss (unchanged behavior) when no curriculum is configured;
            otherwise the loss of the FIRST curriculum phase (kept as the scalar returned for
            ``best_val_loss`` tracking / early-stopping compatibility) -- every phase's loss is
            also logged individually via ``logger.info`` before this method returns.
        """
        self.model.eval()
        if self.phases is None:
            result = self._evaluate_loader(self.val_loader)
            self.model.train()
            return result

        active_weights = self._active_phase_weights(step)
        per_phase_loss: dict[str, float] = {}
        for phase, val_loader in zip(self.phases, self.phase_val_loaders):
            per_phase_loss[phase.name] = self._evaluate_loader(val_loader)
        logger.info(
            "step %d | per-phase val_loss: %s | active mix: %s",
            step,
            {name: round(loss, 4) for name, loss in per_phase_loss.items()},
            {name: round(w, 3) for name, w in active_weights.items()},
        )
        self.model.train()
        return per_phase_loss[self.phases[0].name]

    def train(self) -> None:
        """Run the training loop from ``self.start_step`` through ``TrainConfig.max_steps``.

        ``self.start_step`` is 0 for a fresh run, or the checkpointed step when constructed
        with ``resume_from`` -- either way this resumes/starts seamlessly, continuing the same
        LR and MoE bias-annealing schedules (both are pure functions of the absolute step, so
        they pick up exactly where a resumed run left off rather than restarting warmup/decay).

        Logs training/validation loss, learning rate, and (when applicable) MoE expert-load
        statistics at the configured intervals; saves periodic checkpoints; and updates each
        MoE layer's auxiliary-loss-free routing bias after every optimizer step.

        Raises:
            ValueError: If ``self.start_step >= TrainConfig.max_steps`` (nothing left to train --
                the checkpoint resumed from already reached or exceeded the configured target).
        """
        if self.start_step >= self.train_config.max_steps:
            raise ValueError(
                f"start_step ({self.start_step}) >= max_steps ({self.train_config.max_steps}) -- "
                f"this run already completed; nothing to resume. Increase max_steps in the config "
                f"if you want to continue training further."
            )

        self.model.train()
        train_iter = None if self.phases is not None else self._infinite_batches(self.train_loader)
        progress = tqdm(
            range(self.start_step, self.train_config.max_steps),
            desc="Training",
            dynamic_ncols=True,
            initial=self.start_step,
            total=self.train_config.max_steps,
        )
        last_logged_phase_mix: Optional[dict[str, float]] = None

        for step in progress:
            lr = cosine_lr_with_warmup(step, self.train_config)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            bias_speed = moe_bias_update_speed(
                step=step,
                max_steps=self.train_config.max_steps,
                initial_speed=self.model_config.moe.bias_update_speed,
                decay_frac=self.model_config.moe.bias_update_speed_decay_frac,
            )
            for block in self.model.blocks:
                if hasattr(block.feed_forward, "set_bias_update_speed"):
                    block.feed_forward.set_bias_update_speed(bias_speed)

            if self.phases is not None:
                current_mix = self._active_phase_weights(step)
                if current_mix != last_logged_phase_mix:
                    logger.info("step %d | curriculum mix changed: %s", step, {k: round(v, 3) for k, v in current_mix.items()})
                    last_logged_phase_mix = current_mix

            self.optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for _ in range(self.train_config.gradient_accumulation_steps):
                if self.phases is not None:
                    input_ids, targets = self._move_batch_to_device(self._sample_curriculum_batch(step))
                else:
                    input_ids, targets = self._move_batch_to_device(next(train_iter))
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.train_config.use_amp):
                    _, loss = self.model_forward(input_ids, targets=targets)
                    loss = loss / self.train_config.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
                accumulated_loss += loss.item()

            if self.train_config.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.model.update_all_routing_biases()

            if step % self.train_config.log_interval == 0:
                progress.set_postfix(loss=f"{accumulated_loss:.4f}", lr=f"{lr:.2e}")
                logger.info("step %d | train_loss %.4f | lr %.2e", step, accumulated_loss, lr)

            if step > 0 and step % self.train_config.eval_interval == 0:
                val_loss = self.evaluate(step)
                if self.phases is None:
                    logger.info("step %d | val_loss %.4f", step, val_loss)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss

            if step > 0 and step % self.train_config.checkpoint_interval == 0:
                save_checkpoint(
                    output_dir=self.train_config.output_dir,
                    step=step,
                    model=self.model,
                    optimizer=self.optimizer,
                    model_config=self.model_config,
                    best_val_loss=self.best_val_loss,
                )
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

        final_val_loss = self.evaluate(self.train_config.max_steps)
        if self.phases is None:
            logger.info("Training complete. Final val_loss: %.4f", final_val_loss)
        else:
            logger.info("Training complete.")
        save_checkpoint(
            output_dir=self.train_config.output_dir,
            step=self.train_config.max_steps,
            model=self.model,
            optimizer=self.optimizer,
            model_config=self.model_config,
            best_val_loss=min(self.best_val_loss, final_val_loss),
        )
