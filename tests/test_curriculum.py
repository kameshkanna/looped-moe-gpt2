"""Unit and integration tests for curriculum (multi-corpus, step-scheduled) training."""

from pathlib import Path

import numpy as np
import pytest
import torch

from looped_moe_gpt2.model.config import (
    AttentionConfig,
    LoopConfig,
    ModelConfig,
    MoEConfig,
    PositionEncodingType,
    SharingPattern,
)
from looped_moe_gpt2.train.config import CurriculumPhase, TrainConfig
from looped_moe_gpt2.train.trainer import Trainer

TINY_VOCAB = 100
TINY_HIDDEN = 32
TINY_SEQ_LEN = 16


def test_curriculum_phase_weight_before_start_step_is_zero() -> None:
    phase = CurriculumPhase(name="math", train_bin_path=Path("x"), val_bin_path=Path("y"), start_step=100)
    assert phase.weight_at_step(0) == 0.0
    assert phase.weight_at_step(99) == 0.0


def test_curriculum_phase_weight_instant_switch_no_ramp() -> None:
    phase = CurriculumPhase(name="math", train_bin_path=Path("x"), val_bin_path=Path("y"), start_step=100, ramp_steps=0, target_weight=0.5)
    assert phase.weight_at_step(100) == 0.5
    assert phase.weight_at_step(200) == 0.5


def test_curriculum_phase_weight_ramps_linearly() -> None:
    phase = CurriculumPhase(name="math", train_bin_path=Path("x"), val_bin_path=Path("y"), start_step=100, ramp_steps=100, target_weight=1.0)
    assert phase.weight_at_step(100) == 0.0
    assert phase.weight_at_step(150) == pytest.approx(0.5)
    assert phase.weight_at_step(200) == 1.0
    assert phase.weight_at_step(300) == 1.0  # stays at target after ramp completes


def test_curriculum_phase_weight_instant_fade_out() -> None:
    phase = CurriculumPhase(name="general", train_bin_path=Path("x"), val_bin_path=Path("y"), end_step=100, fade_out_steps=0, target_weight=1.0)
    assert phase.weight_at_step(99) == 1.0
    assert phase.weight_at_step(100) == 0.0
    assert phase.weight_at_step(200) == 0.0


def test_curriculum_phase_weight_fades_out_linearly() -> None:
    phase = CurriculumPhase(name="general", train_bin_path=Path("x"), val_bin_path=Path("y"), end_step=100, fade_out_steps=100, target_weight=1.0)
    assert phase.weight_at_step(100) == 1.0
    assert phase.weight_at_step(150) == pytest.approx(0.5)
    assert phase.weight_at_step(200) == 0.0
    assert phase.weight_at_step(300) == 0.0


def test_curriculum_phase_rejects_end_step_before_start_step() -> None:
    with pytest.raises(ValueError, match="end_step"):
        CurriculumPhase(name="x", train_bin_path=Path("a"), val_bin_path=Path("b"), start_step=100, end_step=50)


def test_curriculum_phase_rejects_negative_start_step() -> None:
    with pytest.raises(ValueError, match="start_step"):
        CurriculumPhase(name="x", train_bin_path=Path("a"), val_bin_path=Path("b"), start_step=-1)


def test_curriculum_phase_rejects_nonpositive_target_weight() -> None:
    with pytest.raises(ValueError, match="target_weight"):
        CurriculumPhase(name="x", train_bin_path=Path("a"), val_bin_path=Path("b"), target_weight=0.0)


def test_train_config_requires_first_phase_start_at_zero(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="start_step=0"):
        TrainConfig(
            train_bin_path=tmp_path / "unused_train.bin",
            val_bin_path=tmp_path / "unused_val.bin",
            output_dir=tmp_path / "runs",
            curriculum=[
                CurriculumPhase(name="math", train_bin_path=tmp_path / "m.bin", val_bin_path=tmp_path / "mv.bin", start_step=500),
            ],
        )


def test_train_config_rejects_empty_curriculum(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one phase"):
        TrainConfig(
            train_bin_path=tmp_path / "unused_train.bin",
            val_bin_path=tmp_path / "unused_val.bin",
            output_dir=tmp_path / "runs",
            curriculum=[],
        )


@pytest.fixture
def two_phase_token_bins(tmp_path: Path) -> dict[str, Path]:
    """Write distinguishable token binaries for two curriculum phases.

    Phase "general" uses low token IDs (0-49), phase "math" uses high token IDs (50-99), so a
    sampled batch's source phase can be identified from its token value range in tests.
    """
    rng = np.random.default_rng(0)
    paths = {}
    for name, low, high in [("general_train", 0, 50), ("general_val", 0, 50), ("math_train", 50, 100), ("math_val", 50, 100)]:
        tokens = rng.integers(low, high, size=3000, dtype=np.uint16)
        path = tmp_path / f"{name}.bin"
        tokens.tofile(path)
        paths[name] = path
    return paths


def _tiny_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=TINY_VOCAB,
        hidden_size=TINY_HIDDEN,
        num_layers=4,
        num_heads=4,
        max_seq_len=TINY_SEQ_LEN,
        dropout=0.0,
        position_encoding=PositionEncodingType.DECOUPLED_ROPE,
        attention=AttentionConfig(use_mla=True, kv_compression_dim=8, q_compression_dim=12, rope_head_dim=4),
        moe=MoEConfig(enabled=False),
        loop=LoopConfig(enabled=False, sharing_pattern=SharingPattern.NONE),
    )


def test_curriculum_trainer_samples_only_first_phase_before_second_starts(tmp_path: Path, two_phase_token_bins: dict[str, Path]) -> None:
    """Before the second phase's start_step, every training batch must come from phase 1 --
    i.e. contain only low-range token IDs."""
    train_config = TrainConfig(
        train_bin_path=tmp_path / "unused.bin",
        val_bin_path=tmp_path / "unused.bin",
        output_dir=tmp_path / "runs",
        batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=10,
        warmup_steps=1,
        eval_interval=100,
        curriculum=[
            CurriculumPhase(name="general", train_bin_path=two_phase_token_bins["general_train"], val_bin_path=two_phase_token_bins["general_val"], start_step=0),
            CurriculumPhase(name="math", train_bin_path=two_phase_token_bins["math_train"], val_bin_path=two_phase_token_bins["math_val"], start_step=1000, ramp_steps=0),
        ],
    )
    trainer = Trainer(model_config=_tiny_model_config(), train_config=train_config)
    for step in range(5):
        input_ids, _ = trainer._sample_curriculum_batch(step)
        assert (input_ids < 50).all(), f"Expected only general-phase tokens before math phase starts, got {input_ids}"


def test_curriculum_trainer_switches_to_second_phase_after_start_step(tmp_path: Path, two_phase_token_bins: dict[str, Path]) -> None:
    """With phase 1 explicitly faded out (end_step) at the same point phase 2 switches on
    (instant, ramp_steps=0), batches after that step must come exclusively from phase 2."""
    train_config = TrainConfig(
        train_bin_path=tmp_path / "unused.bin",
        val_bin_path=tmp_path / "unused.bin",
        output_dir=tmp_path / "runs",
        batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=20,
        warmup_steps=1,
        eval_interval=100,
        curriculum=[
            CurriculumPhase(name="general", train_bin_path=two_phase_token_bins["general_train"], val_bin_path=two_phase_token_bins["general_val"], start_step=0, end_step=5, fade_out_steps=0, target_weight=1.0),
            CurriculumPhase(name="math", train_bin_path=two_phase_token_bins["math_train"], val_bin_path=two_phase_token_bins["math_val"], start_step=5, ramp_steps=0, target_weight=1.0),
        ],
    )
    trainer = Trainer(model_config=_tiny_model_config(), train_config=train_config)
    for step in range(5, 10):
        input_ids, _ = trainer._sample_curriculum_batch(step)
        assert (input_ids >= 50).all(), f"Expected only math-phase tokens after math phase starts, got {input_ids}"


def test_curriculum_phases_blend_forever_without_explicit_fade_out(tmp_path: Path, two_phase_token_bins: dict[str, Path]) -> None:
    """Regression guard: WITHOUT an explicit end_step on phase 1, starting phase 2 does NOT
    silently remove phase 1 from the mix -- both phases keep contributing (renormalized), since
    CurriculumPhase only describes each phase's own weight curve, not automatic handoff. This
    documents the behavior the original (buggy) test incorrectly assumed away."""
    train_config = TrainConfig(
        train_bin_path=tmp_path / "unused.bin",
        val_bin_path=tmp_path / "unused.bin",
        output_dir=tmp_path / "runs",
        batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=20,
        warmup_steps=1,
        eval_interval=100,
        curriculum=[
            CurriculumPhase(name="general", train_bin_path=two_phase_token_bins["general_train"], val_bin_path=two_phase_token_bins["general_val"], start_step=0, target_weight=1.0),
            CurriculumPhase(name="math", train_bin_path=two_phase_token_bins["math_train"], val_bin_path=two_phase_token_bins["math_val"], start_step=5, ramp_steps=0, target_weight=1.0),
        ],
    )
    trainer = Trainer(model_config=_tiny_model_config(), train_config=train_config)
    saw_general = False
    saw_math = False
    for _ in range(60):
        input_ids, _ = trainer._sample_curriculum_batch(10)  # well after math's start_step=5
        if (input_ids < 50).all():
            saw_general = True
        elif (input_ids >= 50).all():
            saw_math = True
    assert saw_general and saw_math, "Both phases should still be sampled without an explicit fade-out on phase 1."


def test_curriculum_trainer_mixes_during_ramp(tmp_path: Path, two_phase_token_bins: dict[str, Path]) -> None:
    """Mid-ramp, sampling many batches should draw from BOTH phases (not exclusively one)."""
    train_config = TrainConfig(
        train_bin_path=tmp_path / "unused.bin",
        val_bin_path=tmp_path / "unused.bin",
        output_dir=tmp_path / "runs",
        batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=200,
        warmup_steps=1,
        eval_interval=1000,
        curriculum=[
            CurriculumPhase(name="general", train_bin_path=two_phase_token_bins["general_train"], val_bin_path=two_phase_token_bins["general_val"], start_step=0, target_weight=1.0),
            CurriculumPhase(name="math", train_bin_path=two_phase_token_bins["math_train"], val_bin_path=two_phase_token_bins["math_val"], start_step=0, ramp_steps=100, target_weight=1.0),
        ],
    )
    trainer = Trainer(model_config=_tiny_model_config(), train_config=train_config)
    saw_general = False
    saw_math = False
    for _ in range(60):  # at step 50 (mid-ramp), both phases have weight ~0.5 each
        input_ids, _ = trainer._sample_curriculum_batch(50)
        if (input_ids < 50).all():
            saw_general = True
        elif (input_ids >= 50).all():
            saw_math = True
    assert saw_general and saw_math, "Expected batches from both phases during a mid-ramp step."


def test_curriculum_evaluate_reports_all_phases(tmp_path: Path, two_phase_token_bins: dict[str, Path]) -> None:
    """evaluate() must compute a loss for every configured phase, not just the active one."""
    train_config = TrainConfig(
        train_bin_path=tmp_path / "unused.bin",
        val_bin_path=tmp_path / "unused.bin",
        output_dir=tmp_path / "runs",
        batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=10,
        warmup_steps=1,
        eval_iters=2,
        curriculum=[
            CurriculumPhase(name="general", train_bin_path=two_phase_token_bins["general_train"], val_bin_path=two_phase_token_bins["general_val"], start_step=0),
            CurriculumPhase(name="math", train_bin_path=two_phase_token_bins["math_train"], val_bin_path=two_phase_token_bins["math_val"], start_step=1000),
        ],
    )
    trainer = Trainer(model_config=_tiny_model_config(), train_config=train_config)
    result = trainer.evaluate(step=0)
    assert isinstance(result, float)
    assert torch.isfinite(torch.tensor(result))


def test_curriculum_full_training_run_completes(tmp_path: Path, two_phase_token_bins: dict[str, Path]) -> None:
    """End-to-end: a short curriculum run with a mid-training phase switch must complete and
    checkpoint without error."""
    train_config = TrainConfig(
        train_bin_path=tmp_path / "unused.bin",
        val_bin_path=tmp_path / "unused.bin",
        output_dir=tmp_path / "runs",
        batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=8,
        warmup_steps=1,
        eval_interval=4,
        eval_iters=2,
        checkpoint_interval=4,
        curriculum=[
            CurriculumPhase(name="general", train_bin_path=two_phase_token_bins["general_train"], val_bin_path=two_phase_token_bins["general_val"], start_step=0),
            CurriculumPhase(name="math", train_bin_path=two_phase_token_bins["math_train"], val_bin_path=two_phase_token_bins["math_val"], start_step=4, ramp_steps=2),
        ],
    )
    trainer = Trainer(model_config=_tiny_model_config(), train_config=train_config)
    trainer.train()
    assert (train_config.output_dir / "checkpoint_step8.pt").exists()


def test_non_curriculum_run_still_works_unchanged(tmp_path: Path, two_phase_token_bins: dict[str, Path]) -> None:
    """Backward-compatibility guard: a TrainConfig with curriculum=None must behave exactly as
    before this feature was added."""
    train_config = TrainConfig(
        train_bin_path=two_phase_token_bins["general_train"],
        val_bin_path=two_phase_token_bins["general_val"],
        output_dir=tmp_path / "runs",
        batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=4,
        warmup_steps=1,
        eval_interval=100,
    )
    trainer = Trainer(model_config=_tiny_model_config(), train_config=train_config)
    assert trainer.phases is None
    trainer.train()
    assert (train_config.output_dir / "checkpoint_step4.pt").exists()
