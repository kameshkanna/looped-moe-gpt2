"""YAML <-> dataclass config loading for reproducible, file-defined experiment variants."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from looped_moe_gpt2.model.config import (
    AttentionConfig,
    LoopConfig,
    MambaConfig,
    MixerType,
    ModelConfig,
    MoEConfig,
    PositionEncodingType,
    RouterConfig,
    SharingPattern,
)
from looped_moe_gpt2.train.config import CurriculumPhase, TrainConfig


def load_model_config(yaml_path: Path) -> ModelConfig:
    """Load a :class:`ModelConfig` from a YAML file.

    Args:
        yaml_path: Path to a YAML file with a ``model:`` top-level key mirroring
            :class:`ModelConfig`'s fields (nested ``attention:``, ``moe:``, ``loop:`` blocks).

    Returns:
        A validated :class:`ModelConfig` instance.

    Raises:
        FileNotFoundError: If ``yaml_path`` does not exist.
        KeyError: If the YAML is missing the required ``model:`` top-level key.
    """
    raw = _load_yaml(yaml_path)
    if "model" not in raw:
        raise KeyError(f"YAML config {yaml_path} is missing the required 'model:' top-level key.")
    model_raw: dict[str, Any] = dict(raw["model"])

    attention_raw = model_raw.pop("attention", {})
    moe_raw = model_raw.pop("moe", {})
    loop_raw = model_raw.pop("loop", {})
    mamba_raw = model_raw.pop("mamba", None)
    router_raw = loop_raw.pop("router", None)

    if "position_encoding" in model_raw:
        model_raw["position_encoding"] = PositionEncodingType(model_raw["position_encoding"])
    if "mixer_type" in model_raw:
        model_raw["mixer_type"] = MixerType(model_raw["mixer_type"])
    if "sharing_pattern" in loop_raw:
        loop_raw["sharing_pattern"] = SharingPattern(loop_raw["sharing_pattern"])
    if router_raw is not None:
        loop_raw["router"] = RouterConfig(**router_raw)

    return ModelConfig(
        attention=AttentionConfig(**attention_raw),
        moe=MoEConfig(**moe_raw),
        loop=LoopConfig(**loop_raw),
        mamba=MambaConfig(**mamba_raw) if mamba_raw is not None else None,
        **model_raw,
    )


def load_train_config(yaml_path: Path) -> TrainConfig:
    """Load a :class:`TrainConfig` from a YAML file.

    Args:
        yaml_path: Path to a YAML file with a ``train:`` top-level key mirroring
            :class:`TrainConfig`'s fields.

    Returns:
        A validated :class:`TrainConfig` instance.

    Raises:
        FileNotFoundError: If ``yaml_path`` does not exist.
        KeyError: If the YAML is missing the required ``train:`` top-level key.
    """
    raw = _load_yaml(yaml_path)
    if "train" not in raw:
        raise KeyError(f"YAML config {yaml_path} is missing the required 'train:' top-level key.")
    train_raw: dict[str, Any] = dict(raw["train"])
    train_raw["train_bin_path"] = Path(train_raw["train_bin_path"])
    train_raw["val_bin_path"] = Path(train_raw["val_bin_path"])
    train_raw["output_dir"] = Path(train_raw["output_dir"])

    curriculum_raw = train_raw.pop("curriculum", None)
    if curriculum_raw is not None:
        train_raw["curriculum"] = [
            CurriculumPhase(
                name=phase["name"],
                train_bin_path=Path(phase["train_bin_path"]),
                val_bin_path=Path(phase["val_bin_path"]),
                start_step=phase.get("start_step", 0),
                ramp_steps=phase.get("ramp_steps", 0),
                end_step=phase.get("end_step"),
                fade_out_steps=phase.get("fade_out_steps", 0),
                target_weight=phase.get("target_weight", 1.0),
            )
            for phase in curriculum_raw
        ]

    return TrainConfig(**train_raw)


def _load_yaml(yaml_path: Path) -> dict[str, Any]:
    """Read and parse a YAML file.

    Args:
        yaml_path: Path to the YAML file.

    Returns:
        The parsed YAML content as a dict.

    Raises:
        FileNotFoundError: If ``yaml_path`` does not exist.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
