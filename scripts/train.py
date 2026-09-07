#!/usr/bin/env python
"""CLI: launch (or resume) a training run from a variant YAML config.

Usage:
    python scripts/train.py --config configs/03_looped_moe_full.yaml

    # Resume an interrupted run from its latest checkpoint:
    python scripts/train.py --config configs/02_looped_dense.yaml --resume runs/02_looped_dense/checkpoint_step1000.pt

See configs/*.yaml for the full ablation sweep this repo is built to run (docs/literature_survey.md §4).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from looped_moe_gpt2.train.trainer import Trainer
from looped_moe_gpt2.utils.config_io import load_model_config, load_train_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Parse CLI arguments, build the model/train configs, and run the training loop."""
    parser = argparse.ArgumentParser(description="Train a looped-MoE-GPT2 variant.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a variant YAML config.")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a checkpoint to resume from (must be a checkpoint from a run using the "
        "SAME --config; resuming with a different architecture config will fail to load).",
    )
    args = parser.parse_args()

    model_config = load_model_config(args.config)
    train_config = load_train_config(args.config)

    logger.info("Loaded model config from %s (effective_depth=%d)", args.config, model_config.effective_depth)
    trainer = Trainer(model_config=model_config, train_config=train_config, resume_from=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
