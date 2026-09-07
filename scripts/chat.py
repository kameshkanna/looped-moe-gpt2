#!/usr/bin/env python
"""CLI: interactively prompt a trained checkpoint and see its generations.

Usage:
    python scripts/chat.py --checkpoint runs/03_looped_moe_full/checkpoint_step5000.pt \
        --config configs/03_looped_moe_full.yaml

Type a prompt and press Enter to generate a continuation; type 'quit' or Ctrl+C to exit.

NOTE: this model was trained on TinyStories (short children's-story-style text), not
instruction-following data -- it completes text, it does not "answer questions" in a chat
sense. A prompt like "Once upon a time" will work much better than "What is the capital of
France?", since the latter is off-distribution for what this checkpoint ever saw during
training.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import tiktoken
import torch

from looped_moe_gpt2.model.gpt import LoopedMoEGPT
from looped_moe_gpt2.train.checkpoint import load_checkpoint
from looped_moe_gpt2.utils.config_io import load_model_config
from looped_moe_gpt2.utils.device import resolve_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Parse CLI arguments, load the checkpoint, and run an interactive generation loop."""
    parser = argparse.ArgumentParser(description="Interactively prompt a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a .pt checkpoint.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the variant's YAML config.")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Tokens to generate per prompt.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    args = parser.parse_args()

    device = resolve_device()
    model_config = load_model_config(args.config)
    model = LoopedMoEGPT(model_config).to(device)

    checkpoint = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(
        "Loaded checkpoint from step %d (best_val_loss=%.4f)",
        checkpoint["step"],
        checkpoint["best_val_loss"],
    )

    encoder = tiktoken.get_encoding("gpt2")

    print("\nType a prompt and press Enter (or 'quit' to exit).")
    print("This model was trained on TinyStories -- try story-style prompts like:")
    print("  'Once upon a time, there was a little' or 'Tom and Lily went to the'\n")

    while True:
        try:
            prompt = input("Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if prompt.lower() in {"quit", "exit"}:
            break
        if not prompt:
            continue

        input_ids = torch.tensor([encoder.encode_ordinary(prompt)], device=device)
        generated = model.generate(
            input_ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature
        )
        output_text = encoder.decode(generated[0].tolist())
        print(f"\n{output_text}\n")


if __name__ == "__main__":
    main()
