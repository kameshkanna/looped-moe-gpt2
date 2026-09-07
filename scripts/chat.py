#!/usr/bin/env python
"""CLI: interactively prompt a trained checkpoint and see its generations.

Usage:
    python scripts/chat.py --checkpoint runs/03_looped_moe_full/checkpoint_step5000.pt \
        --config configs/03_looped_moe_full.yaml
    python scripts/chat.py --hub-repo Kameshr/looped-moe-gpt2-reasoning

Type a prompt and press Enter to generate a continuation; type 'quit' or Ctrl+C to exit.

IMPORTANT: every checkpoint in this project (TinyStories variants 00-04, or the curriculum
general+math variant 05) is a BASE model -- trained purely on next-token prediction over its
training corpus, with no instruction-tuning/RLHF pass on top. None of them "answer questions"
in a chat sense; they complete text in the style of whatever they were trained on. A prompt like
"What is 2 + 2" will get a plausible-sounding CONTINUATION of that text (e.g. more prose in the
training corpus's register), not a computed answer -- the model was never trained on
question-directly-followed-by-answer supervision, so there's no learned behavior to produce
that shape of response. This script prints the loaded checkpoint's own curriculum/corpus info
at startup so you know what kind of completions to expect from it.

--system-prompt and --template DO NOT make this an instruction-follower -- there is no chat
turn structure, role tokens, or system/user/assistant distinction anywhere in this model's
training data, so nothing resembling that concept is "understood" by the model. What they
actually do:
  --system-prompt TEXT   Prepends TEXT before your input every turn, as plain leading context
                          (exactly as if you'd typed it yourself first). This can bias the
                          completion's register/topic (e.g. a system prompt full of formal prose
                          nudges toward more formal continuations) purely because the model
                          conditions on all preceding tokens -- it is NOT understood as an
                          instruction to obey, just more context to continue from.
  --template math         Wraps your input in the EXACT "Problem: {input}\nSolution:" format
                          used for every math example during training (see
                          scripts/prepare_curriculum_data.py's _iter_math_texts) -- this is the
                          one structural cue the model actually saw thousands of times, so it is
                          the closest thing to a real "template" this checkpoint has.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import tiktoken
import torch

from looped_moe_gpt2.model.gpt import LoopedMoEGPT
from looped_moe_gpt2.train.checkpoint import load_checkpoint
from looped_moe_gpt2.utils.config_io import load_model_config, load_train_config
from looped_moe_gpt2.utils.device import resolve_device
from looped_moe_gpt2.utils.hub import resolve_checkpoint_and_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_TEMPLATES = {
    "math": "Problem: {input}\nSolution:",
}


def build_prompt(user_input: str, template: str, system_prompt: str | None) -> str:
    """Compose the final text sent to the model from the user's input, template, and system prompt.

    Args:
        user_input: The raw text the user typed.
        template: One of ``"none"`` (send ``user_input`` unchanged) or a key in ``_TEMPLATES``
            (wrap ``user_input`` in that template's format string).
        system_prompt: If not None, prepended before the (possibly templated) input, separated
            by a blank line -- plain leading context, not an instruction (see the module
            docstring for why this base model has no learned concept of "instruction").

    Returns:
        The final prompt string to tokenize and generate from.

    Raises:
        ValueError: If ``template`` is not ``"none"`` and not a recognized template key.
    """
    if template == "none":
        body = user_input
    elif template in _TEMPLATES:
        body = _TEMPLATES[template].format(input=user_input)
    else:
        raise ValueError(f"Unknown template '{template}'; expected 'none' or one of {list(_TEMPLATES)}.")

    if system_prompt is not None:
        return f"{system_prompt}\n\n{body}"
    return body


def main() -> None:
    """Parse CLI arguments, load the checkpoint, and run an interactive generation loop."""
    parser = argparse.ArgumentParser(description="Interactively prompt a trained checkpoint.")
    parser.add_argument(
        "--hub-repo",
        type=str,
        default=None,
        help="Load checkpoint.pt + config.yaml from this Hugging Face Hub repo id "
        "(e.g. 'Kameshr/looped-moe-gpt2-reasoning') instead of local --checkpoint/--config paths.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a local .pt checkpoint.")
    parser.add_argument("--config", type=Path, default=None, help="Path to the variant's local YAML config.")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Tokens to generate per prompt.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Plain text prepended before your input every turn, as leading context (NOT an "
        "instruction the model is trained to obey -- see the module docstring for what this "
        "actually does to a base model).",
    )
    parser.add_argument(
        "--template",
        choices=["none", "math"],
        default="none",
        help="'math' wraps your input as 'Problem: {input}\\nSolution:', the exact format used "
        "for every math training example (see the module docstring). 'none' (default) sends "
        "your input as-is.",
    )
    args = parser.parse_args()

    if args.hub_repo is not None:
        if args.checkpoint is not None or args.config is not None:
            raise ValueError("Pass either --hub-repo, or --checkpoint/--config, not both.")
        checkpoint_path, config_path = resolve_checkpoint_and_config(args.hub_repo)
    elif args.checkpoint is not None and args.config is not None:
        checkpoint_path, config_path = args.checkpoint, args.config
    else:
        raise ValueError("Must pass either --hub-repo, or both --checkpoint and --config.")

    device = resolve_device()
    model_config = load_model_config(config_path)
    model = LoopedMoEGPT(model_config).to(device)

    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(
        "Loaded checkpoint from step %d (best_val_loss=%.4f)",
        checkpoint["step"],
        checkpoint["best_val_loss"],
    )

    encoder = tiktoken.get_encoding("gpt2")

    train_config = load_train_config(config_path)
    print("\nType a prompt and press Enter (or 'quit' to exit).")
    print("This is a BASE model (no instruction-tuning) -- it completes text, it does not")
    print("directly answer questions. Try continuation-style prompts, not questions.\n")
    if train_config.curriculum is not None:
        corpus_names = ", ".join(phase.name for phase in train_config.curriculum)
        print(f"Trained on a curriculum over: {corpus_names}.")
        print("Try prompts in a general-prose or math-word-problem register, e.g.:")
        print("  'The history of the Roman Empire began with' or 'Problem: John has 5 apples")
        print("  and gives away 2. Solution:'\n")
    else:
        print("Trained on TinyStories -- try story-style prompts like:")
        print("  'Once upon a time, there was a little' or 'Tom and Lily went to the'\n")

    if args.system_prompt is not None:
        print(f"System prompt active (prepended as leading context, not an instruction): {args.system_prompt!r}\n")
    if args.template != "none":
        print(f"Template active: '{args.template}' -- your input will be wrapped before generation.\n")

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

        full_prompt = build_prompt(prompt, args.template, args.system_prompt)
        input_ids = torch.tensor([encoder.encode_ordinary(full_prompt)], device=device)
        generated = model.generate(
            input_ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature
        )
        output_text = encoder.decode(generated[0].tolist())
        print(f"\n{output_text}\n")


if __name__ == "__main__":
    main()
