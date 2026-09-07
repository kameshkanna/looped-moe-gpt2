#!/usr/bin/env python
"""CLI: evaluate answer-accuracy on held-out GSM8K math word problems.

Unlike perplexity/val_loss (which measures next-token-prediction quality and can be low simply
because a corpus has templated, easy-to-predict surface structure -- see this project's own
observation that OpenMathInstruct-2's val_loss dropped unusually low, unusually fast), this
script measures the thing that actually matters for "can this model do math": does its generated
solution's final numeric answer match the ground truth?

Uses GSM8K's official test split (never trained on -- distinct from the training corpus,
OpenMathInstruct-2, which is itself partly *derived* from GSM8K's training problems via
augmentation, but the test split is held out by construction), extracting the '#### <number>'
ground-truth format and the model's own final number from its generated text.

Usage:
    python scripts/eval_reasoning_accuracy.py --checkpoint runs/05_curriculum_reasoning/checkpoint_step51727.pt \
        --config configs/05_curriculum_reasoning.yaml --num-problems 200
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import tiktoken
import torch
from datasets import load_dataset
from tqdm import tqdm

from looped_moe_gpt2.model.gpt import LoopedMoEGPT
from looped_moe_gpt2.train.checkpoint import load_checkpoint
from looped_moe_gpt2.utils.config_io import load_model_config
from looped_moe_gpt2.utils.device import resolve_device
from looped_moe_gpt2.utils.hub import resolve_checkpoint_and_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_GSM8K_ANSWER_PATTERN = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_GENERIC_NUMBER_PATTERN = re.compile(r"-?[\d,]+(?:\.\d+)?")


def extract_gsm8k_ground_truth(answer_field: str) -> float | None:
    """Extract the ground-truth final number from a GSM8K 'answer' field.

    Args:
        answer_field: The raw ``answer`` text, e.g. "...#### 72".

    Returns:
        The ground-truth number as a float, or None if the expected ``####`` marker isn't found
        (should not happen on the official dataset, but fail soft rather than crash a long eval
        run over one malformed record).
    """
    match = _GSM8K_ANSWER_PATTERN.search(answer_field)
    if match is None:
        return None
    return float(match.group(1).replace(",", ""))


def extract_model_answer(generated_text: str) -> float | None:
    """Extract the model's predicted final number from its generated solution text.

    This model was trained on OpenMathInstruct-2 (see ``scripts/prepare_curriculum_data.py``'s
    ``_iter_math_texts``), which formats solutions as free text WITHOUT a GSM8K-style '####'
    marker -- so this looks for that marker first (in case the model learned to imitate it from
    GSM8K-derived content within OpenMathInstruct-2), and falls back to the LAST number
    mentioned anywhere in the generated text, which is a common heuristic for extracting a
    "final answer" from free-form generated math solutions when no explicit marker is present.

    Args:
        generated_text: The model's full generated continuation (prompt + completion).

    Returns:
        The extracted number as a float, or None if no number was found anywhere in the text.
    """
    marked_match = _GSM8K_ANSWER_PATTERN.search(generated_text)
    if marked_match is not None:
        return float(marked_match.group(1).replace(",", ""))

    all_numbers = _GENERIC_NUMBER_PATTERN.findall(generated_text)
    if not all_numbers:
        return None
    return float(all_numbers[-1].replace(",", ""))


def main() -> None:
    """Parse CLI arguments, run generation over held-out GSM8K problems, and report accuracy."""
    parser = argparse.ArgumentParser(description="Evaluate GSM8K answer accuracy.")
    parser.add_argument(
        "--hub-repo",
        type=str,
        default=None,
        help="Load checkpoint.pt + config.yaml from this Hugging Face Hub repo id "
        "(e.g. 'Kameshr/looped-moe-gpt2-reasoning') instead of local --checkpoint/--config paths.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a local .pt checkpoint.")
    parser.add_argument("--config", type=Path, default=None, help="Path to the variant's local YAML config.")
    parser.add_argument("--num-problems", type=int, default=200, help="Number of GSM8K test problems to evaluate.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate per problem.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (low, for more deterministic math generations).")
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
    logger.info("Loaded checkpoint from step %d", checkpoint["step"])

    encoder = tiktoken.get_encoding("gpt2")

    logger.info("Loading GSM8K test split...")
    gsm8k_test = load_dataset("openai/gsm8k", "main", split="test")
    num_problems = min(args.num_problems, len(gsm8k_test))
    logger.info("Evaluating on %d problems (of %d available)", num_problems, len(gsm8k_test))

    correct = 0
    no_answer_extracted = 0
    results = []

    for i in tqdm(range(num_problems), desc="Evaluating"):
        example = gsm8k_test[i]
        ground_truth = extract_gsm8k_ground_truth(example["answer"])
        if ground_truth is None:
            continue

        prompt_text = f"Problem: {example['question']}\nSolution:"
        input_ids = torch.tensor([encoder.encode_ordinary(prompt_text)], device=device)
        if input_ids.shape[1] >= model_config.max_seq_len:
            logger.warning("Problem %d's prompt exceeds max_seq_len, skipping.", i)
            continue

        generated = model.generate(input_ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        generated_text = encoder.decode(generated[0, input_ids.shape[1]:].tolist())

        predicted = extract_model_answer(generated_text)
        if predicted is None:
            no_answer_extracted += 1
            is_correct = False
        else:
            is_correct = abs(predicted - ground_truth) < 1e-4

        if is_correct:
            correct += 1

        results.append({
            "index": i,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "correct": is_correct,
            "generated_text": generated_text[:200],
        })

    accuracy = correct / num_problems if num_problems > 0 else 0.0
    print(f"\n{'=' * 60}")
    print(f"GSM8K Answer Accuracy: {accuracy*100:.1f}% ({correct}/{num_problems})")
    print(f"No answer extracted (empty/malformed generation): {no_answer_extracted}/{num_problems}")
    print(f"{'=' * 60}")
    print("\nFor reference: random guessing on GSM8K (free-form numeric answers) is ~0%.")
    print("A model that has NOT learned to reason, but fits the corpus surface structure well,")
    print("typically scores in the low single digits here despite low perplexity on math text --")
    print("this is the actual signal for whether training produced real reasoning ability.")

    print("\nSample generations (first 5):")
    for r in results[:5]:
        status = "CORRECT" if r["correct"] else "WRONG"
        print(f"\n[{status}] ground_truth={r['ground_truth']}, predicted={r['predicted']}")
        print(f"  Generated: {r['generated_text']}...")


if __name__ == "__main__":
    main()
