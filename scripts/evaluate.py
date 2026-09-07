#!/usr/bin/env python
"""CLI: compute held-out perplexity (and supporting diagnostics) for a trained checkpoint.

Usage:
    python scripts/evaluate.py --checkpoint runs/03_looped_moe_full/checkpoint_step5000.pt \
        --config configs/03_looped_moe_full.yaml --val-bin data/val.bin

Perplexity is exp(mean cross-entropy loss over the held-out set) -- the standard LM evaluation
metric, directly comparable ACROSS VARIANTS ONLY WHEN they share a tokenizer and evaluation
split (true here: all configs/*.yaml use the same GPT-2 BPE vocab and the same data/val.bin).
It is NOT comparable across different tokenizers or different vocab sizes.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from looped_moe_gpt2.data.dataset import MemmapTokenDataset
from looped_moe_gpt2.model.gpt import LoopedMoEGPT
from looped_moe_gpt2.train.checkpoint import load_checkpoint
from looped_moe_gpt2.train.diagnostics import attach_loop_diagnostics, compute_active_ratio
from looped_moe_gpt2.utils.config_io import load_model_config
from looped_moe_gpt2.utils.device import resolve_amp_dtype, resolve_device
from looped_moe_gpt2.utils.hub import resolve_checkpoint_and_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_perplexity(
    model: LoopedMoEGPT,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_batches: int | None = None,
) -> tuple[float, float]:
    """Compute mean cross-entropy loss and perplexity over a DataLoader.

    Args:
        model: The (already .eval()'d) model to evaluate.
        loader: DataLoader yielding ``(input_ids, targets)`` batches.
        device: Device to run evaluation on.
        amp_dtype: Autocast dtype (matches training precision for a fair comparison).
        max_batches: If set, stop after this many batches (for a quick estimate on a huge val
            set); otherwise evaluate the full loader once.

    Returns:
        A tuple ``(mean_loss, perplexity)``. Perplexity is ``exp(mean_loss)``, computed in
        float64 to avoid overflow if loss is unexpectedly large (e.g. an undertrained model).
    """
    total_loss = 0.0
    total_batches = 0
    progress_total = min(max_batches, len(loader)) if max_batches is not None else len(loader)
    for i, (input_ids, targets) in enumerate(tqdm(loader, desc="Evaluating", total=progress_total)):
        if max_batches is not None and i >= max_batches:
            break
        input_ids, targets = input_ids.to(device), targets.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            _, loss = model(input_ids, targets=targets)
        total_loss += loss.item()
        total_batches += 1

    mean_loss = total_loss / max(total_batches, 1)
    perplexity = math.exp(min(mean_loss, 700))  # guard against OverflowError on a broken model
    return mean_loss, perplexity


def main() -> None:
    """Parse CLI arguments, load the checkpoint, and report perplexity + architecture diagnostics."""
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint's perplexity.")
    parser.add_argument(
        "--hub-repo",
        type=str,
        default=None,
        help="Load checkpoint.pt + config.yaml from this Hugging Face Hub repo id "
        "(e.g. 'Kameshr/looped-moe-gpt2-reasoning') instead of local --checkpoint/--config paths. "
        "--val-bin must still be a local file regardless (tokenized data isn't hosted on the Hub repo).",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a local .pt checkpoint.")
    parser.add_argument("--config", type=Path, default=None, help="Path to the variant's local YAML config.")
    parser.add_argument("--val-bin", type=Path, required=True, help="Path to the tokenized validation .bin file.")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--max-batches", type=int, default=None, help="Cap on number of batches (default: full val set).")
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
    amp_dtype = resolve_amp_dtype(device)
    model_config = load_model_config(config_path)
    model = LoopedMoEGPT(model_config).to(device)

    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Loaded checkpoint from step %d", checkpoint["step"])

    dataset = MemmapTokenDataset(args.val_bin, seq_len=model_config.max_seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    logger.info("Evaluating on %d samples (%s)", len(dataset), args.val_bin)

    mean_loss, perplexity = compute_perplexity(model, loader, device, amp_dtype, args.max_batches)
    print(f"\n{'=' * 50}")
    print(f"Checkpoint:       {checkpoint_path}")
    print(f"Config:           {config_path}")
    print(f"Mean cross-entropy loss: {mean_loss:.4f}")
    print(f"Perplexity:              {perplexity:.2f}")
    print(f"{'=' * 50}\n")

    if model_config.loop.enabled:
        diag = attach_loop_diagnostics(model)
        sample_input = next(iter(loader))[0][:4].to(device)
        diag.reset()
        with torch.no_grad():
            model(sample_input)
        try:
            sims = diag.consecutive_iteration_cosine_similarities()
            print(f"Cosine similarity across loop iterations: {[f'{s:.4f}' for s in sims]}")
        except ValueError:
            pass

    if model_config.moe.enabled:
        with torch.no_grad():
            model(sample_input)
        loads = model.expert_load_summary()
        ideal = model_config.moe.top_k / model_config.moe.num_routed_experts
        print(f"\nExpert utilization (ideal uniform = {ideal:.3f} each):")
        for block_idx, load in loads.items():
            print(f"  Block {block_idx}: {load.cpu().numpy().round(3)}")
        print(f"\nActive attention:FFN parameter ratio: {compute_active_ratio(model):.4f}")


if __name__ == "__main__":
    main()
