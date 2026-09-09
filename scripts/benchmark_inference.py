#!/usr/bin/env python
"""CLI: measure real autoregressive generation throughput/latency/memory for a checkpoint.

Establishes an honest baseline BEFORE any inference optimization work -- runs the model's
actual, current ``LoopedMoEGPT.generate()`` exactly as a real caller would use it (no
shortcuts, no synthetic proxies), timing wall-clock generation of a fixed number of new tokens
from a fixed-length prompt, at a range of prompt lengths (to see how cost scales with context,
which matters most for a no-KV-cache baseline where every step recomputes the full prefix).

Usage:
    python scripts/benchmark_inference.py --checkpoint runs/09_hybrid_mamba_mla/checkpoint_step2200.pt \
        --config configs/09_hybrid_mamba_mla.yaml
    python scripts/benchmark_inference.py --checkpoint ... --config ... \
        --prompt-lengths 16,64,128,256 --max-new-tokens 128
"""

from __future__ import annotations

import argparse
import gc
import logging
import time
from pathlib import Path

import torch

from looped_moe_gpt2.model.gpt import LoopedMoEGPT
from looped_moe_gpt2.train.checkpoint import load_checkpoint
from looped_moe_gpt2.utils.config_io import load_model_config
from looped_moe_gpt2.utils.device import resolve_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def benchmark_one_prompt_length(
    model: LoopedMoEGPT,
    device: torch.device,
    prompt_len: int,
    max_new_tokens: int,
    n_warmup: int,
    n_measure: int,
) -> dict[str, float]:
    """Measure generation wall-clock time and peak VRAM at one fixed prompt length.

    Args:
        model: The model to benchmark, already loaded and in eval mode.
        device: Compute device.
        prompt_len: Number of prompt tokens fed to ``generate()`` before any new tokens.
        max_new_tokens: Number of new tokens to generate per call.
        n_warmup: Warmup generation calls before timing (settles CUDA/allocator state).
        n_measure: Number of timed generation calls to average over.

    Returns:
        A dict with ``prompt_len``, ``mean_seconds``, ``tokens_per_sec``, and ``peak_vram_gb``.
    """
    torch.cuda.empty_cache() if device.type == "cuda" else None
    gc.collect()

    prompt = torch.randint(0, model.config.vocab_size, (1, prompt_len), device=device)

    for _ in range(n_warmup):
        model.generate(prompt, max_new_tokens=max_new_tokens, temperature=1.0)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(n_measure):
        model.generate(prompt, max_new_tokens=max_new_tokens, temperature=1.0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    mean_seconds = elapsed / n_measure
    tokens_per_sec = max_new_tokens / mean_seconds
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0

    return {
        "prompt_len": prompt_len,
        "mean_seconds": mean_seconds,
        "tokens_per_sec": tokens_per_sec,
        "peak_vram_gb": peak_vram_gb,
    }


def main() -> None:
    """Parse CLI arguments and run the inference benchmark, printing a summary table."""
    parser = argparse.ArgumentParser(description="Benchmark real autoregressive generation throughput.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a local .pt checkpoint.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the checkpoint's variant YAML config.")
    parser.add_argument(
        "--prompt-lengths",
        type=str,
        default="16,64,128,256",
        help="Comma-separated prompt lengths to benchmark, in increasing order.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128, help="New tokens generated per call.")
    parser.add_argument("--n-warmup", type=int, default=2, help="Warmup generation calls before timing.")
    parser.add_argument("--n-measure", type=int, default=5, help="Timed generation calls to average over.")
    args = parser.parse_args()

    device = resolve_device()
    model_config = load_model_config(args.config)
    model = LoopedMoEGPT(model_config).to(device)

    checkpoint = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Loaded checkpoint from step %d", checkpoint["step"])
    logger.info(
        "Model: mixer_type=%s, %d params, effective_depth=%d",
        model_config.mixer_type.value,
        sum(p.numel() for p in model.parameters()),
        model_config.effective_depth,
    )

    prompt_lengths = [int(p) for p in args.prompt_lengths.split(",")]
    results = []
    for prompt_len in prompt_lengths:
        if prompt_len + args.max_new_tokens > model_config.max_seq_len:
            logger.info(
                "Skipping prompt_len=%d (prompt_len + max_new_tokens exceeds max_seq_len=%d).",
                prompt_len,
                model_config.max_seq_len,
            )
            continue
        logger.info("Benchmarking prompt_len=%d, max_new_tokens=%d...", prompt_len, args.max_new_tokens)
        result = benchmark_one_prompt_length(
            model, device, prompt_len, args.max_new_tokens, args.n_warmup, args.n_measure
        )
        results.append(result)
        logger.info(
            "prompt_len=%d: %.3fs total, %.1f tok/s, %.2f GB peak VRAM",
            result["prompt_len"],
            result["mean_seconds"],
            result["tokens_per_sec"],
            result["peak_vram_gb"],
        )

    print(f"\n{'=' * 70}")
    print(f"{'prompt_len':>10} | {'mean seconds':>13} | {'tokens/sec':>11} | {'peak VRAM (GB)':>15}")
    print(f"{'-' * 70}")
    for r in results:
        print(f"{r['prompt_len']:>10} | {r['mean_seconds']:>13.3f} | {r['tokens_per_sec']:>11.1f} | {r['peak_vram_gb']:>15.2f}")
    print(f"{'=' * 70}")

    if len(results) >= 2:
        first, last = results[0], results[-1]
        slowdown = first["tokens_per_sec"] / last["tokens_per_sec"]
        print(
            f"\nThroughput dropped {slowdown:.2f}x from prompt_len={first['prompt_len']} to "
            f"prompt_len={last['prompt_len']} -- with no KV cache, generation cost grows with "
            f"prompt length even though the number of NEW tokens generated is identical each "
            f"time; a properly cached implementation should show much flatter scaling here."
        )


if __name__ == "__main__":
    main()
