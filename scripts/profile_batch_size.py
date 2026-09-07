#!/usr/bin/env python
"""CLI: empirically find the best batch size for the current GPU, for a given model config.

Sweeps a list of candidate batch sizes, measuring real step time and peak VRAM for each (with
fused AdamW and, optionally, torch.compile -- matching the actual training loop's setup so the
numbers are directly usable, not a rough proxy). Stops sweeping once a candidate OOMs. Prints a
throughput (tokens/sec) table so you can pick the actual optimum for this hardware, rather than
reuse a batch size tuned on a different GPU.

Usage:
    python scripts/profile_batch_size.py --config configs/05_curriculum_reasoning.yaml
    python scripts/profile_batch_size.py --config configs/05_curriculum_reasoning.yaml \
        --batch-sizes 8,16,32,48,64,96,128 --no-compile

This does NOT touch any real data or write any checkpoints -- it runs forward/backward/optimizer
steps on random token ids purely to measure throughput and memory.
"""

from __future__ import annotations

import argparse
import gc
import logging
import time
from pathlib import Path

import torch

from looped_moe_gpt2.model.gpt import LoopedMoEGPT
from looped_moe_gpt2.utils.config_io import load_model_config
from looped_moe_gpt2.utils.device import resolve_amp_dtype, resolve_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def profile_one_batch_size(
    model_config,
    batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_compile: bool,
    n_warmup: int = 3,
    n_measure: int = 15,
) -> dict | None:
    """Measure step time and peak VRAM for one batch size, or return None on OOM.

    Args:
        model_config: Architecture configuration to instantiate.
        batch_size: Micro-batch size to test.
        device: Compute device (must be CUDA for meaningful VRAM measurement).
        amp_dtype: Autocast dtype.
        use_compile: Whether to wrap the model with ``torch.compile`` before measuring (adds a
            one-time compile cost per batch size tested, reported separately from steady-state
            throughput).
        n_warmup: Warmup iterations before timing (lets CUDA/compile caches settle).
        n_measure: Number of iterations to average timing over.

    Returns:
        A dict with ``batch_size``, ``ms_per_microbatch``, ``tokens_per_sec``, ``peak_vram_gb``,
        and ``compile_time_s`` (0.0 if ``use_compile`` is False); or None if this batch size
        raised ``torch.cuda.OutOfMemoryError``.
    """
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    try:
        model = LoopedMoEGPT(model_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=(device.type == "cuda"))
        model_forward = torch.compile(model) if use_compile else model

        x = torch.randint(0, model_config.vocab_size, (batch_size, model_config.max_seq_len), device=device)
        y = torch.randint(0, model_config.vocab_size, (batch_size, model_config.max_seq_len), device=device)

        compile_time_s = 0.0
        t_compile_start = time.time()
        for _ in range(n_warmup):
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                _, loss = model_forward(x, targets=y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            model.update_all_routing_biases()
        torch.cuda.synchronize()
        if use_compile:
            compile_time_s = time.time() - t_compile_start

        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        for _ in range(n_measure):
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                _, loss = model_forward(x, targets=y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            model.update_all_routing_biases()
        torch.cuda.synchronize()
        elapsed = time.time() - start

        ms_per_microbatch = elapsed / n_measure * 1000
        tokens_per_sec = (batch_size * model_config.max_seq_len) / (ms_per_microbatch / 1000)
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9

        del model, optimizer, model_forward, x, y
        return {
            "batch_size": batch_size,
            "ms_per_microbatch": ms_per_microbatch,
            "tokens_per_sec": tokens_per_sec,
            "peak_vram_gb": peak_vram_gb,
            "compile_time_s": compile_time_s,
        }
    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        torch.cuda.empty_cache()
        gc.collect()


def main() -> None:
    """Parse CLI arguments and run the batch-size sweep, printing a summary table."""
    parser = argparse.ArgumentParser(description="Find the best batch size for this GPU.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a variant YAML config.")
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="8,16,24,32,48,64,96,128",
        help="Comma-separated list of batch sizes to try, in increasing order.",
    )
    parser.add_argument("--no-compile", action="store_true", help="Skip torch.compile (faster sweep, less representative).")
    args = parser.parse_args()

    device = resolve_device()
    if device.type != "cuda":
        raise RuntimeError("This profiling script requires a CUDA GPU to produce meaningful results.")
    amp_dtype = resolve_amp_dtype(device)
    model_config = load_model_config(args.config)
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    logger.info("Profiling %s on %s (compile=%s)", args.config, torch.cuda.get_device_name(0), not args.no_compile)
    logger.info("Total VRAM: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    results = []
    for bs in batch_sizes:
        logger.info("Testing batch_size=%d...", bs)
        result = profile_one_batch_size(model_config, bs, device, amp_dtype, use_compile=not args.no_compile)
        if result is None:
            logger.info("batch_size=%d: OOM -- stopping sweep here.", bs)
            break
        results.append(result)
        logger.info(
            "batch_size=%d: %.1f ms/microbatch, %.0f tok/s, %.2f GB peak VRAM%s",
            result["batch_size"],
            result["ms_per_microbatch"],
            result["tokens_per_sec"],
            result["peak_vram_gb"],
            f", compile_time={result['compile_time_s']:.1f}s" if not args.no_compile else "",
        )

    if not results:
        print("\nNo batch size succeeded -- even the smallest candidate OOM'd. Check the config/GPU.")
        return

    best = max(results, key=lambda r: r["tokens_per_sec"])
    print(f"\n{'=' * 70}")
    print(f"{'batch_size':>10} | {'ms/microbatch':>14} | {'tokens/sec':>12} | {'peak VRAM (GB)':>15}")
    print(f"{'-' * 70}")
    for r in results:
        marker = " <-- best" if r is best else ""
        print(f"{r['batch_size']:>10} | {r['ms_per_microbatch']:>14.1f} | {r['tokens_per_sec']:>12.0f} | {r['peak_vram_gb']:>15.2f}{marker}")
    print(f"{'=' * 70}")
    print(f"\nRecommended: batch_size={best['batch_size']} ({best['tokens_per_sec']:.0f} tokens/sec)")
    print("Set this in your config's train.batch_size, and adjust gradient_accumulation_steps")
    print("if you need a specific effective batch size (batch_size * gradient_accumulation_steps).")
    print("\nNOTE: leave some VRAM headroom below the peak shown here -- real training also holds")
    print("dataloader-pinned host buffers and can see brief allocator fragmentation spikes that")
    print("this synthetic sweep (random tensors, no dataloader) does not fully replicate.")


if __name__ == "__main__":
    main()
