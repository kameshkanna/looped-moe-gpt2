#!/usr/bin/env python
"""CLI: probe the Mamba-vs-attention inference crossover point at long context, on THIS codebase.

The literature (see docs/mamba_investigation.md) reports a crossover around 8K-16K context
tokens where SSM/Mamba per-step decode cost overtakes attention's growing (even KV-cached)
per-step cost. This script tests whether that crossover is visible on this project's specific
hybrid architecture, at a range of prompt lengths reaching into that regime, using an
UNTRAINED model of each mixer type at matched dimensions -- valid because this measures pure
compute/memory throughput (a function of shapes and kernels, not learned weights), avoiding the
cost of training multiple long-context checkpoints just to answer a timing question.

Usage:
    python scripts/benchmark_long_context.py --prompt-lengths 512,2048,4096,8192 --max-new-tokens 16
"""

from __future__ import annotations

import argparse
import gc
import logging
import time

import torch

from looped_moe_gpt2.model.config import (
    AttentionConfig,
    LoopConfig,
    MambaConfig,
    MixerType,
    ModelConfig,
    MoEConfig,
    PositionEncodingType,
    SharingPattern,
)
from looped_moe_gpt2.model.gpt import LoopedMoEGPT
from looped_moe_gpt2.utils.device import resolve_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_HIDDEN_SIZE = 576
_NUM_HEADS = 8
_NUM_LAYERS = 8
_NUM_LOOPS = 4


def _build_config(mixer_type: MixerType, max_seq_len: int) -> ModelConfig:
    """Construct a ModelConfig matching configs/08/09/10's dimensions, at a given max_seq_len.

    Args:
        mixer_type: Which sequence mixer to test.
        max_seq_len: Maximum sequence length (must cover the longest prompt_len to be tested).

    Returns:
        A ModelConfig with matched hidden_size/num_layers/num_loops/MoE across all mixer types,
        so only the mixer itself differs -- same design as the original 08/09/10 comparison.
    """
    common = dict(
        vocab_size=50257,
        hidden_size=_HIDDEN_SIZE,
        num_layers=_NUM_LAYERS,
        num_heads=_NUM_HEADS,
        max_seq_len=max_seq_len,
        moe=MoEConfig(enabled=True, num_routed_experts=8, num_shared_experts=1, top_k=2, expert_intermediate_size=_HIDDEN_SIZE),
        loop=LoopConfig(enabled=True, sharing_pattern=SharingPattern.MIDDLE_CYCLE, num_loops=_NUM_LOOPS, num_unique_prefix_layers=1, num_unique_suffix_layers=1),
    )
    mamba_cfg = MambaConfig(d_state=64, d_conv=4, expand=2, headdim=72, ngroups=1)
    if mixer_type == MixerType.MAMBA2:
        return ModelConfig(**common, position_encoding=PositionEncodingType.NONE, attention=AttentionConfig(use_mla=False), mixer_type=mixer_type, mamba=mamba_cfg)
    if mixer_type == MixerType.HYBRID_MAMBA_ATTENTION:
        return ModelConfig(**common, position_encoding=PositionEncodingType.DECOUPLED_ROPE, attention=AttentionConfig(use_mla=True, kv_compression_dim=64, q_compression_dim=160, rope_head_dim=32), mixer_type=mixer_type, mamba=mamba_cfg)
    return ModelConfig(**common, position_encoding=PositionEncodingType.DECOUPLED_ROPE, attention=AttentionConfig(use_mla=True, kv_compression_dim=64, q_compression_dim=160, rope_head_dim=32), mixer_type=MixerType.ATTENTION)


def benchmark(mixer_type: MixerType, prompt_len: int, max_new_tokens: int, device: torch.device) -> dict[str, float]:
    """Measure generation throughput for one (mixer_type, prompt_len) pair on an untrained model.

    Args:
        mixer_type: Which sequence mixer to test.
        prompt_len: Prompt length to generate from.
        max_new_tokens: Number of new tokens to generate.
        device: Compute device.

    Returns:
        A dict with ``tokens_per_sec`` and ``peak_vram_gb``, or ``{"oom": True}`` on OOM.
    """
    torch.cuda.empty_cache()
    gc.collect()
    max_seq_len = prompt_len + max_new_tokens
    config = _build_config(mixer_type, max_seq_len)
    try:
        model = LoopedMoEGPT(config).to(device)
        model.eval()
        prompt = torch.randint(0, config.vocab_size, (1, prompt_len), device=device)

        with torch.no_grad():
            model.generate(prompt, max_new_tokens=max_new_tokens, temperature=1.0)  # warmup
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        with torch.no_grad():
            model.generate(prompt, max_new_tokens=max_new_tokens, temperature=1.0)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        tokens_per_sec = max_new_tokens / elapsed
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
        del model
        return {"tokens_per_sec": tokens_per_sec, "peak_vram_gb": peak_vram_gb}
    except torch.cuda.OutOfMemoryError:
        return {"oom": True}
    finally:
        torch.cuda.empty_cache()
        gc.collect()


def main() -> None:
    """Parse CLI arguments and run the crossover-point sweep across mixer types and prompt lengths."""
    parser = argparse.ArgumentParser(description="Probe the Mamba-vs-attention inference crossover point.")
    parser.add_argument("--prompt-lengths", type=str, default="512,2048,4096,8192")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    device = resolve_device()
    if device.type != "cuda":
        raise RuntimeError("This benchmark requires a CUDA GPU.")
    prompt_lengths = [int(p) for p in args.prompt_lengths.split(",")]
    mixer_types = [MixerType.ATTENTION, MixerType.MAMBA2, MixerType.HYBRID_MAMBA_ATTENTION]

    results: dict[MixerType, list[dict]] = {m: [] for m in mixer_types}
    for prompt_len in prompt_lengths:
        for mixer_type in mixer_types:
            logger.info("Testing mixer=%s, prompt_len=%d...", mixer_type.value, prompt_len)
            result = benchmark(mixer_type, prompt_len, args.max_new_tokens, device)
            results[mixer_type].append({"prompt_len": prompt_len, **result})
            if result.get("oom"):
                logger.info("  OOM.")
            else:
                logger.info("  %.1f tok/s, %.2f GB peak VRAM", result["tokens_per_sec"], result["peak_vram_gb"])

    print(f"\n{'=' * 90}")
    print(f"{'prompt_len':>10} | {'ATTENTION tok/s':>17} | {'MAMBA2 tok/s':>14} | {'HYBRID tok/s':>14}")
    print(f"{'-' * 90}")
    for i, prompt_len in enumerate(prompt_lengths):
        row = [f"{prompt_len:>10}"]
        for mixer_type in mixer_types:
            r = results[mixer_type][i]
            row.append(f"{'OOM':>17}" if r.get("oom") else f"{r['tokens_per_sec']:>17.1f}" if mixer_type == MixerType.ATTENTION else f"{r['tokens_per_sec']:>14.1f}")
        print(" | ".join(row))
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
