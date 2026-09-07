# Speedup Investigation — Request for Second Opinion

**Context:** Training a ~97.5M param looped-MoE transformer with per-token adaptive depth
(Mixture-of-Recursions-style router) on a single consumer GPU (RTX 4060 Laptop, 8GB VRAM).
Current estimate: **~1.5 days (35.6 hours) for the full planned run** (103,454 steps). Looking
for any real speedup that doesn't compromise what the run is testing. Everything tried so far is
listed below with actual measured numbers — please don't re-suggest these without a reason to
think the measurement was wrong.

---

## Hardware

- **GPU:** NVIDIA GeForce RTX 4060 Laptop, 8.6GB VRAM, CUDA 13.1 driver, bf16 tensor cores supported
- **CPU:** AMD Ryzen 7 8845HS (8 cores / 16 threads), integrated Radeon 780M iGPU, integrated NPU (XDNA)
- **RAM:** 16.4GB total
- Confirmed: iGPU and NPU cannot participate in this PyTorch training workload (no ROCm/DirectML
  training backend usable here, NPU is inference-only via ONNX/DirectML with no autograd support)

## Software stack

- PyTorch 2.14.0+cu126, `torch.compile` available with `triton-windows` 3.8.0.post28 installed
- Mixed precision via `torch.autocast(dtype=torch.bfloat16)`, master weights in fp32 (standard practice)
- Windows 11, running via PowerShell + venv (not WSL/Linux)

## Architecture being trained

```
hidden_size=1024, num_heads=16, num_layers=3 (1 prefix + 1 shared loop body + 1 suffix)
Attention: Multi-head Latent Attention (MLA, DeepSeek-V3-style) with decoupled RoPE
  kv_compression_dim=72, q_compression_dim=216, rope_head_dim=32
FFN: Sparse MoE, 8 routed experts (top-2 active) + 1 always-on shared expert,
  expert_intermediate_size=768, dense/vectorized expert computation (see "MoE vectorization" below)
Loop: SharingPattern.FULL_LOOP, num_loops=3 -- the SAME shared block (MLA+MoE) is called
  3 times per forward pass, sandwiched between one unique prefix and one unique suffix layer.
Router: per-token adaptive-depth gating (Mixture-of-Recursions-style) -- see "Router design" below.
Total params: ~97.5M. Effective depth: 5 (1 + 3 + 1).
```

## Training config being used

- batch_size=8, gradient_accumulation_steps=4 (effective batch 32 sequences x 512 tokens = 16,384 tokens/step)
- 103,454 total optimizer steps (~1.695B tokens total, ~13.6 tokens/param -- deliberately below
  Chinchilla-optimal ~20x, reusing an already-downloaded token budget rather than fetching more)
- AdamW, cosine LR schedule, standard warmup

## MoE implementation detail ("BatchedExperts")

Originally implemented as a Python `for expert_idx in range(num_experts)` loop with
boolean-mask gather per expert (`x_flat[token_mask]`). This was found to bottleneck HARD on
CPU/kernel-launch overhead: measured ~697ms/microbatch with GPU utilization pinned near 0-30%
despite "100% GPU utilization" readouts, because a `.any()` check inside the loop forced a
GPU->CPU sync every iteration, serializing what should be independent kernel launches. **Fixed**
by rewriting to a fully dense/vectorized form: every expert computes on every token via a single
batched `einsum` (`(E,H,I)` and `(E,I,H)` stacked weight tensors), then a scatter-built dense
per-token-per-expert weight matrix combines results with one more `einsum`. This cut the same
model's step time roughly in half (~697ms -> ~353ms/microbatch at the time, smaller model).
Wasted FLOPs on non-selected experts are accepted as a worthwhile tradeoff at this model scale.

## Router implementation detail ("AdaptiveDepthRouter")

Masked-dense design: every token is computed by the shared block at EVERY loop iteration
regardless of exit status (keeps tensor shapes static across iterations -- deliberately avoids
dynamic-shape recompilation problems, informed by the MoE experience above); only the
WRITE-BACK of each iteration's hidden-state update is masked by a per-token active/inactive
flag. Per-iteration router: linear `hidden_size -> 1` score, tokens ranked and the current
active set's top-`capacity_ratio` fraction kept for the next iteration, rest frozen. Originally
had the SAME per-batch-element `.item()`-in-a-loop anti-pattern as the pre-fix MoE code (to
handle each sequence having a different, shrinking "keep count" -- PyTorch's `topk` needs one
static k per call). **Fixed** by replacing with a fully vectorized rank-via-argsort-and-scatter
approach (no host-device sync in any loop). Measured impact of that fix: modest, ~254ms ->
~246ms/microbatch on an earlier/smaller test config (~3% improvement) -- the router was never
the dominant cost, just avoidable extra overhead.

**Measured overhead of having the router at all** (compared at the SAME num_loops=3, same
everything else, only router on vs. off): **303.7ms vs 298.8ms/microbatch -- ~1.6% difference.**
The router is not the bottleneck.

## Profiling results (current 97.5M config, bs=8, 5-iteration profile via `torch.profiler`)

CPU and CUDA self-time are roughly balanced this time (~1.48s CPU / ~1.47s CUDA total across 5
iterations) -- unlike the earlier small-model case where CPU dominated by ~4.4x. Top entries by
CUDA time: `aten::bmm` (22%), `aten::mm` (21%), `aten::linear` (wraps mm, not separately
additive), `aten::copy_` (16%, 1680 calls across 5 iters = ~336/iter), `aten::einsum` (16%,
150 calls = 30/iter, this is the MoE `BatchedExperts` computation), `aten::_to_copy` (was going
to investigate further -- see "Open question" below), `Optimizer.step#AdamW.step` (11%).

The `aten::copy_`/`aten::_to_copy` call counts (~300+/iteration) seemed disproportionate for a
model with only 31 `nn.Linear` modules and 5 block-applications per forward pass, so this was
investigated as a possible bug -- see below.

## Levers tried, with measured results

| Lever | Result | Verdict |
|---|---|---|
| Fix MoE Python-loop -> vectorized `einsum` | ~697ms -> ~353ms/microbatch (smaller model, earlier in the project) | Real fix, kept |
| Fix router `.item()`-in-loop -> vectorized argsort | ~254ms -> ~246ms/microbatch (smaller test config) | Real fix, kept |
| Router on vs. off (current 97.5M config) | 303.7ms vs 298.8ms/microbatch | Router overhead is negligible (~1.6%), not worth removing |
| Larger batch size: bs=8 -> 12 -> 16 | 13312 -> 13217 -> 3141 tok/s (bs=16 exceeds 8.6GB VRAM, effectively thrashing) | No gain from bs=8->12 (flat throughput); bs=16 is actively worse. bs=8 is the sweet spot on this card. |
| TF32 matmul mode (`torch.backends.cuda.matmul.allow_tf32=True`) | 306.4ms vs ~304-309ms baseline | No measurable effect (expected -- pipeline already runs under bf16 autocast, TF32 only affects fp32 matmul paths, which aren't used here) |
| `num_loops=3` vs `num_loops=2` (fewer loop iterations) | 303.7ms vs 264.7ms/microbatch (~13% faster) | Real speedup, but REJECTED -- reduces the router's adaptive-depth range (1-2 passes instead of 1-3), which weakens the exact mechanism this run exists to test |
| `num_routed_experts=8` vs `4` (halve MoE experts) | ~303.7ms vs ~242.4ms/microbatch (~20% faster) | Real speedup, but REJECTED -- moves away from the Kimi-K2-style "many fine-grained experts" design this project is specifically replicating |
| `torch.compile()` (default mode) | NOT tried on the current 97.5M router-gated config specifically. Tried previously on the pre-vectorization `SparseMoE` code: 91.7s compile overhead, then ~685ms/microbatch -- WORSE than eager mode (~500ms) at the time, attributed to dynamic control flow (the old Python per-expert loop) causing graph breaks/recompiles. | **Not re-tested since the MoE and router vectorization fixes** -- this is flagged below as the most promising untried option, since the dynamic-control-flow problem that broke it before may no longer apply now that both hot loops are vectorized. |
| NPU (AMD XDNA) | Not usable -- no PyTorch autograd/training backend exists for this NPU; it's an ONNX/DirectML inference-only accelerator (Windows Studio Effects / Copilot+ features), no backward-pass support. | Ruled out, not a viable path on this hardware/software combo as far as we know. |
| AMD Radeon 780M iGPU | Not attempted -- would need ROCm or DirectML backend for PyTorch training on Windows for an integrated GPU, which is not a mature/available path today, and even if it worked the iGPU is weaker than the discrete 4060 for this workload (shared system RAM bandwidth, far fewer compute units). | Ruled out as not worth pursuing. |
| Splitting compute across NVIDIA dGPU + AMD iGPU simultaneously | Not attempted -- CUDA and ROCm/DirectML are different, non-interoperable backends; a single model's autograd graph can't trivially span both without real model-parallel infrastructure, which doesn't exist as an off-the-shelf tool for this heterogeneous-vendor case. | Ruled out. |

## Open question / not fully investigated

The `aten::copy_` / `aten::_to_copy` call volume (~300+/iteration) was flagged as
disproportionately high and worth investigating further, but the investigation was cut short
(profiler API attribute error when trying to trace call stacks: `FunctionEventAvg` object has no
`self_cuda_time_total` attribute in this PyTorch version, needed `self_cpu_time_total` or a
different aggregation method instead). **This was not resolved** -- it's possible there is a
real, fixable source of excess copy/cast traffic somewhere in the model that hasn't been found
yet, given the CPU/CUDA time split here (1.48s/1.47s) is much more balanced than earlier smaller
models profiled in this project (where CPU dominated ~4.4x due to the pre-fix MoE/router loops).
Whether ~300 copies/iteration is actually excessive for a model with 5 block-applications x
(5 MLA linears + MoE dense einsum + shared-expert linears) per forward, or whether this is just
normal autocast fp32-master-weight-to-bf16 casting cost multiplied by every parameter tensor
touched, was not conclusively determined.

## Question for review

1. Is `torch.compile()` worth retrying now that both the MoE and router hot paths are fully
   vectorized (no more Python-level per-expert or per-batch-element loops with host-device
   syncs)? Any known gotchas with `torch.compile` + `FULL_LOOP` sharing (the same
   `nn.Module` instance called multiple times in one forward pass with a different
   `iteration_idx` each time) or with the router's `argsort`/`scatter_`-based masking?
2. Is there a real, fixable source in this architecture for the `aten::copy_`/`_to_copy`
   overhead noted above, or is ~300 copy-ops/iteration actually normal/expected for a model
   with fp32 master weights + bf16 autocast + this many distinct Linear/Parameter tensors
   touched per forward pass?
3. Any other Windows-specific or consumer-GPU-specific PyTorch training speedups not covered
   above (e.g. `torch.backends.cudnn.benchmark`, memory format tricks, fused optimizer
   variants) that would apply to this specific bottleneck profile (compute-bound at this
   width, not launch-overhead-bound)?
