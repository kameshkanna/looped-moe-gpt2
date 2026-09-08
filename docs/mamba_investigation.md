# Mamba-2 Integration: Install Journey, Design, and Diagnostic Configs

**Date:** 2026-09-07
**Purpose:** Document the WSL/CUDA-kernel install path for `mamba_ssm`, the real dependency
pinning issues hit along the way (worth remembering if this environment is ever rebuilt), the
`MixerType`/`MambaConfig` design added to the model, and the three new diagnostic configs
comparing pure Mamba-2, hybrid Mamba-2+MLA, and the existing MLA-only architecture at matched
scale.

**Novelty note:** looping a Mamba-2 + attention hybrid block is *not* a novel idea — arXiv
2602.12078 already loops a Mamba-2→Mamba-2→Attention→MLP hybrid block, including the full
hybrid block (verified directly against that paper, not just its abstract). This work is
explicitly a from-scratch replication-and-extension at GPT-2 scale on this project's existing
looped-MoE-MLA codebase, in the same spirit as this project's earlier LoopMoE/MoR replication —
see `literature_survey.md`'s "Bottom line" framing, which applies equally here.

---

## 1. Why WSL, not native Windows

`mamba_ssm`'s selective-scan kernels and `causal-conv1d`'s depthwise-convolution kernels are
both CUDA/Triton extensions compiled from source at install time (`pip install ... --no-build-
isolation`). Their build scripts and issue trackers document unreliable compilation on native
Windows (MSVC toolchain incompatibilities with the CUDA extension build path that `nvcc` +
`ninja` + PyTorch's `cpp_extension` assume on Linux). WSL2 Ubuntu, with GPU passthrough, gives a
real Linux CUDA toolchain (`nvcc`, `gcc`, `ninja`) against the same physical GPU, sidestepping
this entirely. `nvidia-smi` inside WSL confirmed the RTX 4060 Laptop GPU was visible before any
of this work started.

## 2. `/mnt/c` vs. native ext4: a real, measured performance difference

The first install attempt put the venv at `/mnt/c/Users/.../looped-moe-gpt2/.venv-wsl` — on the
Windows filesystem, accessed from WSL over the 9p protocol. Unpacking `pip`'s downloaded wheels
there (torch's own wheel alone is 780MB) took long enough that the process was still running
after ~15 minutes; `ps` showed the process in `D` state (uninterruptible I/O sleep) with CPU
time barely advancing, and at one point the WSL VM itself became briefly unresponsive to new
`wsl -d Ubuntu` invocations (`Wsl/Service/0x8007274c`) during a later, unrelated CUDA-kernel
compile under similar memory pressure.

**Fix:** moved the venv to native WSL ext4 (`~/venvs/looped-moe-mamba`), leaving the *project
source* on `/mnt/c` (read once at import time, not download/unpack-heavy) but keeping the venv,
pip cache, and build tempdirs on ext4. The identical `pip install torch` command that had not
finished after 15+ minutes on `/mnt/c` completed the download+unpack in under 3 minutes on ext4.
**Lesson for any future WSL ML environment on this machine: always put the venv (and ideally
`~/.cache/pip`) on native ext4, never under `/mnt/c`.**

## 3. Package version pinning: three separate real problems, in order encountered

1. **`pip install mamba-ssm` (unpinned) resolves to a much heavier package than expected.**
   The latest release (2.3.2.post1) pulls in `tilelang`, `apache-tvm-ffi`, `quack-kernels`,
   and `triton>=3.5.0` — the last of which is incompatible with the `torch==2.5.1+cu121` +
   `triton==3.1.0` pairing already installed, so pip's resolver tried to upgrade torch itself to
   `2.14.0` with CUDA 13, a completely different (and much larger, ~10GB+ additional download)
   stack than intended. **Fix:** pin `mamba-ssm==2.2.4`, the last release before this TileLang/
   TVM rewrite — a lightweight package whose only dependencies are `torch`, `ninja`, `einops`,
   `transformers`, `packaging`, `setuptools`.

2. **Both `causal-conv1d==1.4.0` AND `mamba-ssm==2.2.4`'s PyPI sdists are missing their own
   CUDA source directories.** Confirmed on two separate machines (this project's RTX 4060 via
   WSL, and later a RunPod A100 pod) that the exact same class of bug hits both packages, not
   just one: the PyPI tarball for `causal-conv1d==1.4.0` contains only the Python wrapper files
   (`causal_conv1d_interface.py`, `causal_conv1d_varlen.py`, `__init__.py`) — no `csrc/`
   directory at all; `mamba-ssm==2.2.4`'s PyPI sdist is similarly missing
   `csrc/selective_scan/selective_scan.cpp`. Both fail identically: `pip install ... --no-
   build-isolation` gets past dependency resolution and metadata prep, then fails at the
   `ninja`/`build_ext` step with `... missing and no known rule to make it` (or `cc1plus: fatal
   error: ... No such file or directory` when a compiler invocation runs directly instead of
   through ninja) — in both cases, the package's own `setup.py` normally tries to download a
   precompiled wheel from GitHub Releases first and only falls back to a from-source build if
   that URL 404s, which is what happens whenever no prebuilt wheel matches the exact torch/CUDA/
   Python combination in use (true on both the WSL/4060 and RunPod/A100 environments tested).
   **Fix, for both packages:** install directly from the git tag instead of PyPI:
   ```
   pip install --no-build-isolation 'mamba-ssm @ git+https://github.com/state-spaces/mamba.git@v2.2.4'
   pip install --no-build-isolation 'causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d.git@v1.4.0'
   ```
   Both git tags contain the full `csrc/` sources the PyPI sdists are missing. Given this now
   confirmed pattern (two-for-two packages from the same `state-spaces`/`Dao-AILab` ecosystem),
   treat "PyPI sdist missing csrc/" as the default expectation for any package in this family
   at a version without a matching prebuilt wheel, not a one-off fluke -- go straight to the
   git-tag install rather than debugging the PyPI path first.

3. **`transformers` version skew breaks `mamba_ssm`'s own import chain, unrelated to the CUDA
   kernels.** `mamba_ssm==2.2.4`'s `utils/generation.py` imports `GreedySearchDecoderOnlyOutput`
   and `SampleDecoderOnlyOutput` from `transformers.generation` — names that existed in the
   2024-era `transformers` this package was built against but were removed by `transformers==
   5.16.1` (whatever the environment happened to resolve to). This import chain fires even
   though this project only ever calls `mamba_ssm.Mamba2` directly as a layer, never
   `MambaLMHeadModel`/its generation utilities. **Fix:** pin `transformers==4.44.2` (contemporary
   with `mamba-ssm==2.2.4`).

All three pins are recorded in `pyproject.toml`'s `mamba` extra
(`pip install -e '.[mamba]'`, WSL only).

## 4. CUDA kernel compile time and memory: what to expect

Both `causal-conv1d` and `mamba-ssm` compile `.cu` files for multiple GPU architectures
(`sm_53` through `sm_90` for the git-sourced `causal-conv1d==1.4.0` build; 4 architectures for
the PyPI `causal-conv1d==1.7.0` build used in an earlier, since-superseded attempt). Each
architecture's `cicc`/`ptxas` invocation for the heaviest kernel (`causal_conv1d_bwd.cu`) uses
roughly 1-2GB RSS; running these with high parallelism (`MAX_JOBS=16`, one per logical CPU) on
this machine's 7.4GB WSL2 VM memory limit caused genuine swap thrashing (swap usage briefly hit
1.3GB of the 2GB configured swap) and, once, an apparent WSL VM stall. **Fix:** `MAX_JOBS=2` for
all CUDA-kernel builds on this machine. This roughly doubles wall-clock build time (a `causal-
conv1d` build went from ~13 minutes at `MAX_JOBS=16` to ~28 minutes at `MAX_JOBS=2`) but keeps
memory headroom throughout — worth the trade given the alternative is an unpredictable stall or
outright OOM-kill mid-build.

## 5. A real, load-bearing dimensionality constraint

`mamba_ssm.Mamba2`'s fused Triton kernel path (`mamba_split_conv1d_scan_combined`) requires
`causal_conv1d`'s channel-last tensor strides to be multiples of 8. Concretely, this means
`(expand * d_model) // headdim` — the number of SSM heads — must be a multiple of 8. A first
smoke test with `d_model=128, expand=2, headdim=64` (the `mamba_ssm` default) gives only 4 heads
and fails with `RuntimeError: causal_conv1d with channel last layout requires strides ... to be
multiples of 8`; `d_model=256, expand=2, headdim=64` (8 heads) works. This is now enforced as an
explicit, fail-fast validation in `ModelConfig.__post_init__` (see §6) rather than surfacing as
an opaque runtime error deep in a training loop.

## 6. Model-code design: `MixerType` and `MambaConfig`

Added to `looped_moe_gpt2.model.config`:

- **`MixerType`** enum: `ATTENTION` (existing behavior, default — zero change to any existing
  config), `MAMBA2` (Mamba-2 mixer only, no attention), `HYBRID_MAMBA_ATTENTION` (both, in
  sequence within one block).
- **`MambaConfig`** dataclass: thin, validated passthrough of `mamba_ssm.Mamba2`'s own
  constructor arguments (`d_state`, `d_conv`, `expand`, `headdim`, `ngroups`).
- **`ModelConfig`** gained `mixer_type: MixerType = MixerType.ATTENTION` and
  `mamba: Optional[MambaConfig] = None`, with `__post_init__` validation covering: `mamba`
  required when a Mamba mixer is used; the multiples-of-8 constraint from §5; `MixerType.MAMBA2`
  requires `PositionEncodingType.NONE` (Mamba-2's own recurrence is position-sensitive without
  any explicit positional encoding, unlike attention); all of the pre-existing MLA/RoPE
  validation is preserved unchanged and only applies when an attention path is actually in use.
- **`PositionEncodingType.NONE`** added as a new variant (only valid with pure `MAMBA2`).

`looped_moe_gpt2.model.mamba_mixer.Mamba2Mixer` wraps `mamba_ssm.Mamba2` with the same
`forward(x) -> tensor` signature as the existing attention mixers, importing `mamba_ssm` lazily
inside `__init__` (raising a clear `ImportError` pointing at this doc) so the rest of the
package stays importable on platforms where `mamba_ssm` cannot be installed at all (e.g. native
Windows, or any CPU-only CI).

`looped_moe_gpt2.model.block.TransformerBlock` now branches on `mixer_type`:

- `ATTENTION` (default): unchanged from before this work — one pre-norm attention sub-block,
  one pre-norm feed-forward sub-block.
- `MAMBA2`: one pre-norm Mamba-2 sub-block in place of attention, then feed-forward.
- `HYBRID_MAMBA_ATTENTION`: THREE pre-norm sub-blocks in sequence — Mamba-2 first, then
  attention, then feed-forward — each with its own residual connection and, when
  `use_iter_adaln=True`, its own `IterAdaLN` instance (`norm_mamba`, `norm1`, `norm2`). Looping
  over this block means every iteration re-applies both mixers, not just one.

All of this composes with the existing looping (`LoopConfig`), MoE (`MoEConfig`), and IterAdaLN
machinery unchanged — a looped Mamba-2 block and a looped hybrid block are both just
`TransformerBlock` instances re-applied via the same `loop_plan` mechanism used throughout this
codebase.

## 7. Verification performed

- All 121 pre-existing tests still pass unchanged (`pytest -q`) — the refactor to
  `TransformerBlock.__init__`/`forward` is additive and backward-compatible.
- 7 new tests in `tests/test_mamba_mixer.py` (GPU-gated via
  `pytest.importorskip("mamba_ssm")` + `pytest.mark.skipif(not torch.cuda.is_available())`, so
  the suite stays fully runnable on CPU-only/native-Windows setups): config validation (rejects
  a non-multiple-of-8 head count, requires `mamba` config, requires
  `PositionEncodingType.NONE`), and full forward+backward shape/gradient checks for both pure
  Mamba-2 and hybrid Mamba-2+MLA through the complete looped `LoopedMoEGPT` model.
- 3 new tests in `tests/test_config_io.py` covering the new YAML fields
  (`mixer_type`/`mamba`) parsing correctly via `load_model_config` — this caught a real gap:
  `config_io.py` predated this work and did not handle either field at all; fixed alongside the
  model-code changes.
- End-to-end smoke test loading all three real `configs/08-10` files through the exact
  `load_model_config`/`load_train_config` path `scripts/train.py` uses, running one real
  forward+backward pass on GPU for each — all three produced a finite loss around
  `ln(50257) ≈ 10.82` (as expected for an untrained model with a ~50k vocabulary).

## 8. The three diagnostic configs

All three share identical `hidden_size=576, num_layers=8, num_heads=8, max_seq_len=512`,
`MoEConfig` (8 routed experts, top-2, 1 shared, `expert_intermediate_size=576`), and
`LoopConfig` (`middle_cycle`, `num_loops=4`, 1 prefix + 1 suffix layer,
`effective_depth=26`) — only the sequence mixer differs, so the comparison isolates that one
architectural axis as directly as this codebase's config surface allows.

| Config | Mixer | Params | Notes |
|---|---|---|---|
| `configs/08_pure_mamba2.yaml` | `MAMBA2` only | 93.6M | No attention, no positional encoding. `mamba.headdim=72` keeps `(2*576)/72=16` heads, a multiple of 8. |
| `configs/09_hybrid_mamba_mla.yaml` | `HYBRID_MAMBA_ATTENTION` | 97.3M | Mamba-2 then MLA, both looped together. Same `mamba` block as variant 08. |
| `configs/10_mla_baseline_comparison.yaml` | `ATTENTION` (MLA) | 80.6M | This project's pre-existing architecture (configs/03's design), rescaled to the same hidden_size/num_layers/num_loops as 08/09 so it is a fair control, not artificially inflated to match Mamba's extra SSM parameters. |

**Revision note:** an initial pass used `hidden_size=640, num_heads=10` (111.8M/115.9M/95.4M
params) and `max_steps=2850`. The first real run (`configs/08`) showed a much higher per-step
cost on this GPU than the existing MLA-only architecture at the same scale, pushing the tqdm ETA
to ~2.5 hours -- likely `mamba_ssm`'s Triton kernels being tuned for datacenter GPUs (A100/H100)
rather than a laptop 4060, combined with `compile_model: false` (no torch.compile) on the Mamba
variants. Trimmed both `hidden_size` (640->576, ~10% smaller) and `max_steps` (2850->2200,
~23% fewer) moderately -- not either axis drastically -- while keeping the architecture shape
(`num_layers`, `num_loops`, MoE expert count/top-k) identical, so all three variants stay a
fair, matched comparison at a still-meaningful diagnostic scale.

Train/data settings mirror `configs/06_deep_loop_diagnostic.yaml`'s already-validated local
budget: `data/general_train.bin` (FineWeb-Edu general pretraining corpus, no math curriculum —
this comparison is about the mixer, not reasoning capability), `batch_size=4,
gradient_accumulation_steps=8` (effective batch 32), `max_steps=2200` (~36M tokens),
`eval_interval=55`. `compile_model` is left `false` for the two Mamba variants (torch.compile
wrapping a module that itself dispatches into hand-written Triton kernels is an unnecessary
extra moving part for a first diagnostic run) and `true` for the MLA baseline (matching how that
architecture has always been run in this project).

## 8.5. Real A100 profiling: a 3-part memory diagnosis (SDPA mask, no checkpointing, Mamba-2 itself)

Profiling `configs/11` on a real RunPod A100 SXM (80GB) at `max_seq_len=2048` found peak VRAM
of ~42-54GB at `batch_size=8` alone, with `batch_size=16` already OOM'ing -- far higher than
the ~2.2GB of fixed weight/optimizer/gradient overhead this model size implies, meaning
roughly 40-50GB was activation memory for just 16,384 tokens (~3MB/token, abnormally high).
Diagnosed to three separate, additive causes:

1. **A real SDPA bug, now fixed** (see the commit "Fix SDPA attention to use is_causal=True
   instead of an explicit bool mask tensor"): both `MultiHeadLatentAttention` and
   `StandardMultiHeadAttention` called `scaled_dot_product_attention` with an explicit boolean
   `attn_mask` tensor (a plain causal mask, no padding) instead of `is_causal=True`. An
   explicit mask tensor prevents SDPA from dispatching to the Flash Attention / memory-
   efficient backend on many PyTorch versions, forcing a fallback to the "math" backend that
   materializes a full `(B, H, T, T)` score matrix. This predates the Mamba work entirely --
   every attention-using config in this project paid this cost silently. Fixed by switching to
   `is_causal=True`, which is functionally identical for this codebase (the mask was always
   plain causal, never used for padding).

2. **No gradient/activation checkpointing anywhere in the loop, now added** (see
   `ModelConfig.use_gradient_checkpointing` and `LoopedMoEGPT._run_block`): with
   `effective_depth=26` (this project's looped architectures routinely have double-digit
   effective depth) and no checkpointing, the backward pass had to hold EVERY loop iteration's
   activations simultaneously -- a 26x activation-memory multiplier that dominates at this
   scale regardless of what the per-layer cost is. Added as an opt-in `ModelConfig` field
   (`use_gradient_checkpointing: bool = False`, defaulting to unchanged behavior for every
   existing config) wrapping each block application in `torch.utils.checkpoint.checkpoint`
   with `use_reentrant=False`. Verified numerically identical to the non-checkpointed path
   (same loss, same gradients to `atol=1e-4`) via a dedicated test
   (`test_gradient_checkpointing_matches_non_checkpointed_loss_and_grads`), not just "runs
   without crashing."

3. **Mamba-2's own SSD algorithm has genuinely higher backward memory than plain attention or
   Mamba-1 at this sequence length** -- this is a documented, upstream characteristic, not a
   bug in this codebase: at `seq_len=2048`, Mamba-2's chunked-scan formulation is reported to
   use roughly 33% more memory than Mamba-1 due to block-wise state materialization, and its
   higher backward memory in general comes from storing per-chunk intermediates needed for the
   backward pass. `mamba_ssm.Mamba2`'s default `chunk_size=256` divides 2048 evenly (8 chunks),
   so this is not a misconfiguration -- it is simply a real cost of the algorithm at this
   context length, and gradient checkpointing (item 2) is the primary lever available to
   manage it, since it applies per-loop-iteration regardless of what makes any single
   iteration's activations expensive.

**Practical upshot:** `configs/11_hybrid_mamba_mla_scaleup.yaml` now sets
`use_gradient_checkpointing: true`. Re-run `scripts/profile_batch_size.py` on the actual pod
after pulling both fixes -- expect a substantially larger usable batch size, at the cost of
somewhat higher per-step wall-clock time (checkpointing recomputes each block's forward pass
during backward). The exact new numbers were not yet measured as of this writing; do not trust
the config's placeholder `max_steps` until re-profiled.

## 9. Running the three variants

One at a time (single GPU) — see the commands in the session's own record; in short:
`python scripts/train.py --config configs/08_pure_mamba2.yaml`, then `09`, then `10`, from the
WSL venv at `~/venvs/looped-moe-mamba` with the project checked out under `/mnt/c/...` as
before. Each run's `output_dir` (`runs/08_pure_mamba2`, etc.) follows this project's existing
checkpoint/log convention exactly.
