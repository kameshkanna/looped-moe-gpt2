# looped-moe-gpt2

A GPT-2-scale looped (recurrent-depth) transformer combining Kimi-K2-style sparse Mixture-of-Experts
and Multi-head Latent Attention (MLA) with per-token adaptive loop depth
(Mixture-of-Recursions-style routing). See `docs/literature_survey.md` for the full research
grounding and `docs/adaptive_depth_router_design.md` for the adaptive-depth router design.

## What this is

- **Looped transformer**: a shared transformer block is re-applied multiple times per forward
  pass (`LoopConfig`), following Universal Transformer / Huginn / Nanbeige4.2-style recurrent depth.
- **Sparse MoE**: DeepSeek-V3/Kimi-K2-style routed experts + 1 always-on shared expert, with
  auxiliary-loss-free load balancing (a per-expert bias, not an aux loss term).
- **MLA**: DeepSeek-V3-style low-rank KV/Q compression with decoupled RoPE.
- **Adaptive depth router**: a per-token, per-iteration router (Mixture-of-Recursions-style)
  decides how many of the loop's passes each token actually needs, rather than a fixed depth
  for every token.
- **Curriculum training**: multi-corpus, step-scheduled data mixing (e.g. general text early,
  reasoning-dense data ramped in partway through) — see `CurriculumPhase` in
  `src/looped_moe_gpt2/train/config.py`.

Validated at ~85M params on TinyStories (see `docs/literature_survey.md` §4 for the ablation
results: looped+MoE beat matched-compute dense baselines; looping alone did not). The
`configs/05_curriculum_reasoning.yaml` run (~97.5M params, router-gated adaptive depth, general
English + math-reasoning curriculum) is the current larger-scale experiment.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# or: .venv\Scripts\Activate.ps1  # Windows PowerShell

pip install -e .
```

Requires Python >=3.10, PyTorch >=2.2 (CUDA build for GPU training — see
[pytorch.org](https://pytorch.org/get-started/locally/) for the correct install command for your
CUDA version). `torch.compile` (used by default in `configs/05_curriculum_reasoning.yaml`) needs
Triton; on Windows this means `pip install triton-windows` in addition to the base install (not
needed on Linux, where Triton ships as part of the standard PyTorch install).

## Data preparation

This repo does not ship tokenized data (see `.gitignore` — `data/` is excluded, GB-scale files
don't belong in git). Regenerate it:

```bash
# General English (FineWeb-Edu) + math-reasoning (OpenMathInstruct-2) for the curriculum run:
python scripts/prepare_curriculum_data.py --phase general --target-tokens 1200000000 \
    --output data/general_train.bin --val-tokens 5000000 --val-output data/general_val.bin
python scripts/prepare_curriculum_data.py --phase math --target-tokens 500000000 \
    --output data/math_train.bin --val-tokens 5000000 --val-output data/math_val.bin

# TinyStories, for the smaller ablation configs (00-04):
python scripts/prepare_data.py --train-text <path> --val-text <path> --output-dir data/
```

Both scripts stream from HuggingFace `datasets` (or download directly for TinyStories) and stop
once the requested token budget is hit — they do not download the full source dataset.

## Training

```bash
python scripts/train.py --config configs/05_curriculum_reasoning.yaml
```

Resume an interrupted run:

```bash
python scripts/train.py --config configs/05_curriculum_reasoning.yaml \
    --resume runs/05_curriculum_reasoning/checkpoint_stepN.pt
```

Checkpoints, logs, and per-phase validation loss are written to the config's `output_dir`.

## Evaluation & interaction

```bash
# Perplexity + architecture diagnostics (expert utilization, loop-iteration cosine similarity):
python scripts/evaluate.py --checkpoint runs/<variant>/checkpoint_stepN.pt \
    --config configs/<variant>.yaml --val-bin data/<val>.bin

# Interactive text generation:
python scripts/chat.py --checkpoint runs/<variant>/checkpoint_stepN.pt --config configs/<variant>.yaml
```

## Running on a cloud GPU (e.g. RunPod)

The codebase is platform-agnostic (device resolution in `src/looped_moe_gpt2/utils/device.py`
picks CUDA/MPS/CPU automatically), but a few things are Windows-specific to this project's
development environment and need adjusting for a Linux cloud pod:

1. **No `triton-windows` needed** — on Linux, `pip install -e .` plus a standard CUDA PyTorch
   install already includes a working Triton for `torch.compile`.
2. **Batch size / VRAM**: configs in this repo were sized for an 8GB RTX 4060 laptop GPU
   (`batch_size: 8` in most configs). An H100 (80GB) can go much larger — raising `batch_size`
   (and proportionally lowering `gradient_accumulation_steps` to keep the same effective batch
   size, or just increasing the effective batch size outright) will use the extra VRAM and
   likely improve throughput further, since this architecture was found to be compute-bound
   rather than launch-overhead-bound at 8GB-card batch sizes — re-profile on the actual target
   GPU before assuming the same batch size is still optimal.
3. **Data**: re-run the `scripts/prepare_curriculum_data.py` commands above on the pod (or
   upload the tokenized `.bin` files directly) — they are not in this git repo.
4. **`torch.compile` mode**: `default` mode was found to be as fast as `max-autotune` on the
   RTX 4060 for this architecture (see `docs/speedup_investigation_for_review.md`) — worth
   re-checking on H100, since a different GPU architecture may see a different result from
   `max-autotune`'s more exhaustive kernel search.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

## Project structure

```
src/looped_moe_gpt2/
  model/       # architecture: config, attention (MLA), MoE, router, blocks, full GPT
  data/        # tokenization, memory-mapped dataset, sampling
  train/       # training loop, checkpointing, LR/bias schedules, curriculum, diagnostics
  utils/       # device resolution, seeding, YAML config loading
configs/       # one YAML per experiment variant
docs/          # literature survey, design docs, investigation notes
scripts/       # CLI entry points (train, evaluate, chat, data prep)
tests/         # pytest suite
```
