# RunPod H100 Setup — Step by Step

**Pod spec used:** RunPod PyTorch 2.8.0 template, 1x H100 SXM, 80GB VRAM, 125GB RAM, 8 vCPU,
CUDA 12.8/13.0/13.2 available.

## 1. Clone and set up the environment

```bash
git clone https://github.com/kameshkanna/looped-moe-gpt2.git
cd looped-moe-gpt2
bash scripts/setup_runpod.sh
```

This creates a venv (inheriting the base image's pre-validated torch/CUDA install via
`--system-site-packages`, so it doesn't re-download torch from scratch), installs this package,
verifies CUDA + `torch.compile` both work, and runs the test suite. If anything fails here, fix
it before proceeding — don't debug environment issues mid-training-run.

## 2. Prepare data

```bash
source .venv/bin/activate

python scripts/prepare_curriculum_data.py --phase general --target-tokens 1200000000 \
    --output data/general_train.bin --val-tokens 5000000 --val-output data/general_val.bin
python scripts/prepare_curriculum_data.py --phase math --target-tokens 500000000 \
    --output data/math_train.bin --val-tokens 5000000 --val-output data/math_val.bin
```

This streams from HuggingFace `datasets` and stops once the token budget is hit — it does not
download the full source datasets. Expect this to take a while on the pod's network (was
~427k tokens/sec on the original dev machine's connection; RunPod's network may differ).

## 3. Find the right batch size for THIS GPU — don't reuse the 4060's numbers

The configs in this repo (`batch_size: 8`) were tuned for an 8GB RTX 4060 laptop GPU. An H100
has 80GB — 10x the VRAM — and profiling on the 4060 found this architecture to be **compute-bound
at small batch sizes, not launch-overhead-bound**, meaning a bigger, more powerful GPU can
genuinely push more batch size AND more raw throughput, not just "run the same work faster."
Guessing a number here wastes GPU-hours either by OOMing mid-run or by leaving throughput on
the table — measure it instead:

```bash
python scripts/profile_batch_size.py --config configs/05_curriculum_reasoning.yaml \
    --batch-sizes 8,16,32,48,64,96,128,192,256
```

This runs the exact same architecture through real forward/backward/optimizer steps at each
batch size (with `torch.compile` + fused AdamW, matching real training), stopping automatically
at the first OOM, and prints a tokens/sec table. Takes a few minutes per batch size tested
(each does its own `torch.compile` warmup) — expect this step to take 20-40 minutes total for
the full sweep above, which is a worthwhile one-time cost against a multi-hour training run.

**Then update `configs/05_curriculum_reasoning.yaml`:**
- Set `train.batch_size` to the recommended value.
- Adjust `train.gradient_accumulation_steps` to preserve (or intentionally change) the effective
  batch size (`batch_size * gradient_accumulation_steps` = sequences per optimizer step). Going
  from `batch_size=8, grad_accum=4` (effective 32) to, say, `batch_size=64, grad_accum=1`
  (effective 64) doubles the effective batch size — this changes training dynamics (larger
  batches generally tolerate/want a higher learning rate), so if you increase the *effective*
  batch size substantially, consider scaling `learning_rate` up correspondingly (a common
  starting heuristic is linear LR scaling with batch size, though this is a heuristic, not a
  guarantee — watch the loss curve after changing it).
- **Recompute `train.max_steps`**: this repo's configs are sized by total TOKEN budget, not
  step count. If you increase batch size, you need FEWER steps to consume the same
  1.695B-token budget:
  ```
  new_max_steps = 1_695_000_000 // (new_batch_size * new_gradient_accumulation_steps * 512)
  ```
  Also rescale `warmup_steps`, `eval_interval`, and `checkpoint_interval` proportionally (they
  were originally set as ~1%, ~2.5%, and ~2.5% of `max_steps` respectively) — and rescale the
  curriculum's `start_step`/`ramp_steps` in `train.curriculum` the same way, since those are
  also expressed as raw step counts tied to the original `max_steps=103454`.

## 4. Launch training

```bash
python scripts/train.py --config configs/05_curriculum_reasoning.yaml
```

Expect a multi-minute `torch.compile` warmup on the first step (longer than the 4060's ~149s,
since the H100 will be compiling different/larger kernel configurations) — this is normal, not
a hang. Watch `nvidia-smi` in another terminal to confirm GPU utilization climbs once past the
first step.

To resume after any interruption:
```bash
python scripts/train.py --config configs/05_curriculum_reasoning.yaml \
    --resume runs/05_curriculum_reasoning/checkpoint_stepN.pt
```

## 5. Don't forget to stop the pod

RunPod bills by the hour while the pod is running, regardless of whether training is active.
Stop or terminate the pod as soon as the run completes (or you're done for the session) —
checkpoints are written to the pod's disk (`runs/`), so download them (or push to a storage
bucket) before terminating, since pod storage is not guaranteed to persist after termination
depending on your RunPod volume configuration.
