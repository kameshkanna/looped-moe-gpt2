# Design: Per-Token Adaptive Loop Depth (Test-Time Scaling Knob)

**Status: step 2 (masked-dense router) implemented and tested** —
`src/looped_moe_gpt2/model/router.py` (`AdaptiveDepthRouter`),
`src/looped_moe_gpt2/model/config.py` (`RouterConfig`, `LoopConfig.use_router_gating`), wired
into `LoopedMoEGPT._forward_router_gated_loop`. 15 tests in `tests/test_router.py`, including a
regression test for the MoE+router combination this doc originally flagged as unexplored in the
literature. Verified: per-token exit-depth genuinely varies (not a degenerate fixed-depth
collapse), `min_loops` floor is enforced, exited tokens never reactivate, router receives real
gradients, and it composes correctly with `SparseMoE`.

**Current limitation, by design (see step 1 vs step 2 below):** this is the masked-dense
variant — every token is computed by the shared block at every iteration regardless of exit
status, so **there is no real inference FLOPs/latency saving yet**, only the training-time
adaptive-depth *signal* (which tokens the router considers "hard"). Steps 3 (KV caching for
`MultiHeadLatentAttention`, not yet implemented at all) and 4 (gather/scatter-dense conversion)
from this doc's original recommended order are still open, and are the actual prerequisite for
turning this into a real inference-time test-time-compute knob rather than a research/training
diagnostic. Currently requires `SharingPattern.FULL_LOOP` (Middle-Cycle + router-gating
interaction remains an open question, as originally noted below).

## Empirical finding: the router learns confidence-gating, not difficulty-gating (and structurally cannot do otherwise under `capacity_ratio`)

**Setup:** `configs/06_deep_loop_diagnostic.yaml`, ~75.7M params, `num_loops=12`,
`capacity_ratio=0.7`, `min_loops=1`, trained from scratch on ~46.7M tokens of FineWeb-Edu (2850
steps, ~1h16m on an RTX 4060). This run exists specifically because an earlier check on the
`num_loops=3` curriculum checkpoint (`configs/05_curriculum_reasoning.yaml`) was inconclusive —
with only 2 non-degenerate exit points after `min_loops=1`, there wasn't enough dynamic range to
tell a real signal from noise. `num_loops=12` gives 11 non-degenerate exit points.

**Method:** for each of several checkpoints spanning the run, computed per-token cross-entropy
loss (as an objective difficulty proxy — higher loss means the model finds that token harder to
predict) against `last_exit_iteration` (which loop iteration each token actually stopped
updating at) over the same 16-sequence, 512-token held-out validation batch (8,192 tokens),
correlating the two.

**Result:**

| step | correlation (loss vs. exit depth) | mean exit depth |
|---|---|---|
| 71 | -0.041 | 3.29 |
| 355 | -0.170 | 3.29 |
| 710 | -0.202 | 3.29 |
| 1420 | -0.206 | 3.29 |
| 2130 | -0.210 | 3.29 |
| 2850 (final) | -0.209 | 3.29 |

Per-depth-bucket mean loss at the final checkpoint is a clean, near-monotonic decrease from
depth 1 (loss 6.13, hardest) to depth 11 (loss 2.98, easiest) — this is not noisy scatter, it is
a consistent trend across the full depth range.

**Two findings, not one:**

1. **The router learned the OPPOSITE of the intended "harder tokens get more compute" behavior.**
   The correlation is negative and stable: tokens the model finds EASY (low loss) are the ones
   that get routed DEEPER, not tokens it finds hard. This pattern is not present at
   initialization (correlation ≈ 0 at step 71) — it emerges within the first ~25% of training and
   then holds essentially constant for the rest of the run. A plausible mechanism: a token the
   model is already confident/correct about may genuinely benefit from further refinement passes
   through the shared block (there is "useful work" left to do that improves an already-good
   prediction), while a token the model is fundamentally uncertain about (rare word, genuinely
   ambiguous continuation) may not be resolved by more passes of the SAME shared computation —
   so a loss-minimizing router learns to stop investing in those tokens rather than "try harder."
   This is closer to how confidence-based early-exit classifiers are usually framed in other
   literature than to MoR's "adaptive reasoning depth" framing this project set out to test.

2. **The mean exit depth (3.29) is IDENTICAL across every checkpoint checked, to two decimal
   places** — this is not because training stabilized on that number, it is a direct, forced
   consequence of the masked-dense router's own mechanism (see `router.py`): at every iteration,
   the top `capacity_ratio` FRACTION of currently-active tokens survive, regardless of what the
   router has or hasn't learned. With `capacity_ratio=0.7`, the survivor fraction after `n`
   iterations past `min_loops` is deterministically `0.7^n` (100% -> 70% -> 49% -> 34.3% -> ...),
   which sets the exit-depth DISTRIBUTION's shape independent of training. **What the router
   actually learns is only WHICH tokens (by relative score) occupy that fixed-size survivor
   pool at each cutoff — never HOW MANY survive, and never the total compute budget.** This is
   an architectural ceiling, not a training artifact: as designed, this router cannot express
   "this problem is harder, so spend more total depth on it than that other problem" -- it can
   only reallocate a fixed depth budget across tokens within one input. A design that wanted
   genuine total-compute scaling with difficulty would need a different mechanism (e.g. a
   per-SEQUENCE rather than per-token capacity signal, or removing the fixed-ratio cutoff in
   favor of a learned per-token halting probability threshold, closer to the original Universal
   Transformer ACT mechanism this whole lineage descends from).

**Implication for `docs/literature_survey.md`'s framing**: the "does adaptive depth help
reasoning" question this router was built to test is still open, but this finding narrows it —
before asking whether the LEARNED routing helps, note that the CAPACITY-RATIO mechanism itself
bounds what kind of adaptivity is even representable. Any future ablation of this router should
control for `capacity_ratio`'s fixed-shape effect explicitly, and a next design iteration should
consider whether a per-token halting-probability mechanism (no fixed survivor fraction) would
let genuine difficulty-driven depth-scaling emerge, if it exists to be learned at this scale at
all.

## What this adds, precisely

Today, `LoopConfig.num_loops` is a fixed integer baked into the model at construction time
(`LoopedMoEGPT._build_blocks`) — every token in every sequence gets exactly `num_loops` passes
through the shared block. This design adds a **learned per-token router** that decides, at each
loop iteration, whether a given token continues looping or exits early — following
Mixture-of-Recursions' expert-choice routing (arXiv:2507.10524), adapted to this repo's
Middle-Cycle sharing pattern.

Two capabilities fall out of this:
1. **Adaptive computation during training/inference**: "easy" tokens (e.g. common function
   words) exit after 1 pass; "hard" tokens (rare words, tokens requiring more context
   integration) get the full `max_loops` passes — without any manual intervention.
2. **A genuine test-time compute knob**: a global `min_loops`/`max_loops` bound can be tightened
   or loosened at inference time to trade latency for quality, per MoR's framing.

## Why this is NOT a small change

Two separable pieces of new machinery, both required:

### 1. The router itself

- One router per loop iteration (MoR uses per-step routers, not one shared router for all
  steps) — a small linear layer `hidden_size -> 1` producing a per-token continue/exit score.
- **Expert-choice routing** (MoR's better-performing variant per their own ablation, 42.6% vs
  40.0% few-shot accuracy at Nr=3): at each iteration, rank all *still-active* tokens by router
  score and keep only the top-`capacity_ratio` fraction active for the next iteration; the rest
  exit and their current hidden state is carried forward unchanged to the final output.
- Router training signal: MoR trains the router jointly with the LM loss using a
  differentiable relaxation (e.g. the router's continue-probability multiplicatively gates the
  block's contribution during training, so gradients flow through the routing decision even
  though inference uses a hard top-k cutoff). This needs care — a naive hard top-k router has no
  gradient; the standard fix is a sigmoid-gated soft version during training that anneals toward
  hard behavior, or an auxiliary loss encouraging the soft and hard decisions to agree.

### 2. Variable-depth batched computation

This is the harder engineering problem, structurally similar to what `SparseMoE` already does
along the *expert* axis, but now needed along the *depth* axis:

- At iteration `t`, only a subset of tokens in the batch are still "active" (haven't exited).
  The block's forward pass must run ONLY on those tokens, not the full `(batch, seq_len,
  hidden)` tensor — otherwise there's no compute savings and the whole point is lost.
- Two implementation strategies, with a real tradeoff:
  - **Masked-dense** (simpler, matches this repo's existing "vectorize over Python loops"
    lesson from `SparseMoE`'s fix): run the block on the FULL tensor every iteration regardless
    of exit status, then mask which tokens' updates actually get written back to the running
    hidden state. Wastes compute on exited tokens (same tradeoff `BatchedExperts` makes
    deliberately) but keeps tensor shapes static, which is far friendlier to CUDA graphs,
    `torch.compile`, and avoids the dynamic-shape recompilation issues we hit  when testing
    `torch.compile` on the current `SparseMoE`. Recommended starting point.
  - **Gather/scatter-dense** (MoR's actual approach, real compute savings): gather only active
    tokens into a smaller dense tensor before each block call, scatter results back after. Saves
    real FLOPs as tokens exit, but reintroduces the dynamic-shape-per-iteration problem that
    made this repo's own diagnostics code fragile (see the forward-hook aliasing bug fixed
    earlier this session) — needs careful handling if reused with the existing
    `attach_loop_diagnostics` hooks, which currently assume every looped block call corresponds
    to a fixed, known iteration index for the FULL batch.
- **KV-cache implications for inference**: once tokens exit at different depths, the attention
  KV cache per layer is no longer uniform across tokens either — MoR found "recursive KV
  sharing" (caching only at the first recursion step) hurts expert-choice routing specifically
  more than token-choice. This repo's current `MultiHeadLatentAttention` doesn't implement KV
  caching at all yet (each forward call recomputes attention over the full provided sequence),
  so adaptive depth would need KV-cache support added as a prerequisite for any real inference
  latency win — without it, this feature only demonstrates the mechanism, it doesn't actually
  save wall-clock time at inference.

## Recommended implementation order (once phase 1 results justify starting)

1. **Global manual knob first** (cheap, no router): make `num_loops` a `forward()`-time
   parameter instead of construction-time. `IterAdaLN`'s embedding table is already sized by
   `max_iterations`, so this is close to a pure plumbing change in `LoopedMoEGPT.forward` and
   `_build_blocks`'s `loop_plan` generation (build the plan for `max_loops`, slice it at call
   time). Validates that variable loop count doesn't break anything before adding routing
   complexity on top.
2. **Masked-dense router**, training-time only, measuring whether per-token adaptive depth
   improves the loss/compute tradeoff over fixed depth AT ALL (the actual scientific question,
   independent of whether it saves real inference FLOPs yet).
3. **KV caching for `MultiHeadLatentAttention`** — needed regardless of adaptive depth, for any
   realistic inference deployment; do this before chasing real inference-time savings.
4. **Gather/scatter-dense** conversion, once masked-dense has validated the router is worth it
   and KV caching exists to make the latency win real and measurable.

## Config surface (already partially present)

`LoopConfig.use_router_gating: bool` already exists in `src/looped_moe_gpt2/model/config.py` as
a forward-looking flag (currently unused — `_build_blocks` never checks it). A new
`RouterConfig` dataclass would be needed alongside it:

```python
@dataclass(frozen=True)
class RouterConfig:
    capacity_ratio: float = 0.5  # fraction of tokens kept active per iteration
    min_loops: int = 1           # every token gets at least this many passes
    aux_loss_weight: float = 0.01  # weight on the soft/hard routing-agreement auxiliary loss
```

## Open questions to revisit after phase 1

- Does MoR's expert-choice routing (rank tokens, hard top-k cutoff) or token-choice (each token
  commits to a full depth upfront) fit this repo's Middle-Cycle sharing pattern better? MoR
  tested both on a simple fully-tied loop; interaction with Middle-Cycle's "distinct-per-pass,
  tied-across-iterations" structure is untested anywhere in the literature surveyed so far.
- How does the router interact with `SparseMoE`'s own routing? Two independent routing
  decisions (which experts, how many loops) compound — MoR's own paper flags MoE integration as
  unsolved future work, so this repo would be genuinely novel territory here, not a replication.
