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
