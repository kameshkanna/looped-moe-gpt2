# Literature Survey: Looped (Recurrent-Depth) Transformers × Mixture-of-Experts

**Date:** 2026-09-06
**Purpose:** Ground the design of a GPT-2-scale "looped MoE" model — sharing a transformer
block across recursive passes, with a Kimi-K2-style sparse MoE FFN and Multi-head Latent
Attention (MLA) inside the shared block — in what has actually been published, so the
experiment isolates a genuinely open question rather than re-deriving a known result or a
known failure mode.

---

## 1. Timeline / lineage

| Year | Work | Core idea |
|---|---|---|
| 2018 | Universal Transformer (Dehghani et al.) | Share one transformer layer/block across depth, with a per-step halting mechanism (ACT) and timestep + position signal added at each step. First "looped transformer for language" formulation. |
| 2024-10 | Relaxed Recursive Transformers (Bae et al., Google DeepMind, [2410.20672](https://arxiv.org/abs/2410.20672)) | Convert an *existing* pretrained LLM into a looped model by tying layers into a shared block, then recover the lost capacity with **per-iteration low-rank (LoRA) adapters** rather than fully independent layers. Shows uptraining recovers most of full-model quality at a fraction of unique params. |
| 2025-07 | Huginn-3.5B / "depth-recurrent" (Geiping et al., [2507.02199](https://arxiv.org/abs/2507.02199) and related) | Explicit **Prelude → Loop → Coda** structure: non-shared input layers, a shared block unrolled a variable number of times at *test time* (latent "pondering"), non-shared output layers. Reasoning gains scale with more unrolls without new params. Motivates using recurrence depth as a test-time compute knob, analogous to chain-of-thought but in latent space. |
| 2025-07 | **Mixture-of-Recursions (MoR)**, NeurIPS 2025 ([2507.10524](https://arxiv.org/abs/2507.10524), [code](https://github.com/raymin0223/mixture_of_recursions)) | Adds a **lightweight per-token router** on top of a shared recursive block: router decides, per token per recursion step, whether that token continues looping (expert-choice) or how many total loops it commits to upfront (token-choice). Tested at 135M/360M/730M/1.7B — i.e. exactly GPT-2 scale and below. Found "Middle-Cycle" sharing (keep first/last layer unique, cycle the middle) beats fully-tied; expert-choice routing (42.6%) beat token-choice (40.0%) at Nr=3; recursive KV-sharing hurts expert-choice more than token-choice. Explicitly lists MoE/sparsity integration as **future work**, not implemented. |
| 2025 (rumor) | GPT-6 "Astra" (The Information, Sept 1 2025 report, unconfirmed) | Reported to use a "constrained recurrent depth" — reuses the same layer stack more than once per forward pass. No official architecture disclosure; treat as directional signal, not a spec. Raised interpretability concerns since looped latent computation isn't naturally chain-of-thought-legible. |
| 2025 (open) | Nanbeige4.2-3B ([2607.22083](https://arxiv.org/abs/2607.22083)) | Concrete open example matching the Astra rumor: 22-layer decoder stack run **twice** (`num_loops=2`), pretrained from scratch on 28T tokens. Reports ~75% of the token-efficiency of an equivalent non-looped model at ~2× the FLOPs (compute cost, not param cost). This is the empirical "two-loop, single deep block" data point your original question was implicitly comparing against. |
| 2024-12 | DeepSeek-V3 ([2412.19437](https://arxiv.org/abs/2412.19437)) | Introduces the two components Kimi K2 reuses: **Multi-head Latent Attention (MLA)** and **auxiliary-loss-free load balancing** for MoE (see §3). |
| 2025-07 | **Kimi K2** ([2507.20534](https://arxiv.org/abs/2507.20534)) | 1.04T-param / 32B-active MoE using MLA + 384 fine-grained routed experts (8 active) + 1 always-on shared expert. No looping — this is the "MoE" half of your combination, not the "looped" half. |
| 2025-06 | **LoopMoE** ([2606.04438](https://arxiv.org/abs/2606.04438)) | **The closest existing work to what you're proposing.** Explicitly combines a looped/shared block with MoE FFN + MLA attention. See §2 — this should be your primary reference architecture. |
| 2025-06 | **Tying the Loop** ([2606.16825](https://arxiv.org/abs/2606.16825)) | A *partial*-loop alternative: only tie the MoE FFN expert weights across a small group of consecutive layers; keep attention, router, and norms independent per layer. See §2. |
| 2025-07 | Loopie ("Loop the Loopies!", [2607.16051](https://huggingface.co/papers/2607.16051)) | Another looped-MoE variant, reported to beat larger vanilla models at equal compute on reasoning benchmarks (IMO/IPhO-style). Less architectural detail surfaced than LoopMoE; worth a follow-up read before final design lock, not blocking. |
| 2024 | MoEUT ([2405.16039](https://arxiv.org/abs/2405.16039), referenced via LoopMoE citations) | Earlier "Universal Transformer + MoE" combination — worth citing as prior art predating both 2025 papers above. |

**Bottom line:** the specific thing you asked for — loop the block, put Kimi-style MoE+MLA
inside it — is not a hypothetical. LoopMoE (June 2025) and Tying the Loop (June 2025) already
ran this experiment, from two different design philosophies (full loop with conditioning vs.
partial loop of FFN-only). Your GPT-2-scale run is a **replication-and-extension** at a scale
neither paper tested (both went 3B+), which is a legitimate and useful thing to do — but the
framing should be "does the LoopMoE/Tying-the-Loop finding hold at 125M," not "novel idea,"
and the writeup should cite both.

---

## 2. The core problem this survey identifies: naive loop+MoE breaks in two specific ways

LoopMoE's own ablation motivation section is the most load-bearing citation for your design,
because it explains *why* you can't just wrap a Kimi MoE block in a loop and expect it to work:

1. **Representational collapse under weight sharing** (same failure mode discussed for the
   single-layer-loop case previously): tying weights across iterations imposes a structural
   symmetry that, without an explicit per-iteration signal, gives the model no way to make
   iteration *t* compute something different from iteration *t−1*. LoopMoE's fix is
   **IterAdaLN** — iteration-conditioned adaptive layer norm, generating affine (scale/shift)
   parameters from a combination of the iteration index and the per-token hidden state, applied
   at every pre-norm site inside the loop. This is the same mechanism family recommended in the
   prior turn of this conversation (AdaLN conditioned on loop index) — the survey confirms it's
   the standard fix, not a guess.

2. **Active-parameter-ratio skew specific to MoE-in-a-loop**: attention parameters are reused
   identically every iteration, but *token-to-expert routing can change every iteration* (a
   token may hit a different top-k expert subset each pass). This means the effective
   attention-FLOPs-to-FFN-FLOPs ratio (`ρ`) drifts away from the ratio a non-looped model would
   have been tuned at, because `A_ffn` (active FFN capacity actually touched across iterations)
   grows sublinearly with the number of loops `K` (later iterations increasingly re-hit
   already-activated experts) while `A_attn` stays flat. LoopMoE's fix is an explicit
   **balancing** term/adjustment to restore the intended ratio; their ablation shows this step
   specifically recovers knowledge-benchmark performance (MMLU) that the base loop + IterAdaLN
   alone had regressed.

Both fixes are cheap to implement and should be included in the GPT-2-scale build from the
start rather than discovered the hard way in an ablation — skipping them would just reproduce
LoopMoE's *own* "Loop Base" ablation row (their weakest variant), not test anything new.

---

## 3. Component specs to reimplement at small scale

### 3.1 Multi-head Latent Attention (MLA) — from DeepSeek-V3 / reused by Kimi K2

DeepSeek-V3's published dimensions (7168 hidden, 128 heads) don't map directly to GPT-2-small
(768 hidden, 12 heads) — scale all compressed dims proportionally, keeping DeepSeek-V3's
**ratios**, not its absolute numbers:

| Quantity | DeepSeek-V3 | Ratio to hidden dim (7168) | GPT-2-small equivalent (hidden=768) |
|---|---|---|---|
| KV compression dim `d_c` | 512 | ×0.0715 | ≈ 64 (round to 64) |
| Q compression dim `d_c'` | 1536 | ×0.214 | ≈ 160 (round to 160 or 192) |
| Per-head dim `d_h` | 128 | — | keep GPT-2's 64 (768/12) or widen heads |
| Decoupled RoPE dim `d_R^h` | (subset of `d_h`, typically half) | — | ≈ 32 (half of 64) |

Mechanism (per head, per DeepSeek-V3 / Kimi K2):
1. Down-project the residual stream `h_t` once to a shared low-rank KV latent `c_t^{KV} ∈ R^{d_c}`.
2. Up-project `c_t^{KV}` per-head to get the "NoPE" (position-agnostic) K/V content.
3. Separately compute a small decoupled RoPE-carrying key/query slice of dim `d_R^h` directly
   from `h_t` (not through the compressed latent), and apply standard RoPE only to this slice.
4. Concatenate NoPE + RoPE-carrying parts per head for the attention score.
5. Do the analogous low-rank down/up projection for queries via `d_c'`.
6. **Only the compressed latents (`c_t^{KV}`, plus the small RoPE slice) need to be cached at
   inference** — this is MLA's actual payoff (KV-cache size), which matters less at GPT-2 scale
   / short context but is free to implement correctly since you're building from scratch.

This directly answers "should we add RoPE" from the earlier discussion: **yes, but decoupled
RoPE on a small slice per head, not full RoPE on the whole head dim** — that's the specific
form Kimi/DeepSeek use, and it composes cleanly with the loop (RoPE still keys off *sequence*
position; it's orthogonal to iteration index, which is handled by IterAdaLN instead, per LoopMoE).

### 3.2 MoE routing — Kimi K2 / DeepSeek-V3 style, scaled down

Kimi K2: 384 routed experts, top-8 active + 1 always-on shared expert, per token, per layer.
At GPT-2-small scale this ratio is absurd (384 experts for a 768-dim model is over-parameterized
and won't train meaningfully on a 4060) — scale **expert count and top-k down**, keep the
**mechanism**:

- Routed experts: 8–16 (small enough to actually get gradient signal per expert at this scale)
- Active per token: top-2 (standard small-MoE choice; matches Mixtral/OLMoE-style baselines
  more than Kimi's top-8, which assumes hundreds of experts)
- Shared expert: 1, always active (cheap, and directly what Kimi/DeepSeek use — keep it, since
  it's exactly the "stable base representation" mechanism that should help offset the
  representational-collapse risk from looping, per LoopMoE's own concern in §2.1)
- Expert intermediate size: keep roughly GPT-2's own MLP ratio (4× hidden) *per expert*, not
  4× the full model — i.e. each expert is a small FFN, not a full GPT-2 MLP, or the model won't
  fit on a 4060 with 8+ experts resident.

**Load balancing: use DeepSeek-V3's auxiliary-loss-free scheme, not an aux loss.**
Formula (exact, from DeepSeek-V3 technical report):
- Routing score per expert `i`: `s_i = h^T e_i` (`e_i` = learned expert centroid vector)
- Top-k selection uses the **biased** score: `TopK({s_i + b_i}, k)` — `b_i` is a per-expert
  scalar bias, *not* used when computing the combination weights (only affects which experts
  are picked, not how their outputs are weighted)
- After each step (or batch), update: if expert `i` was overloaded this step, `b_i -= γ`; if
  underloaded, `b_i += γ`. DeepSeek-V3 used `γ=0.001` for most of training, decayed to `0` near
  the end. At GPT-2/4060 scale, treat `γ` as a tunable hyperparameter to sweep (start at 0.001,
  disable — set to 0 — for the last ~10% of steps).
- This avoids the auxiliary-loss gradient-interference problem that plain load-balancing losses
  introduce, and is simple to implement (no extra loss term, just a bias vector updated outside
  the autograd graph).

### 3.3 Looping mechanics — synthesizing MoR + LoopMoE + Tying the Loop

Three concrete design points, each with a paper backing it, that should be **ablation axes**
in your experiment rather than pre-decided:

1. **Full loop vs. partial loop.** LoopMoE loops the *entire* block (MLA + MoE together, with
   IterAdaLN). Tying the Loop only ties the MoE FFN expert weights, keeping attention/router
   independent per nominal layer position — and found specifically that *untying attention*
   (i.e. keeping it per-layer, not shared) mattered far more than untying the router
   (loss cost ~0.043 vs ~0 respectively). This is a direct, citable data point suggesting: if
   you must choose one thing to keep un-shared for capacity, make it attention, not routing.
   Worth running both configurations at GPT-2-scale as your primary ablation.
2. **Block size within the loop.** MoR's "Middle-Cycle" result (keep first/last layer unique,
   cycle the middle stack) beat fully-tied and other tying strategies at 135M–1.7B — directly
   relevant precedent for your GPT-2-small (~125M) target. Use Middle-Cycle as the default
   sharing pattern, not naive "tie everything."
3. **Fixed loop count vs. router-gated (MoR-style) exit.** Start with **fixed** loop count
   (e.g. sweep K ∈ {1, 2, 3, 4}, matching Nanbeige's empirically-best K=2 as a prior) since it's
   simpler to get right on limited compute; add MoR's per-token expert-choice router as a
   stretch goal once the fixed-K + MoE + MLA baseline trains correctly. Don't build router
   complexity and MoE complexity and loop-conditioning complexity all at once on a first pass —
   MoR itself treats MoE integration as unsolved future work, so there's no existing recipe to
   copy for router+MoE together; isolate that as its own follow-up ablation rather than a launch
   requirement.

---

## 4. Recommended experiment design (ties directly to code in this repo)

**Baseline:** standard GPT-2-small (125M, 12 layers, 768 hidden, 12 heads, learned/absolute
pos-emb as in original GPT-2, or swap to RoPE — see below) trained on a fixed small corpus.

**Variants (each isolates one axis above):**
1. `dense_rope` — GPT-2 baseline but with standard full RoPE instead of learned position
   embeddings (isolates the RoPE swap by itself; cheap, do this first as a sanity check).
2. `looped_dense` — same dense FFN/attention, no MoE, block shared with Middle-Cycle pattern,
   fixed K loops, IterAdaLN conditioning. (Replicates Huginn/Nanbeige-style looping alone.)
3. `looped_moe_full` — LoopMoE-style: entire block (MLA + MoE) shared across K loops +
   IterAdaLN + the active-ratio balancing correction.
4. `looped_moe_partial` — Tying-the-Loop-style: only MoE expert weights tied across the K
   virtual passes; MLA attention and router stay independent per pass.
5. (stretch) `looped_moe_router` — add MoR-style expert-choice per-token router for early exit
   on top of variant 3.

Hold **total training FLOPs** (not wall-clock, not step count) roughly constant across variants
when comparing loss curves — looping inherently trades params for repeated compute, so an
unmatched-FLOPs comparison will just show "more compute wins," which isn't the question.

**Metrics:** validation loss/perplexity at matched FLOPs; representational-collapse diagnostic
(cosine similarity of hidden states across consecutive loop iterations, per the earlier
discussion — directly checks whether IterAdaLN is actually preventing the fixed-point collapse
LoopMoE warns about); expert utilization histogram (checks the load-balancing bias is working);
active-ratio `ρ` tracked over iterations (replicates LoopMoE's own diagnostic).

---

## 5. Primary sources (for citation)

- Universal Transformer — Dehghani et al., 2018 — https://arxiv.org/abs/1807.03819
- Relaxed Recursive Transformers — Bae et al., 2024 — https://arxiv.org/abs/2410.20672
- Depth-recurrent / Huginn — Geiping et al., 2025 — https://arxiv.org/abs/2507.02199
- Mixture-of-Recursions (MoR), NeurIPS 2025 — https://arxiv.org/abs/2507.10524 · https://github.com/raymin0223/mixture_of_recursions
- Nanbeige4.2-3B — https://arxiv.org/abs/2607.22083
- DeepSeek-V3 Technical Report — https://arxiv.org/abs/2412.19437
- Kimi K2 Technical Report — https://arxiv.org/abs/2507.20534
- LoopMoE — https://arxiv.org/abs/2606.04438
- Tying the Loop — https://arxiv.org/abs/2606.16825
- Loopie ("Loop the Loopies!") — https://huggingface.co/papers/2607.16051
- MoEUT — https://arxiv.org/abs/2405.16039
- GPT-6 "Astra" recurrent-depth reporting (unconfirmed, secondary) — The Information (Sept 1
  2025), summarized in Sebastian Raschka's writeup: https://sebastianraschka.com/blog/2026/openai-astra-looped-transformers.html
