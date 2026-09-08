"""Centralized configuration dataclasses for all model variants.

Every architectural axis identified in ``docs/literature_survey.md`` (dense vs. MoE FFN, tied
vs. untied attention, fixed-depth vs. router-gated looping, RoPE vs. learned positions) is
expressed as an explicit, typed, validated field here rather than as scattered booleans through
the model code. Instantiate one :class:`ModelConfig` per experiment variant; see
``configs/*.yaml`` for the concrete variants used in the ablation sweep.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class SharingPattern(str, Enum):
    """Which layers within the recursive block share weights across loop iterations.

    Attributes:
        NONE: No weight sharing; a standard, fully unique-layer transformer (the dense baseline).
        FULL_LOOP: The entire block (attention + FFN) is tied and re-applied ``LoopConfig.
            num_loops`` times, following LoopMoE's "sandwich" design (Bae et al. 2024;
            arXiv:2606.04438).
        MIDDLE_CYCLE: First and last layers of the stack remain unique; only the middle layers
            cycle through a shared parameter block. Found to be the most robust tying strategy
            in Mixture-of-Recursions (arXiv:2507.10524) across 135M-1.7B scales.
        FFN_ONLY: Only the MoE feed-forward expert weights are tied across a group of
            consecutive layers; attention, router, and normalization stay independent per layer.
            Follows "Tying the Loop" (arXiv:2606.16825), which found untying attention costs far
            more quality than untying the router.
    """

    NONE = "none"
    FULL_LOOP = "full_loop"
    MIDDLE_CYCLE = "middle_cycle"
    FFN_ONLY = "ffn_only"


class PositionEncodingType(str, Enum):
    """Positional encoding scheme applied to attention queries/keys.

    Attributes:
        LEARNED: Original GPT-2-style learned absolute position embeddings.
        ROPE: Standard rotary position embedding applied to the full per-head dimension.
        DECOUPLED_ROPE: DeepSeek-V3 / Kimi K2 style MLA decoupled RoPE, applied only to a small
            per-head slice; the remaining ("NoPE") slice is position-agnostic and derived from
            the compressed latent. Required when ``AttentionConfig.use_mla`` is True.
        NONE: No explicit positional encoding is added by the block-level attention path.
            Used with ``MixerType.MAMBA2``, whose sequential recurrence is itself
            position-sensitive (no permutation-equivariant attention step requires RoPE).
    """

    LEARNED = "learned"
    ROPE = "rope"
    DECOUPLED_ROPE = "decoupled_rope"
    NONE = "none"


class MixerType(str, Enum):
    """Which sequence-mixing mechanism a transformer block uses in place of (or alongside)
    attention.

    Attributes:
        ATTENTION: Standard or MLA self-attention only (the original architecture of this
            codebase), selected via ``AttentionConfig.use_mla``.
        MAMBA2: Mamba-2 state-space sequence mixer only (arXiv:2405.21060), replacing
            attention entirely. Uses ``mamba_ssm.Mamba2``'s fused SSD (structured state-space
            duality) CUDA/Triton kernels. See :class:`MambaConfig` and
            ``docs/mamba_investigation.md``.
        HYBRID_MAMBA_ATTENTION: Both a Mamba-2 mixer and MLA/MHA attention are applied in
            sequence within the same shared block (Mamba-2 first, then attention), each with
            its own pre-norm and residual connection, following the Jamba/Zamba-style
            hybridization pattern of interleaving SSM and attention layers -- here fused into
            a single block position rather than alternated across separate block positions, so
            that looping over one shared block still lets a token pass through both mechanisms
            every iteration.
    """

    ATTENTION = "attention"
    MAMBA2 = "mamba2"
    HYBRID_MAMBA_ATTENTION = "hybrid_mamba_attention"


@dataclass(frozen=True)
class MoEConfig:
    """Sparse Mixture-of-Experts feed-forward configuration.

    Mirrors the Kimi K2 / DeepSeek-V3 mechanism (fine-grained routed experts + one always-on
    shared expert + auxiliary-loss-free load balancing via a per-expert bias), with expert count
    and top-k scaled down from Kimi K2's (384 routed, top-8) to a size that fits a GPT-2-small
    hidden dimension on a single consumer GPU. See ``docs/literature_survey.md`` §3.2.

    Attributes:
        enabled: If False, the block uses a standard dense FFN instead of MoE.
        num_routed_experts: Number of experts selected via learned top-k routing.
        num_shared_experts: Number of experts always active for every token, added on top of
            the routed experts' output. Kimi K2 uses 1; keep 1 unless ablating this away.
        top_k: Number of routed experts activated per token.
        expert_intermediate_size: Hidden width of each individual expert's FFN. Note this is
            the width of ONE expert, not the aggregate MoE layer width.
        bias_update_speed: Learning rate ``gamma`` for the auxiliary-loss-free load-balancing
            bias update (DeepSeek-V3 uses 0.001, annealed to 0 near the end of training).
        bias_update_speed_decay_frac: Fraction of total training steps, at the end of training,
            during which ``bias_update_speed`` is linearly annealed to zero.
    """

    enabled: bool = True
    num_routed_experts: int = 8
    num_shared_experts: int = 1
    top_k: int = 2
    expert_intermediate_size: int = 512
    bias_update_speed: float = 1e-3
    bias_update_speed_decay_frac: float = 0.1

    def __post_init__(self) -> None:
        if self.enabled and self.top_k > self.num_routed_experts:
            raise ValueError(
                f"top_k ({self.top_k}) cannot exceed num_routed_experts "
                f"({self.num_routed_experts})."
            )
        if self.enabled and self.num_routed_experts < 1:
            raise ValueError("num_routed_experts must be >= 1 when MoE is enabled.")
        if not 0.0 <= self.bias_update_speed_decay_frac <= 1.0:
            raise ValueError("bias_update_speed_decay_frac must be in [0, 1].")


@dataclass(frozen=True)
class AttentionConfig:
    """Attention mechanism configuration, supporting standard MHA or DeepSeek-V3-style MLA.

    Compressed dimensions default to GPT-2-small-scale equivalents of DeepSeek-V3's ratios
    (see ``docs/literature_survey.md`` §3.1: DeepSeek-V3 uses d_c=512, d_c'=1536 at hidden=7168,
    i.e. ratios of ~0.0715 and ~0.214 respectively).

    Attributes:
        use_mla: If True, use Multi-head Latent Attention with low-rank KV/Q compression and
            decoupled RoPE. If False, use standard multi-head attention.
        kv_compression_dim: Latent dimension ``d_c`` for the shared KV down-projection.
        q_compression_dim: Latent dimension ``d_c'`` for the query down-projection.
        rope_head_dim: Per-head dimension ``d_R^h`` carrying rotary position information.
            Only the decoupled RoPE slice uses this; the remaining per-head dimension is
            position-agnostic (NoPE) and derived from the compressed latent.
    """

    use_mla: bool = True
    kv_compression_dim: int = 64
    q_compression_dim: int = 160
    rope_head_dim: int = 32

    def __post_init__(self) -> None:
        if self.kv_compression_dim < 1 or self.q_compression_dim < 1:
            raise ValueError("Compression dimensions must be positive.")
        if self.rope_head_dim < 1:
            raise ValueError("rope_head_dim must be positive.")


@dataclass(frozen=True)
class MambaConfig:
    """Mamba-2 state-space sequence mixer configuration (arXiv:2405.21060).

    Only meaningful when ``ModelConfig.mixer_type`` is ``MixerType.MAMBA2`` or
    ``MixerType.HYBRID_MAMBA_ATTENTION``. Dimension names follow ``mamba_ssm.Mamba2``'s own
    constructor argument names directly, so this dataclass is a thin, validated, typed
    passthrough rather than a reinterpretation.

    Attributes:
        d_state: SSM state expansion factor per channel (Mamba-2's ``d_state``). The original
            Mamba-2 paper uses 128 at large scale; 64 is a reasonable reduction for GPT-2 scale.
        d_conv: Width of the local causal depthwise convolution applied before the SSM scan.
        expand: Expansion factor for the mixer's internal (inner) channel width relative to
            ``hidden_size``, i.e. inner_dim = ``expand * hidden_size``.
        headdim: Per-head dimension inside the SSM. ``inner_dim`` must be evenly divisible by
            ``headdim``, and ``mamba_ssm``'s fused Triton kernels additionally require the
            resulting tensor strides be multiples of 8 -- concretely, keep
            ``(expand * hidden_size) // headdim`` a multiple of 8 (e.g. hidden_size=256,
            expand=2, headdim=64 gives 8 heads exactly).
        ngroups: Number of groups sharing a single set of B/C SSM parameters (Mamba-2's
            multi-value-attention-style grouping); 1 recovers the original single-group design.
    """

    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1

    def __post_init__(self) -> None:
        if self.d_state < 1 or self.d_conv < 1 or self.expand < 1 or self.headdim < 1 or self.ngroups < 1:
            raise ValueError("All MambaConfig dimensions must be positive integers.")


@dataclass(frozen=True)
class LoopConfig:
    """Recurrent-depth (looping) configuration for the shared transformer block.

    Attributes:
        enabled: If False, the model is a standard fully-unrolled stack (no weight sharing).
        sharing_pattern: Which layers share weights; see :class:`SharingPattern`.
        num_loops: Fixed number of times the shared block is re-applied. Nanbeige4.2 found
            K=2 to be the empirically best trade-off at 3B scale; sweep {1, 2, 3, 4} here.
        num_unique_prefix_layers: Non-shared layers before the loop (LoopMoE "sandwich" prefix).
        num_unique_suffix_layers: Non-shared layers after the loop (LoopMoE "sandwich" suffix).
        use_iter_adaln: If True, apply iteration-conditioned adaptive LayerNorm (IterAdaLN,
            LoopMoE arXiv:2606.04438) at every pre-norm site inside the loop, generating
            per-iteration affine parameters from the loop index. Without this, weight sharing
            collapses to a near-fixed-point after a few iterations (see literature survey §2.1).
        use_active_ratio_balancing: If True, apply LoopMoE's correction for the
            attention-to-FFN active-parameter ratio drift that MoE routing introduces under
            looping (see literature survey §2.2). Only meaningful when MoE is also enabled.
        use_router_gating: If True, add a Mixture-of-Recursions-style per-token router that can
            exit the loop early for "easy" tokens, instead of a fixed loop count for every
            token. When True, ``router`` must be provided. See :class:`RouterConfig` and
            ``docs/adaptive_depth_router_design.md`` for the full mechanism.
        router: Router hyperparameters; required and only meaningful when ``use_router_gating``
            is True.
    """

    enabled: bool = True
    sharing_pattern: SharingPattern = SharingPattern.MIDDLE_CYCLE
    num_loops: int = 2
    num_unique_prefix_layers: int = 1
    num_unique_suffix_layers: int = 1
    use_iter_adaln: bool = True
    use_active_ratio_balancing: bool = True
    use_router_gating: bool = False
    router: Optional["RouterConfig"] = None

    def __post_init__(self) -> None:
        if self.enabled and self.num_loops < 1:
            raise ValueError("num_loops must be >= 1 when looping is enabled.")
        if self.num_unique_prefix_layers < 0 or self.num_unique_suffix_layers < 0:
            raise ValueError("Prefix/suffix layer counts must be non-negative.")
        if self.use_router_gating and self.router is None:
            raise ValueError("router config must be provided when use_router_gating=True.")
        if self.use_router_gating and not self.enabled:
            raise ValueError("use_router_gating requires loop.enabled=True.")
        if self.use_router_gating and self.router.min_loops > self.num_loops:
            raise ValueError(
                f"router.min_loops ({self.router.min_loops}) cannot exceed num_loops "
                f"({self.num_loops}) -- every token must be able to reach min_loops within the "
                f"configured loop count."
            )


@dataclass(frozen=True)
class RouterConfig:
    """Per-token adaptive loop-depth router configuration.

    Three routing mechanisms are available (``mechanism``), with important structural
    differences documented empirically in ``docs/adaptive_depth_router_design.md``'s "Empirical
    finding" sections:

    - ``"capacity"`` (Mixture-of-Recursions style, the original implementation): every
      iteration, exactly the top ``capacity_ratio`` FRACTION of currently-active tokens survive
      -- this fraction is fixed by the config, not learned, so this mechanism can only learn
      WHICH tokens occupy a predetermined-size survivor pool at each depth, never how many
      tokens (or how much total compute) a given input receives. Found empirically to learn a
      confidence-gating pattern (easy tokens routed deeper, not hard ones) rather than the
      intended difficulty-gating.
    - ``"act"`` (Adaptive Computation Time style, Graves 2016 / Universal Transformer's ACT):
      each token independently accumulates a per-iteration halting probability and stops once
      its own cumulative probability crosses a threshold -- no fixed survivor fraction, so
      different inputs CAN receive different total amounts of compute. Found empirically (via
      direct gradient inspection during this project's development) to reproduce a
      literature-documented ACT limitation: gradient for the halting decision only flows
      through the LAST step a token was routed through, via a "remainder" term with no
      dependency on that step's own halting-probability output -- routers whose only role in a
      batch is causing tokens to halt (never continuing tokens past them) receive ZERO training
      signal. Kept in this codebase for reference/comparison, not recommended for new use.
    - ``"pondernet"`` (Banino et al. 2021, arXiv:2107.05407): the field's own fix for ACT's
      gradient-bias problem. Defines the probability of halting at exactly step n as
      ``p_n = lambda_n * prod_{j<n}(1 - lambda_j)`` -- because every later p_n multiplies in
      EVERY earlier step's lambda, gradients reach every step's halting head, not just the
      last one. Uses a KL-divergence regularization against a geometric prior
      (``geometric_prior_lambda``) instead of ACT's ponder-cost penalty. Recommended over
      ``"act"`` for any real experiment.

    Implements the masked-dense strategy recommended in
    ``docs/adaptive_depth_router_design.md`` step 2 for all three mechanisms: every token is
    computed at every iteration (so tensor shapes stay static, avoiding the dynamic-shape/
    recompilation issues found when profiling :class:`~looped_moe_gpt2.model.moe.SparseMoE`),
    but only tokens still active (or, for act/pondernet, not yet fully halted) have their
    updated hidden state written back.

    Attributes:
        mechanism: ``"capacity"``, ``"act"``, or ``"pondernet"`` -- selects which router
            implementation :class:`~looped_moe_gpt2.model.gpt.LoopedMoEGPT` constructs.
        capacity_ratio: Only used when ``mechanism="capacity"``. Fraction of still-active
            tokens kept active at each iteration (the rest exit). E.g. 0.5 means each loop
            iteration roughly halves the active token count.
        min_loops: Every token receives at least this many iterations before the router can
            exit it, preventing a degenerate all-tokens-exit-immediately collapse early in
            training. Used by all three mechanisms.
        aux_loss_weight: Only used when ``mechanism="capacity"``. Weight on the auxiliary loss
            encouraging the soft (differentiable, training-time) and hard (top-k,
            inference-time) routing decisions to agree.
        ponder_cost_weight: Used by ``"act"`` (as the ponder cost penalty weight) and
            ``"pondernet"`` (as ``beta``, the KL-regularization weight). Start small (e.g. 0.01)
            and increase if tokens never halt before ``num_loops`` in practice; too large a
            value trades away prediction quality for adaptivity.
        geometric_prior_lambda: Only used when ``mechanism="pondernet"``. The geometric prior's
            constant per-step halting rate -- e.g. 0.2 encodes a prior belief of an expected
            ponder depth of ``1 / geometric_prior_lambda`` = 5 steps.
    """

    mechanism: str = "capacity"
    capacity_ratio: float = 0.5
    min_loops: int = 1
    aux_loss_weight: float = 0.01
    ponder_cost_weight: float = 0.01
    geometric_prior_lambda: float = 0.2

    def __post_init__(self) -> None:
        if self.mechanism not in ("capacity", "act", "pondernet"):
            raise ValueError(f"mechanism must be 'capacity', 'act', or 'pondernet', got '{self.mechanism}'.")
        if not 0.0 < self.capacity_ratio <= 1.0:
            raise ValueError("capacity_ratio must be in (0, 1].")
        if self.min_loops < 1:
            raise ValueError("min_loops must be >= 1.")
        if self.mechanism in ("act", "pondernet") and self.ponder_cost_weight <= 0.0:
            raise ValueError(
                f"ponder_cost_weight must be positive when mechanism='{self.mechanism}' -- "
                f"without this regularization, the model has no incentive to ever halt early "
                f"(more compute never hurts prediction quality, only efficiency), so this would "
                f"silently degenerate to every token always running the full num_loops."
            )
        if self.mechanism == "pondernet" and not 0.0 < self.geometric_prior_lambda < 1.0:
            raise ValueError("geometric_prior_lambda must be in (0, 1) when mechanism='pondernet'.")
        if self.aux_loss_weight < 0.0:
            raise ValueError("aux_loss_weight must be non-negative.")


@dataclass(frozen=True)
class ModelConfig:
    """Top-level model configuration composing embedding, attention, MoE, and loop settings.

    Attributes:
        vocab_size: Tokenizer vocabulary size (50257 for GPT-2's tiktoken ``gpt2`` encoding).
        hidden_size: Model (residual stream) width.
        num_layers: Number of unique transformer layers *before* any loop expansion — i.e. the
            physical parameter count is governed by this, while effective depth is
            ``num_layers`` scaled by ``loop.num_loops`` when looping is enabled.
        num_heads: Number of attention heads.
        max_seq_len: Maximum sequence length (context window).
        dropout: Residual/attention/embedding dropout probability.
        position_encoding: See :class:`PositionEncodingType`.
        attention: See :class:`AttentionConfig`.
        moe: See :class:`MoEConfig`.
        loop: See :class:`LoopConfig`.
        mixer_type: See :class:`MixerType`. Determines whether each block uses attention only,
            a Mamba-2 mixer only, or both in a hybrid block.
        mamba: See :class:`MambaConfig`. Required (and only meaningful) when ``mixer_type`` is
            ``MixerType.MAMBA2`` or ``MixerType.HYBRID_MAMBA_ATTENTION``.
        tie_word_embeddings: If True, tie the input embedding and output projection weights
            (standard GPT-2 practice; reduces params).
        seed: Random seed for reproducible initialization.
    """

    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    max_seq_len: int = 1024
    dropout: float = 0.1
    position_encoding: PositionEncodingType = PositionEncodingType.DECOUPLED_ROPE
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    mixer_type: MixerType = MixerType.ATTENTION
    mamba: Optional[MambaConfig] = None
    tie_word_embeddings: bool = True
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by num_heads "
                f"({self.num_heads})."
            )
        uses_mamba = self.mixer_type in (MixerType.MAMBA2, MixerType.HYBRID_MAMBA_ATTENTION)
        uses_attention = self.mixer_type in (MixerType.ATTENTION, MixerType.HYBRID_MAMBA_ATTENTION)
        if uses_mamba and self.mamba is None:
            raise ValueError(
                f"mamba config must be provided when mixer_type={self.mixer_type.value!r}."
            )
        if uses_mamba:
            inner_dim = self.mamba.expand * self.hidden_size
            if inner_dim % self.mamba.headdim != 0:
                raise ValueError(
                    f"mamba.expand * hidden_size ({inner_dim}) must be divisible by "
                    f"mamba.headdim ({self.mamba.headdim})."
                )
            num_mamba_heads = inner_dim // self.mamba.headdim
            if num_mamba_heads % 8 != 0:
                raise ValueError(
                    f"(mamba.expand * hidden_size) // mamba.headdim ({num_mamba_heads}) must be "
                    f"a multiple of 8 -- mamba_ssm's fused Triton kernels require tensor strides "
                    f"that are multiples of 8. Adjust hidden_size, mamba.expand, or "
                    f"mamba.headdim."
                )
        if self.mixer_type == MixerType.MAMBA2 and self.position_encoding != PositionEncodingType.NONE:
            raise ValueError(
                "mixer_type=MixerType.MAMBA2 (no attention path) requires "
                "position_encoding=PositionEncodingType.NONE -- Mamba-2's recurrence is itself "
                "position-sensitive and does not consume RoPE/learned position embeddings."
            )
        if uses_attention:
            if self.attention.use_mla and self.position_encoding != PositionEncodingType.DECOUPLED_ROPE:
                raise ValueError(
                    "AttentionConfig.use_mla=True requires "
                    "position_encoding=PositionEncodingType.DECOUPLED_ROPE."
                )
            if not self.attention.use_mla and self.position_encoding == PositionEncodingType.DECOUPLED_ROPE:
                raise ValueError(
                    "position_encoding=DECOUPLED_ROPE requires AttentionConfig.use_mla=True."
                )
            head_dim = self.hidden_size // self.num_heads
            if self.attention.use_mla and self.attention.rope_head_dim > head_dim:
                raise ValueError(
                    f"rope_head_dim ({self.attention.rope_head_dim}) cannot exceed the per-head "
                    f"dimension ({head_dim})."
                )
        if self.loop.enabled:
            min_layers = self.loop.num_unique_prefix_layers + self.loop.num_unique_suffix_layers
            if self.loop.sharing_pattern != SharingPattern.NONE and self.num_layers <= min_layers:
                raise ValueError(
                    f"num_layers ({self.num_layers}) must exceed the combined prefix+suffix "
                    f"unique layers ({min_layers}) to leave a non-empty shared loop block."
                )
        logger.debug("Validated ModelConfig: %s", self)

    @property
    def head_dim(self) -> int:
        """Per-head dimension, derived from hidden_size and num_heads."""
        return self.hidden_size // self.num_heads

    @property
    def effective_depth(self) -> int:
        """Total number of block applications in a forward pass, accounting for looping."""
        if not self.loop.enabled:
            return self.num_layers
        loop_body_layers = self.num_layers - (
            self.loop.num_unique_prefix_layers + self.loop.num_unique_suffix_layers
        )
        return (
            self.loop.num_unique_prefix_layers
            + loop_body_layers * self.loop.num_loops
            + self.loop.num_unique_suffix_layers
        )
