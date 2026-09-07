"""Top-level looped MoE GPT model.

Assembles the LoopMoE-style "sandwich" architecture (arXiv:2606.04438): a run of unique prefix
layers, a shared block re-applied ``LoopConfig.num_loops`` times (or, under
``SharingPattern.NONE``, a plain unrolled stack with no sharing at all), and a run of unique
suffix layers. Weight sharing is realized simply — by re-registering the SAME
:class:`~looped_moe_gpt2.model.block.TransformerBlock` module instances at each virtual
position in the loop, rather than deep-copying them, so ``nn.Module`` parameter registration and
the optimizer naturally see one set of shared parameters. See
:class:`looped_moe_gpt2.model.config.ModelConfig` for the full configuration surface and
``docs/literature_survey.md`` for the design rationale.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import nn

from looped_moe_gpt2.model.block import TransformerBlock
from looped_moe_gpt2.model.config import ModelConfig, PositionEncodingType, SharingPattern
from looped_moe_gpt2.model.router import AdaptiveDepthRouter

logger = logging.getLogger(__name__)


class LoopedMoEGPT(nn.Module):
    """GPT-2-scale decoder-only transformer with optional block looping and sparse MoE.

    Args:
        config: Full model configuration; see :class:`looped_moe_gpt2.model.config.ModelConfig`.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        torch.manual_seed(config.seed)

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.use_learned_pos_emb = config.position_encoding == PositionEncodingType.LEARNED
        self.position_embedding: Optional[nn.Embedding] = (
            nn.Embedding(config.max_seq_len, config.hidden_size) if self.use_learned_pos_emb else None
        )
        self.embedding_dropout = nn.Dropout(config.dropout)

        self.blocks, self.loop_plan = self._build_blocks(config)

        self.router_gated = config.loop.enabled and config.loop.use_router_gating
        self.routers: Optional[nn.ModuleList] = None
        if self.router_gated:
            if config.loop.sharing_pattern != SharingPattern.FULL_LOOP:
                raise ValueError(
                    "use_router_gating currently requires sharing_pattern=FULL_LOOP -- combining "
                    "per-token adaptive exit with Middle-Cycle's multi-layer-per-iteration "
                    "structure is an open question (see docs/adaptive_depth_router_design.md) "
                    "and not implemented."
                )
            assert config.loop.router is not None  # validated in ModelConfig.__post_init__
            self.routers = nn.ModuleList(
                [
                    AdaptiveDepthRouter(config.hidden_size, config.loop.router)
                    for _ in range(config.loop.num_loops)
                ]
            )
            self.router_min_loops = config.loop.router.min_loops
            self.last_router_aux_loss: torch.Tensor = torch.zeros(())
            self.last_exit_iteration: Optional[torch.Tensor] = None

        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        logger.info(
            "Initialized LoopedMoEGPT: %d unique layers, effective depth %d, %d total params",
            config.num_layers,
            config.effective_depth,
            sum(p.numel() for p in self.parameters()),
        )

    def _build_blocks(self, config: ModelConfig) -> tuple[nn.ModuleList, list[tuple[int, Optional[int]]]]:
        """Construct the unique block modules and the (block_index, iteration_idx) execution plan.

        Args:
            config: Full model configuration.

        Returns:
            A tuple ``(blocks, loop_plan)`` where ``blocks`` holds one
            :class:`~looped_moe_gpt2.model.block.TransformerBlock` per *unique* layer, and
            ``loop_plan`` is an ordered list of ``(block_index, iteration_idx)`` pairs describing
            the actual forward-pass execution sequence — ``iteration_idx`` is None for
            non-looped blocks and the 0-indexed loop iteration for looped ones.
        """
        if not config.loop.enabled or config.loop.sharing_pattern == SharingPattern.NONE:
            blocks = nn.ModuleList(
                [self._make_block(config, use_iter_adaln=False, max_iterations=None) for _ in range(config.num_layers)]
            )
            plan = [(i, None) for i in range(config.num_layers)]
            return blocks, plan

        num_prefix = config.loop.num_unique_prefix_layers
        num_suffix = config.loop.num_unique_suffix_layers
        num_loop_body = config.num_layers - num_prefix - num_suffix
        num_loops = config.loop.num_loops

        unique_blocks: list[TransformerBlock] = []
        plan: list[tuple[int, Optional[int]]] = []

        for _ in range(num_prefix):
            unique_blocks.append(self._make_block(config, use_iter_adaln=False, max_iterations=None))
            plan.append((len(unique_blocks) - 1, None))

        if config.loop.sharing_pattern == SharingPattern.MIDDLE_CYCLE:
            # Following MoR's "Middle-Cycle" pattern (arXiv:2507.10524): the middle section
            # keeps `num_loop_body` DISTINCT layers (preserving inter-layer diversity within one
            # pass through the stack), and that whole distinct sub-stack is cycled `num_loops`
            # times end-to-end -- i.e. layer i at iteration t shares weights with layer i at
            # iteration t' != t, but not with layer j != i within the same iteration. Combined
            # with the unique prefix/suffix, effective depth is
            # num_prefix + num_loop_body * num_loops + num_suffix.
            loop_body_blocks = [
                self._make_block(config, use_iter_adaln=config.loop.use_iter_adaln, max_iterations=num_loops)
                for _ in range(num_loop_body)
            ]
            unique_blocks.extend(loop_body_blocks)
            body_start_idx = len(unique_blocks) - num_loop_body
            for iteration_idx in range(num_loops):
                for layer_offset in range(num_loop_body):
                    plan.append((body_start_idx + layer_offset, iteration_idx))
        elif config.loop.sharing_pattern in (SharingPattern.FULL_LOOP, SharingPattern.FFN_ONLY):
            # FULL_LOOP: exactly one shared block, re-applied num_loops times.
            # FFN_ONLY: modeled here as a single shared block for simplicity (attention/router
            # untying at finer granularity is a documented follow-up; see literature survey §4).
            shared_block = self._make_block(
                config, use_iter_adaln=config.loop.use_iter_adaln, max_iterations=num_loops
            )
            unique_blocks.append(shared_block)
            body_idx = len(unique_blocks) - 1
            for iteration_idx in range(num_loops):
                plan.append((body_idx, iteration_idx))
        else:
            raise ValueError(f"Unhandled sharing pattern: {config.loop.sharing_pattern}")

        for _ in range(num_suffix):
            unique_blocks.append(self._make_block(config, use_iter_adaln=False, max_iterations=None))
            plan.append((len(unique_blocks) - 1, None))

        return nn.ModuleList(unique_blocks), plan

    def _make_block(
        self, config: ModelConfig, use_iter_adaln: bool, max_iterations: Optional[int]
    ) -> TransformerBlock:
        """Instantiate a single :class:`TransformerBlock` from the model configuration.

        Args:
            config: Full model configuration.
            use_iter_adaln: Whether this block should use iteration-conditioned normalization.
            max_iterations: Required if ``use_iter_adaln`` is True.

        Returns:
            A configured :class:`TransformerBlock` instance.
        """
        return TransformerBlock(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            attention_config=config.attention,
            position_encoding=config.position_encoding,
            moe_config=config.moe,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            use_iter_adaln=use_iter_adaln,
            max_iterations=max_iterations,
        )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Apply GPT-2-style weight initialization.

        Args:
            module: Submodule being visited by ``nn.Module.apply``.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the model forward, optionally computing cross-entropy loss.

        Args:
            input_ids: Token ids of shape ``(batch, seq_len)``.
            targets: Optional target token ids of shape ``(batch, seq_len)`` for next-token
                cross-entropy loss (typically ``input_ids`` shifted by one).

        Returns:
            A tuple ``(logits, loss)``: ``logits`` has shape ``(batch, seq_len, vocab_size)``;
            ``loss`` is a scalar tensor if ``targets`` was provided, else None.

        Raises:
            ValueError: If ``input_ids``'s sequence length exceeds ``config.max_seq_len``.
        """
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds configured max_seq_len "
                f"({self.config.max_seq_len})."
            )

        x = self.token_embedding(input_ids)
        if self.use_learned_pos_emb:
            assert self.position_embedding is not None
            positions = torch.arange(seq_len, device=input_ids.device)
            x = x + self.position_embedding(positions)
        x = self.embedding_dropout(x)

        if self.router_gated:
            x = self._forward_router_gated_loop(x)
        else:
            for block_idx, iteration_idx in self.loop_plan:
                x = self.blocks[block_idx](x, iteration_idx=iteration_idx)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            if self.router_gated:
                loss = loss + self.last_router_aux_loss
        return logits, loss

    def _forward_router_gated_loop(self, x: torch.Tensor) -> torch.Tensor:
        """Run unique prefix layers, the router-gated shared-block loop, then unique suffix layers.

        Implements the masked-dense strategy from ``docs/adaptive_depth_router_design.md``:
        every token passes through the shared block at every iteration (static shapes,
        compile/CUDA-graph friendly), but a token's contribution to the running hidden state is
        scaled by that iteration's router-determined keep-probability -- once a token exits, its
        keep-probability is 0 for all subsequent iterations, so its hidden state stops updating
        (frozen at its last-active value) even though the block keeps computing on it.

        Args:
            x: Hidden states entering this method, shape ``(batch, seq_len, hidden_size)``,
                i.e. straight from the embedding layer (this method owns prefix/suffix
                application itself, unlike the non-router-gated path which reads directly from
                ``self.loop_plan``).

        Returns:
            Hidden states after prefix, loop, and suffix, same leading shape as ``x``. Also
            populates ``self.last_router_aux_loss`` (scalar, to be added to the training loss)
            and ``self.last_exit_iteration`` (``(batch, seq_len)`` int tensor of the iteration
            index each token last updated at -- a direct measurement of adaptive depth per
            token, for diagnostics).
        """
        assert self.routers is not None
        prefix_indices = [idx for idx, it in self.loop_plan[: self.config.loop.num_unique_prefix_layers] if it is None]
        body_block_idx = next(idx for idx, it in self.loop_plan if it is not None)
        suffix_indices = [
            idx
            for idx, it in self.loop_plan[self.config.loop.num_unique_prefix_layers + self.config.loop.num_loops :]
            if it is None
        ]

        for block_idx in prefix_indices:
            x = self.blocks[block_idx](x, iteration_idx=None)

        batch_size, seq_len, _ = x.shape
        active_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
        exit_iteration = torch.full((batch_size, seq_len), -1, dtype=torch.long, device=x.device)
        total_aux_loss = torch.zeros((), device=x.device, dtype=x.dtype)

        for iteration_idx in range(self.config.loop.num_loops):
            block_out = self.blocks[body_block_idx](x, iteration_idx=iteration_idx)
            force_keep_all = iteration_idx < self.router_min_loops
            next_active_mask, soft_keep_prob, aux_loss = self.routers[iteration_idx](
                x, active_mask, force_keep_all=force_keep_all
            )
            total_aux_loss = total_aux_loss + aux_loss

            keep_prob = soft_keep_prob.unsqueeze(-1) if self.training else active_mask.float().unsqueeze(-1)
            x = x + keep_prob * (block_out - x)  # blend toward block_out only where still active

            newly_exited = active_mask & ~next_active_mask
            exit_iteration = torch.where(newly_exited, iteration_idx, exit_iteration)
            active_mask = next_active_mask

        exit_iteration = torch.where(exit_iteration == -1, self.config.loop.num_loops - 1, exit_iteration)
        self.last_router_aux_loss = total_aux_loss
        self.last_exit_iteration = exit_iteration

        for block_idx in suffix_indices:
            x = self.blocks[block_idx](x, iteration_idx=None)

        return x

    def update_all_routing_biases(self) -> None:
        """Call :meth:`TransformerBlock.update_routing_bias` on every block with an MoE layer.

        Must be invoked once per training step, after ``optimizer.step()``, so each MoE layer's
        auxiliary-loss-free routing bias reflects that step's expert load.
        """
        for block in self.blocks:
            block.update_routing_bias()

    def expert_load_summary(self) -> dict[int, torch.Tensor]:
        """Collect the most recent per-expert load from every MoE block, keyed by block index.

        Returns:
            A dict mapping unique-block index to its last forward pass's per-expert load tensor,
            for blocks that use MoE. Empty if no block uses MoE.
        """
        return {
            idx: load
            for idx, block in enumerate(self.blocks)
            if (load := block.expert_load()) is not None
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressively sample new tokens, optionally stopping early at an end-of-text token.

        Args:
            input_ids: Prompt token ids of shape ``(batch, seq_len)``.
            max_new_tokens: Maximum number of tokens to generate (generation may stop earlier,
                per-sequence, if ``eos_token_id`` is given and gets sampled).
            temperature: Sampling temperature; must be positive.
            eos_token_id: If given, once a sequence samples this token it stops being extended
                (subsequent positions for that sequence are padded with ``eos_token_id`` rather
                than continuing to sample) -- generation for the whole batch still runs until
                EVERY sequence has stopped or ``max_new_tokens`` is reached, whichever is first.
                If None (the default), every sequence always runs the full ``max_new_tokens``,
                matching this method's original behavior.

        Returns:
            Token ids of shape ``(batch, seq_len + n)`` where ``n <= max_new_tokens`` is however
            many steps actually ran before every sequence had stopped (or ``max_new_tokens`` if
            ``eos_token_id`` is None or never sampled). Positions after a given sequence's own
            stopping point are filled with ``eos_token_id``.

        Raises:
            ValueError: If ``temperature`` is not positive.
        """
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}.")

        self.eval()
        batch_size = input_ids.shape[0]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.max_seq_len :]
            logits, _ = self.forward(context)
            next_token_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1), torch.full_like(next_token, eos_token_id), next_token
                )
                finished = finished | (next_token.squeeze(-1) == eos_token_id)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if eos_token_id is not None and finished.all():
                break

        return input_ids
