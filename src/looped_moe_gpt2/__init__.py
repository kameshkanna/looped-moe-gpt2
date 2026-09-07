"""Looped Mixture-of-Experts GPT-2: a GPT-2-scale recurrent-depth transformer.

This package implements a family of GPT-2-small-scale architectures for studying the
interaction between block-level weight sharing (looped / recurrent-depth transformers, as in
Universal Transformer, Huginn, Nanbeige4.2, and the rumored GPT-6 "Astra") and sparse
Mixture-of-Experts routing with Multi-head Latent Attention (as in DeepSeek-V3 and Kimi K2).

See ``docs/literature_survey.md`` for the full survey grounding these design choices, and
``configs/`` for the specific model variants this package can instantiate.
"""

from looped_moe_gpt2.model.config import ModelConfig, MoEConfig, LoopConfig

__all__ = ["ModelConfig", "MoEConfig", "LoopConfig"]
