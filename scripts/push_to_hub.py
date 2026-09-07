#!/usr/bin/env python
"""CLI: push a trained checkpoint to the Hugging Face Hub.

This model's architecture (looped MoE + MLA + adaptive-depth router) is NOT a `transformers`
built-in model class, so this pushes the raw checkpoint file plus a model card explaining how
to load it using this repo's own code (`looped_moe_gpt2.model.gpt.LoopedMoEGPT`) -- it does not
produce a `transformers`-AutoModel-compatible repo. Anyone using this checkpoint needs to
`pip install` (or clone) the `looped-moe-gpt2` package first; the model card says so explicitly.

Usage (run this on whatever machine/pod actually holds the checkpoint -- do not copy a
multi-GB checkpoint elsewhere just to run this):
    hf auth login   # if not already logged in
    python scripts/push_to_hub.py --checkpoint runs/05_curriculum_reasoning/checkpoint_step51727.pt \
        --config configs/05_curriculum_reasoning.yaml --repo-id <your-hf-username>/looped-moe-gpt2-reasoning
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from pathlib import Path

import torch
import yaml
from huggingface_hub import HfApi, whoami

from looped_moe_gpt2.train.checkpoint import load_checkpoint
from looped_moe_gpt2.utils.config_io import load_model_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def build_model_card(
    repo_id: str,
    config_path: Path,
    checkpoint_step: int,
    val_loss: float,
    architecture_summary: dict,
) -> str:
    """Compose the model card markdown, with honest caveats about what val_loss does/doesn't show.

    Args:
        repo_id: Target Hub repo id (e.g. "username/model-name"), used in the card's title/example.
        config_path: Path to the YAML config this checkpoint was trained with (linked in the card
            for reproducibility, not uploaded itself unless the caller also uploads it).
        checkpoint_step: The optimizer step this checkpoint was saved at.
        val_loss: The checkpoint's recorded best validation loss (from the checkpoint's own
            metadata) -- reported as-is, with an explicit caveat about what it does and doesn't
            demonstrate (see the "Evaluation" section this function writes).
        architecture_summary: Key architecture facts (params, effective_depth, etc.) to render
            in the card's summary table.

    Returns:
        The complete model card as a markdown string, including YAML frontmatter.
    """
    return f"""---
license: mit
tags:
  - pytorch
  - looped-transformer
  - mixture-of-experts
  - recurrent-depth
  - adaptive-computation
language:
  - en
---

# {repo_id.split('/')[-1]}

A GPT-2-scale looped (recurrent-depth) transformer combining sparse Mixture-of-Experts (MoE),
Multi-head Latent Attention (MLA), and a per-token adaptive loop-depth router. Trained on a
general-English + math-reasoning curriculum. See the
[project repository](https://github.com/kameshkanna/looped-moe-gpt2) for the full architecture,
literature survey, and training code.

## Architecture

| | |
|---|---|
| Total parameters | {architecture_summary['total_params']:,} |
| Effective depth (with looping) | {architecture_summary['effective_depth']} |
| Hidden size | {architecture_summary['hidden_size']} |
| Attention | Multi-head Latent Attention (MLA), DeepSeek-V3-style, decoupled RoPE |
| Feed-forward | Sparse MoE, {architecture_summary['num_routed_experts']} routed experts (top-{architecture_summary['top_k']}) + 1 shared expert, auxiliary-loss-free load balancing |
| Looping | `SharingPattern.FULL_LOOP`, {architecture_summary['num_loops']} iterations, per-token adaptive-depth routing (Mixture-of-Recursions-style) |
| Checkpoint step | {checkpoint_step:,} |

This is **not** a `transformers`-library `AutoModel`-compatible checkpoint -- the architecture
(block-level weight sharing, sparse MoE routing, per-token adaptive depth) has no equivalent
built-in model class. Loading this checkpoint requires this project's own code.

## Training data

A two-phase curriculum: general English ([FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu))
for the first 25% of training, with math-reasoning data
([OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)) ramped in
linearly from 25%-45% of training and held at equal weight with general text for the remainder.
General text is never fully removed from the mix, to avoid the narrow-capability collapse risk
of a pure-math cold start. See `configs/{config_path.name}` in the project repo for the exact
schedule.

## Evaluation

**Validation loss at this checkpoint: {val_loss:.4f}** (cross-entropy, on held-out text from the
same corpora used for training).

**Important caveat**: this is a next-token-prediction loss, not a measure of correctness on
reasoning tasks. A low loss on math-formatted text means the model predicts the *surface
structure* of math solutions well (it has learned the templates/phrasing common in the training
corpus) -- it does **not** by itself demonstrate that the model performs arithmetic or logical
reasoning correctly. No answer-accuracy evaluation (e.g. GSM8K-style exact-match scoring) has
been run on this checkpoint as of this upload; treat any math-reasoning claims about this model
as unverified until such an evaluation is added.

## How to load this checkpoint

```bash
git clone https://github.com/kameshkanna/looped-moe-gpt2.git
cd looped-moe-gpt2
pip install -e .
```

```python
from pathlib import Path
from huggingface_hub import hf_hub_download
from looped_moe_gpt2.utils.config_io import load_model_config
from looped_moe_gpt2.model.gpt import LoopedMoEGPT
from looped_moe_gpt2.train.checkpoint import load_checkpoint
import torch

checkpoint_path = Path(hf_hub_download(repo_id="{repo_id}", filename="checkpoint.pt"))
config_path = Path(hf_hub_download(repo_id="{repo_id}", filename="config.yaml"))

model_config = load_model_config(config_path)
model = LoopedMoEGPT(model_config)
checkpoint = load_checkpoint(checkpoint_path, device=torch.device("cpu"))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

See `scripts/chat.py` in the project repo for a ready-to-run interactive generation script.

## License

MIT (matches the project repository's license).
"""


def main() -> None:
    """Parse CLI arguments and push the checkpoint + config + model card to the Hub."""
    parser = argparse.ArgumentParser(description="Push a trained checkpoint to the HF Hub.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the .pt checkpoint.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the variant's YAML config.")
    parser.add_argument("--repo-id", type=str, required=True, help="Target repo, e.g. username/model-name.")
    parser.add_argument("--private", action="store_true", help="Create the Hub repo as private.")
    args = parser.parse_args()

    try:
        user = whoami()
    except Exception as e:
        raise RuntimeError(
            "Not logged in to the Hugging Face Hub in this environment. Run `hf auth login` first."
        ) from e
    logger.info("Logged in as: %s", user["name"])

    model_config = load_model_config(args.config)
    checkpoint = load_checkpoint(args.checkpoint, device=torch.device("cpu"))
    total_params = sum(v.numel() for v in checkpoint["model_state_dict"].values())

    architecture_summary = {
        "total_params": total_params,
        "effective_depth": model_config.effective_depth,
        "hidden_size": model_config.hidden_size,
        "num_routed_experts": model_config.moe.num_routed_experts,
        "top_k": model_config.moe.top_k,
        "num_loops": model_config.loop.num_loops,
    }

    api = HfApi()
    logger.info("Creating (or reusing) repo: %s (private=%s)", args.repo_id, args.private)
    api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        logger.info("Copying checkpoint (%.2f GB)...", args.checkpoint.stat().st_size / 1e9)
        shutil.copy(args.checkpoint, tmp_path / "checkpoint.pt")
        shutil.copy(args.config, tmp_path / "config.yaml")

        model_card = build_model_card(
            repo_id=args.repo_id,
            config_path=args.config,
            checkpoint_step=checkpoint["step"],
            val_loss=checkpoint["best_val_loss"],
            architecture_summary=architecture_summary,
        )
        (tmp_path / "README.md").write_text(model_card, encoding="utf-8")

        logger.info("Uploading to %s ...", args.repo_id)
        api.upload_folder(folder_path=str(tmp_path), repo_id=args.repo_id, repo_type="model")

    logger.info("Done. https://huggingface.co/%s", args.repo_id)


if __name__ == "__main__":
    main()
