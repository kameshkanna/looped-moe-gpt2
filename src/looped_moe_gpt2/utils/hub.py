"""Resolve a checkpoint/config source that may be a local path or a Hugging Face Hub repo id.

Used by every script that loads a trained checkpoint (``scripts/chat.py``,
``scripts/evaluate.py``, ``scripts/eval_reasoning_accuracy.py``) so a Hub repo id can be passed
in place of a local file path everywhere, without duplicating the download logic in each script.
"""

from __future__ import annotations

import logging
from pathlib import Path

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


def resolve_checkpoint_and_config(source: str) -> tuple[Path, Path]:
    """Resolve a checkpoint path and its config path from either a local directory or a Hub repo id.

    A source is treated as a Hugging Face Hub repo id (not a local path) if it is NOT an
    existing local path and contains exactly one ``/`` (the ``owner/repo-name`` shape) --
    otherwise it's treated as a local checkpoint file path, with the config expected alongside
    it (see the ``Raises`` note below for the exact local-path convention this assumes).

    Args:
        source: Either:

            - A Hugging Face Hub repo id, e.g. ``"Kameshr/looped-moe-gpt2-reasoning"`` -- expects
              the repo to contain ``checkpoint.pt`` and ``config.yaml`` at its root (this is
              exactly the layout :func:`scripts.push_to_hub.main` uploads).
            - A local path to a checkpoint ``.pt`` file -- in this case a separate config path
              must be supplied by the caller directly (this function only handles the Hub case;
              for a plain local checkpoint, skip this function and pass both paths as before).

    Returns:
        A tuple ``(checkpoint_path, config_path)``, both local filesystem paths (downloaded into
        the Hugging Face Hub cache if ``source`` was a repo id, so repeated calls with the same
        repo id are cheap after the first download).

    Raises:
        ValueError: If ``source`` looks like a local path (exists on disk, or does not have the
            ``owner/repo-name`` shape) -- callers should not pass a local checkpoint path to this
            function; it exists only to resolve the Hub-repo-id case.
    """
    local_path = Path(source)
    if local_path.exists() or "/" not in source or source.count("/") != 1:
        raise ValueError(
            f"'{source}' does not look like a Hugging Face Hub repo id (expected exactly one "
            f"'/', e.g. 'owner/repo-name', and no existing local path with this name). If you "
            f"meant to pass a local checkpoint file, use --checkpoint/--config directly instead "
            f"of this resolver."
        )

    logger.info("Resolving '%s' as a Hugging Face Hub repo id -- downloading checkpoint.pt and config.yaml...", source)
    checkpoint_path = Path(hf_hub_download(repo_id=source, filename="checkpoint.pt"))
    config_path = Path(hf_hub_download(repo_id=source, filename="config.yaml"))
    logger.info("Downloaded (or found cached): %s, %s", checkpoint_path, config_path)
    return checkpoint_path, config_path
