"""Tokenization: raw text -> a flat uint16 token-id array persisted as a binary file.

Uses tiktoken's GPT-2 BPE encoding (matching ``ModelConfig.vocab_size=50257``). The output is
memory-mapped rather than loaded into RAM in :mod:`looped_moe_gpt2.data.dataset`, so this module
processes text in chunks to keep peak memory bounded regardless of corpus size — important for
running comfortably on a single consumer GPU's host machine.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tiktoken
from tqdm import tqdm

logger = logging.getLogger(__name__)

_GPT2_EOT_TOKEN_ID = 50256  # tiktoken "gpt2" encoding's end-of-text token.


def tokenize_text_file(
    input_path: Path,
    output_path: Path,
    chunk_size_chars: int = 10_000_000,
    encoding_name: str = "gpt2",
) -> int:
    """Tokenize a UTF-8 text file into a flat uint16 binary token array.

    Args:
        input_path: Path to a UTF-8-encoded plain text file.
        output_path: Path to write the resulting ``.bin`` file (raw uint16 little-endian array).
        chunk_size_chars: Number of characters read and encoded per chunk, bounding peak memory.
        encoding_name: tiktoken encoding name; must match ``ModelConfig.vocab_size`` downstream
            (the default "gpt2" encoding has vocab_size=50257).

    Returns:
        Total number of tokens written.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        ValueError: If the input file is empty.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input text file not found: {input_path}")

    encoder = tiktoken.get_encoding(encoding_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_size_bytes = input_path.stat().st_size
    if file_size_bytes == 0:
        raise ValueError(f"Input text file is empty: {input_path}")

    total_tokens = 0
    with (
        input_path.open("r", encoding="utf-8") as text_file,
        output_path.open("wb") as out_file,
        tqdm(
            total=file_size_bytes, unit="B", unit_scale=True, desc=f"Tokenizing {input_path.name}"
        ) as progress,
    ):
        while True:
            chunk = text_file.read(chunk_size_chars)
            if not chunk:
                break
            token_ids = encoder.encode_ordinary(chunk)
            token_ids.append(_GPT2_EOT_TOKEN_ID)
            token_array = np.array(token_ids, dtype=np.uint16)
            out_file.write(token_array.tobytes())
            total_tokens += len(token_array)
            progress.update(len(chunk.encode("utf-8")))

    logger.info("Tokenized %s -> %s (%d tokens)", input_path, output_path, total_tokens)
    return total_tokens
