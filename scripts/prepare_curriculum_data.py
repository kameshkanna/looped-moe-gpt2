#!/usr/bin/env python
"""CLI: stream and tokenize the two curriculum-phase corpora into separate .bin files.

Downloads via HuggingFace `datasets` streaming mode (no full-dataset download required --
pulls only as many documents as needed to hit the requested token budget), then tokenizes with
the same GPT-2 BPE encoding used everywhere else in this repo, writing flat uint16 binaries
compatible with :class:`looped_moe_gpt2.data.dataset.MemmapTokenDataset`.

Usage:
    python scripts/prepare_curriculum_data.py --phase general --target-tokens 500000000 \
        --output data/general.bin
    python scripts/prepare_curriculum_data.py --phase math --target-tokens 200000000 \
        --output data/math.bin

Each phase also needs a held-out validation slice; pass --val-output and --val-tokens to write
one alongside the training file (drawn from a disjoint slice of the same streamed dataset).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_GPT2_EOT_TOKEN_ID = 50256


def _iter_general_texts() -> Iterator[str]:
    """Stream document text from the FineWeb-Edu 10BT sample.

    Yields:
        Plain document text, one per FineWeb-Edu record.
    """
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
    )
    for record in dataset:
        yield record["text"]


def _iter_math_texts() -> Iterator[str]:
    """Stream problem+solution text from OpenMathInstruct-2, formatted as plain documents.

    Yields:
        A plain-text document combining each problem and its generated chain-of-thought
        solution, so the base LM learns the step-by-step-reasoning pattern as ordinary next-
        token prediction (no special instruction-formatting tokens are introduced here).
    """
    dataset = load_dataset("nvidia/OpenMathInstruct-2", split="train", streaming=True)
    for record in dataset:
        yield f"Problem: {record['problem']}\nSolution: {record['generated_solution']}"


_PHASE_ITERATORS = {
    "general": _iter_general_texts,
    "math": _iter_math_texts,
}


def stream_tokenize(
    text_iterator: Iterator[str],
    output_path: Path,
    target_tokens: int,
    encoding_name: str = "gpt2",
) -> int:
    """Stream documents, tokenize each, and write to a flat uint16 binary until the token budget is hit.

    Args:
        text_iterator: Yields document strings (e.g. from :func:`_iter_general_texts`).
        output_path: Path to write the resulting ``.bin`` file.
        target_tokens: Stop once at least this many tokens have been written.
        encoding_name: tiktoken encoding name.

    Returns:
        Total number of tokens written.

    Raises:
        ValueError: If ``target_tokens`` is not positive.
    """
    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be positive, got {target_tokens}.")

    encoder = tiktoken.get_encoding(encoding_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    with output_path.open("wb") as out_file, tqdm(total=target_tokens, unit="tok", unit_scale=True, desc=f"Tokenizing -> {output_path.name}") as progress:
        for text in text_iterator:
            if not text:
                continue
            token_ids = encoder.encode_ordinary(text)
            token_ids.append(_GPT2_EOT_TOKEN_ID)
            token_array = np.array(token_ids, dtype=np.uint16)
            out_file.write(token_array.tobytes())
            total_tokens += len(token_array)
            progress.update(len(token_array))
            if total_tokens >= target_tokens:
                break

    logger.info("Wrote %s (%d tokens)", output_path, total_tokens)
    return total_tokens


def main() -> None:
    """Parse CLI arguments and stream-tokenize the requested curriculum phase."""
    parser = argparse.ArgumentParser(description="Stream and tokenize a curriculum-phase corpus.")
    parser.add_argument("--phase", choices=sorted(_PHASE_ITERATORS), required=True)
    parser.add_argument("--target-tokens", type=int, required=True, help="Training tokens to write.")
    parser.add_argument("--output", type=Path, required=True, help="Training .bin output path.")
    parser.add_argument("--val-tokens", type=int, default=None, help="Validation tokens to write (optional).")
    parser.add_argument("--val-output", type=Path, default=None, help="Validation .bin output path.")
    args = parser.parse_args()

    if bool(args.val_tokens) != bool(args.val_output):
        raise ValueError("--val-tokens and --val-output must be given together, or not at all.")

    # Write ONE combined stream covering target_tokens + val_tokens, then split the tail off as
    # validation locally (in-memory, on the already-downloaded file) -- rather than re-invoking
    # the streaming iterator a second time with a document skip-count. The latter was tried
    # first and found to be a real problem: restarting a HuggingFace streaming dataset from
    # scratch and skipping millions of documents one at a time in a Python loop is extremely
    # slow (no tokenization work happens during the skip, just iteration/re-fetch overhead) --
    # observed to stall for many minutes with zero bytes written on a 1.2B-token corpus. Splitting
    # a local file after the fact requires zero additional network activity and is effectively
    # instant by comparison.
    combined_target = args.target_tokens + (args.val_tokens or 0)
    stream_tokenize(_PHASE_ITERATORS[args.phase](), args.output, combined_target)

    if args.val_tokens:
        _split_val_tail(args.output, args.val_output, args.val_tokens)


def _split_val_tail(train_path: Path, val_path: Path, val_tokens: int) -> None:
    """Move the last ``val_tokens`` tokens of a tokenized binary into a separate validation file.

    Args:
        train_path: Path to the combined tokenized binary (train + held-out val, concatenated).
            Overwritten in place with only the training portion after this call.
        val_path: Path to write the held-out validation tokens to.
        val_tokens: Number of trailing tokens to hold out as validation.

    Raises:
        ValueError: If ``train_path`` has fewer than ``val_tokens`` tokens.
    """
    tokens = np.memmap(train_path, dtype=np.uint16, mode="r")
    if len(tokens) < val_tokens:
        raise ValueError(
            f"{train_path} has only {len(tokens)} tokens, fewer than the requested "
            f"val_tokens ({val_tokens})."
        )
    val_slice = np.array(tokens[len(tokens) - val_tokens :])
    train_slice = np.array(tokens[: len(tokens) - val_tokens])
    del tokens  # release the memmap before overwriting train_path

    val_slice.tofile(val_path)
    train_slice.tofile(train_path)
    logger.info(
        "Split %s: %d training tokens retained, %d held out to %s",
        train_path,
        len(train_slice),
        len(val_slice),
        val_path,
    )


if __name__ == "__main__":
    main()
