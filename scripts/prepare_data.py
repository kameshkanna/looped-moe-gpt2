#!/usr/bin/env python
"""CLI: download-free data preparation -- tokenize a local train/val text file pair.

Usage:
    python scripts/prepare_data.py --train-text path/to/train.txt --val-text path/to/val.txt \
        --output-dir data/

If you don't have a corpus yet, TinyStories or a small OpenWebText/FineWeb subset are reasonable
choices for a single-4060 sanity-check run; point --train-text/--val-text at plain-text files.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from looped_moe_gpt2.data.tokenize import tokenize_text_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Parse CLI arguments and tokenize the given train/val text files."""
    parser = argparse.ArgumentParser(description="Tokenize train/val text files for pretraining.")
    parser.add_argument("--train-text", type=Path, required=True, help="Path to raw training text.")
    parser.add_argument("--val-text", type=Path, required=True, help="Path to raw validation text.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for output .bin files.")
    parser.add_argument("--encoding", type=str, default="gpt2", help="tiktoken encoding name.")
    args = parser.parse_args()

    train_tokens = tokenize_text_file(
        args.train_text, args.output_dir / "train.bin", encoding_name=args.encoding
    )
    val_tokens = tokenize_text_file(
        args.val_text, args.output_dir / "val.bin", encoding_name=args.encoding
    )
    logger.info("Done. train=%d tokens, val=%d tokens", train_tokens, val_tokens)


if __name__ == "__main__":
    main()
