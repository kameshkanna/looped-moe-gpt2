"""Memory-mapped token dataset for language model pretraining.

Loads a flat uint16 token array (produced by :mod:`looped_moe_gpt2.data.tokenize`) via
``numpy.memmap`` so the full tokenized corpus never needs to fit in RAM — the OS page cache
handles paging, which matters when running on a single consumer GPU's host machine alongside
GPU memory constraints from the model itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class MemmapTokenDataset(Dataset):
    """A fixed-length-context language modeling dataset backed by a memory-mapped token array.

    Each item is a contiguous ``(input_ids, target_ids)`` pair of length ``seq_len``, with
    ``target_ids`` being ``input_ids`` shifted by one position (standard next-token prediction).

    NOTE: ``__len__`` returns nearly the full token count, since every token offset is a valid
    sample start -- for a large corpus this is hundreds of millions. Do NOT use this dataset
    with ``DataLoader(shuffle=True)``: PyTorch's default ``RandomSampler`` would materialize a
    ``torch.randperm`` over the ENTIRE length up front, which is exactly the pathology
    :class:`RandomOffsetSampler` below exists to avoid (observed in practice: a ~200 second
    stall and multi-GB RAM spike on a ~560M-sample dataset before the first batch was even
    fetched). Always pair this dataset with :class:`RandomOffsetSampler` for training.

    Args:
        token_bin_path: Path to a flat uint16 binary token array (see
            :func:`looped_moe_gpt2.data.tokenize.tokenize_text_file`).
        seq_len: Context length of each sample.

    Raises:
        FileNotFoundError: If ``token_bin_path`` does not exist.
        ValueError: If the token array is too short to produce even one sample of length
            ``seq_len + 1``.
    """

    def __init__(self, token_bin_path: Path, seq_len: int) -> None:
        if not token_bin_path.exists():
            raise FileNotFoundError(f"Token binary file not found: {token_bin_path}")

        self.seq_len = seq_len
        self.tokens = np.memmap(token_bin_path, dtype=np.uint16, mode="r")

        if len(self.tokens) < seq_len + 1:
            raise ValueError(
                f"Token array length ({len(self.tokens)}) is too short to produce a sample of "
                f"seq_len+1 ({seq_len + 1})."
            )

    def __len__(self) -> int:
        """Number of non-overlapping-start samples available."""
        return len(self.tokens) - self.seq_len

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch one ``(input_ids, target_ids)`` pair starting at ``index``.

        Args:
            index: Starting position in the token array.

        Returns:
            A tuple of two ``torch.LongTensor``s, each of shape ``(seq_len,)``.
        """
        chunk = torch.from_numpy(self.tokens[index : index + self.seq_len + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]


class RandomOffsetSampler(Sampler[int]):
    """Draws random sample-start indices on the fly, without ever materializing a permutation.

    A plain ``DataLoader(dataset, shuffle=True)`` uses ``torch.utils.data.RandomSampler``,
    which calls ``torch.randperm(len(dataset))`` up front -- for a :class:`MemmapTokenDataset`
    over a large corpus, ``len(dataset)`` is close to the total token count (every offset is a
    valid start), so this permutation can be hundreds of millions of elements: slow to generate
    and several GB of RAM just for the index list, all before training's first batch. This
    sampler instead draws each index independently via ``torch.randint`` at iteration time,
    which is O(1) per sample and never materializes more than one batch's worth of indices.

    This trades perfect epoch-level coverage (each token offset visited exactly once per epoch)
    for O(1) memory and startup cost -- standard practice for large-corpus LM pretraining
    (e.g. nanoGPT's training loop samples offsets the same way), since a training run rarely
    completes even one full epoch over a large corpus anyway.

    Args:
        data_source: The dataset to sample from; only ``len(data_source)`` is used.
        num_samples: Number of indices to yield per iteration (typically set to
            ``len(data_source)`` so one call to ``iter()`` looks like one epoch's worth of draws,
            though :class:`~looped_moe_gpt2.train.trainer.Trainer` cycles the loader indefinitely
            regardless).
    """

    def __init__(self, data_source: Dataset, num_samples: int) -> None:
        self.num_samples = num_samples
        self.data_len = len(data_source)  # type: ignore[arg-type]

    def __iter__(self) -> Iterator[int]:
        """Yield ``self.num_samples`` independently-drawn random indices."""
        for _ in range(self.num_samples):
            yield int(torch.randint(0, self.data_len, (1,)).item())

    def __len__(self) -> int:
        return self.num_samples
