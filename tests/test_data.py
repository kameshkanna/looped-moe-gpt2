"""Unit tests for tokenization and the memory-mapped dataset."""

import time
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from looped_moe_gpt2.data.dataset import MemmapTokenDataset, RandomOffsetSampler
from looped_moe_gpt2.data.tokenize import tokenize_text_file


@pytest.fixture
def tiny_text_file(tmp_path: Path) -> Path:
    text_path = tmp_path / "sample.txt"
    text_path.write_text("Hello world. " * 200, encoding="utf-8")
    return text_path


def test_tokenize_text_file_produces_nonempty_output(tiny_text_file: Path, tmp_path: Path) -> None:
    output_path = tmp_path / "sample.bin"
    num_tokens = tokenize_text_file(tiny_text_file, output_path)
    assert output_path.exists()
    assert num_tokens > 0
    on_disk_tokens = np.fromfile(output_path, dtype=np.uint16)
    assert len(on_disk_tokens) == num_tokens


def test_tokenize_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        tokenize_text_file(tmp_path / "does_not_exist.txt", tmp_path / "out.bin")


def test_tokenize_empty_file_raises(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        tokenize_text_file(empty_file, tmp_path / "out.bin")


def test_dataset_getitem_shapes_and_shift(tmp_path: Path) -> None:
    token_array = np.arange(100, dtype=np.uint16)
    bin_path = tmp_path / "tokens.bin"
    token_array.tofile(bin_path)

    dataset = MemmapTokenDataset(bin_path, seq_len=10)
    input_ids, targets = dataset[0]
    assert input_ids.shape == (10,)
    assert targets.shape == (10,)
    assert (targets == input_ids + 1).all()


def test_dataset_length(tmp_path: Path) -> None:
    token_array = np.arange(100, dtype=np.uint16)
    bin_path = tmp_path / "tokens.bin"
    token_array.tofile(bin_path)
    dataset = MemmapTokenDataset(bin_path, seq_len=10)
    assert len(dataset) == 90


def test_dataset_too_short_raises(tmp_path: Path) -> None:
    token_array = np.arange(5, dtype=np.uint16)
    bin_path = tmp_path / "tokens.bin"
    token_array.tofile(bin_path)
    with pytest.raises(ValueError, match="too short"):
        MemmapTokenDataset(bin_path, seq_len=10)


def test_dataset_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        MemmapTokenDataset(tmp_path / "missing.bin", seq_len=10)


def test_random_offset_sampler_yields_requested_count(tmp_path: Path) -> None:
    token_array = np.arange(1000, dtype=np.uint16)
    bin_path = tmp_path / "tokens.bin"
    token_array.tofile(bin_path)
    dataset = MemmapTokenDataset(bin_path, seq_len=10)

    sampler = RandomOffsetSampler(dataset, num_samples=50)
    indices = list(sampler)
    assert len(indices) == 50
    assert len(sampler) == 50
    assert all(0 <= i < len(dataset) for i in indices)


def test_random_offset_sampler_does_not_materialize_full_permutation(tmp_path: Path) -> None:
    """Regression test for the startup stall this sampler was written to fix: a plain
    DataLoader(shuffle=True) over a MemmapTokenDataset calls torch.randperm(len(dataset)), which
    was measured to take ~200 seconds and several GB of RAM on a ~560M-sample real corpus before
    the first training batch could be fetched. RandomOffsetSampler must stay fast regardless of
    how large the dataset claims to be, since it never allocates an index array proportional to
    dataset length -- only a fixed-size iterator of `num_samples` independent draws."""
    token_array = np.arange(2000, dtype=np.uint16)
    bin_path = tmp_path / "tokens.bin"
    token_array.tofile(bin_path)
    dataset = MemmapTokenDataset(bin_path, seq_len=10)

    # Simulate a dataset claiming to have hundreds of millions of samples (as a large real
    # corpus's MemmapTokenDataset would) without needing to write a multi-GB file for the test.
    huge_len = 560_000_000
    sampler = RandomOffsetSampler.__new__(RandomOffsetSampler)
    sampler.num_samples = 8
    sampler.data_len = huge_len

    start = time.time()
    indices = list(sampler)
    elapsed = time.time() - start

    assert len(indices) == 8
    assert all(0 <= i < huge_len for i in indices)
    assert elapsed < 1.0, f"Sampling took {elapsed:.2f}s -- expected O(1), independent of data_len."


def test_dataloader_with_random_offset_sampler_fetches_batch(tmp_path: Path) -> None:
    token_array = np.arange(500, dtype=np.uint16)
    bin_path = tmp_path / "tokens.bin"
    token_array.tofile(bin_path)
    dataset = MemmapTokenDataset(bin_path, seq_len=16)

    sampler = RandomOffsetSampler(dataset, num_samples=len(dataset))
    loader = DataLoader(dataset, batch_size=4, sampler=sampler, shuffle=False, drop_last=True)

    input_ids, targets = next(iter(loader))
    assert input_ids.shape == (4, 16)
    assert targets.shape == (4, 16)
