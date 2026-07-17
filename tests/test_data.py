import json
import os

import numpy as np
import pytest
import torch

from tinyllm.data import ShardWriter, TokenShards


def _write(tmp_path, split, tokens, shard_tokens=1000):
    w = ShardWriter(str(tmp_path), split, shard_tokens=shard_tokens)
    for i in range(0, len(tokens), 300):
        w.write(tokens[i : i + 300])
    w.close()
    return w


def test_writer_shards_and_index(tmp_path):
    tokens = list(range(2500))
    tokens = [t % 65536 for t in tokens]
    w = _write(tmp_path, "train", tokens)
    assert w.total_written == 2500
    idx = json.load(open(tmp_path / "index.json"))
    entries = idx["splits"]["train"]
    assert [e["tokens"] for e in entries] == [1000, 1000, 500]
    back = np.concatenate(
        [np.fromfile(tmp_path / e["file"], dtype=np.uint16) for e in entries]
    )
    assert back.tolist() == tokens


def test_two_splits_share_index(tmp_path):
    _write(tmp_path, "val", list(range(500)))
    _write(tmp_path, "train", list(range(1500)))
    idx = json.load(open(tmp_path / "index.json"))
    assert set(idx["splits"]) == {"val", "train"}


def test_sampler_shapes_and_shift(tmp_path):
    _write(tmp_path, "train", [i % 50 for i in range(5000)])
    ds = TokenShards(str(tmp_path), "train")
    assert ds.total_tokens == 5000
    rng = np.random.default_rng(0)
    x, y = ds.sample_batch(4, 32, rng)
    assert x.shape == y.shape == (4, 32)
    assert x.dtype == torch.int64
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_sampler_deterministic(tmp_path):
    _write(tmp_path, "train", [i % 50 for i in range(5000)])
    ds = TokenShards(str(tmp_path), "train")
    x1, _ = ds.sample_batch(4, 32, np.random.default_rng([1, 2, 3]))
    x2, _ = ds.sample_batch(4, 32, np.random.default_rng([1, 2, 3]))
    assert torch.equal(x1, x2)


def test_sampler_crops_contiguous_and_all_shards_reachable(tmp_path):
    # tokens encode their global position -> any crop must be a contiguous run
    _write(tmp_path, "train", list(range(3000)), shard_tokens=1000)
    ds = TokenShards(str(tmp_path), "train")
    rng = np.random.default_rng(7)
    seen = set()
    for _ in range(50):
        x, y = ds.sample_batch(2, 64, rng)
        assert ((x[:, 1:] - x[:, :-1]) == 1).all()
        seen.update((x[:, 0] // 1000).tolist())
    assert seen == {0, 1, 2}


def test_sampler_exact_length_shard_is_usable(tmp_path):
    _write(tmp_path, "train", list(range(65)), shard_tokens=1000)
    ds = TokenShards(str(tmp_path), "train")
    x, y = ds.sample_batch(2, 64, np.random.default_rng(0))
    assert torch.equal(x[0], torch.arange(64))
    assert torch.equal(y[0], torch.arange(1, 65))


def test_rewriting_existing_split_raises(tmp_path):
    _write(tmp_path, "val", list(range(500)))
    with pytest.raises(ValueError, match="already contains"):
        ShardWriter(str(tmp_path), "val", shard_tokens=1000)


def test_writer_rejects_non_positive_shard_size(tmp_path):
    with pytest.raises(ValueError, match="shard_tokens must be positive"):
        ShardWriter(str(tmp_path), "train", shard_tokens=0)


@pytest.mark.parametrize(
    "tokens",
    [
        [-1],
        [65536],
        np.array([65536], dtype=np.int64),
        [1.5],
        [[1, 2]],
    ],
)
def test_writer_rejects_invalid_token_ids(tmp_path, tokens):
    w = ShardWriter(str(tmp_path), "train", shard_tokens=1000)
    with pytest.raises((TypeError, ValueError)):
        w.write(tokens)


def test_sampler_rejects_invalid_sizes(tmp_path):
    _write(tmp_path, "train", list(range(5000)))
    ds = TokenShards(str(tmp_path), "train")
    with pytest.raises(ValueError, match="batch_size"):
        ds.sample_batch(0, 32, np.random.default_rng(0))
    with pytest.raises(ValueError, match="seq_len"):
        ds.sample_batch(1, 0, np.random.default_rng(0))
