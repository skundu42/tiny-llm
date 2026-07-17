import json
import os

import numpy as np
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


def test_sampler_crosses_shard_correctly(tmp_path):
    # tokens encode their own position -> any crop must be contiguous
    _write(tmp_path, "train", [i % 1000 for i in range(3000)], shard_tokens=1000)
    ds = TokenShards(str(tmp_path), "train")
    rng = np.random.default_rng(7)
    for _ in range(20):
        x, y = ds.sample_batch(2, 64, rng)
        diffs = (x[:, 1:] - x[:, :-1]) % 1000
        assert (diffs == 1).all()
