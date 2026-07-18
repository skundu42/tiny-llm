"""Uint16 token shards on disk + memory-mapped random-crop sampling."""
from __future__ import annotations

import json
import os
from collections.abc import Sequence

import numpy as np
import torch

INDEX_NAME = "index.json"


class ShardWriter:
    """Append token ids into fixed-size uint16 .bin shards plus index.json.

    total_written counts tokens accepted by write(); they reach disk only at close().
    Re-opening an existing split raises.
    """

    def __init__(self, out_dir: str, split: str, shard_tokens: int) -> None:
        if shard_tokens <= 0:
            raise ValueError(f"shard_tokens must be positive, got {shard_tokens}")
        os.makedirs(out_dir, exist_ok=True)
        index_path = os.path.join(out_dir, INDEX_NAME)
        if os.path.exists(index_path):
            with open(index_path) as f:
                existing = json.load(f)
            if existing.get("splits", {}).get(split):
                raise ValueError(
                    f"{out_dir} already contains a '{split}' split; use a fresh directory"
                )
        self.out_dir, self.split, self.shard_tokens = out_dir, split, shard_tokens
        self.buffer = np.empty(shard_tokens, dtype=np.uint16)
        self.fill = 0
        self.entries: list[dict] = []
        self.total_written = 0

    def write(self, ids: Sequence[int]) -> None:
        raw = np.asarray(ids)
        if raw.ndim != 1:
            raise ValueError(f"ids must be one-dimensional, got shape {raw.shape}")
        if raw.size == 0:
            return
        if not np.issubdtype(raw.dtype, np.integer):
            raise TypeError(f"ids must contain integers, got dtype {raw.dtype}")
        if raw.min() < 0 or raw.max() > np.iinfo(np.uint16).max:
            raise ValueError("token ids must be in the uint16 range [0, 65535]")
        arr = raw.astype(np.uint16, copy=False)
        self.total_written += len(arr)
        while len(arr) > 0:
            n = min(len(arr), self.shard_tokens - self.fill)
            self.buffer[self.fill : self.fill + n] = arr[:n]
            self.fill += n
            arr = arr[n:]
            if self.fill == self.shard_tokens:
                self._flush()

    def _flush(self) -> None:
        if self.fill == 0:
            return
        name = f"{self.split}_{len(self.entries):05d}.bin"
        self.buffer[: self.fill].tofile(os.path.join(self.out_dir, name))
        self.entries.append({"file": name, "tokens": int(self.fill)})
        self.fill = 0

    def close(self) -> None:
        self._flush()
        path = os.path.join(self.out_dir, INDEX_NAME)
        index = {"dtype": "uint16", "splits": {}}
        if os.path.exists(path):
            with open(path) as f:
                index = json.load(f)
        index["splits"][self.split] = self.entries
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(index, f, indent=1)
        os.replace(tmp, path)


class TokenShards:
    """Memmapped random-crop sampler over the shards of one split."""

    def __init__(self, data_dir: str, split: str) -> None:
        with open(os.path.join(data_dir, INDEX_NAME)) as f:
            index = json.load(f)
        if index.get("dtype") != "uint16":
            raise ValueError(f"unsupported shard dtype: {index.get('dtype')!r}")
        entries = index["splits"][split]
        if not entries:
            raise ValueError(f"split {split!r} contains no shards")
        self.shards = [
            np.memmap(os.path.join(data_dir, e["file"]), dtype=np.uint16, mode="r")
            for e in entries
        ]
        for entry, shard in zip(entries, self.shards):
            if entry["tokens"] != len(shard):
                raise ValueError(
                    f"shard {entry['file']!r} has {len(shard)} tokens, "
                    f"index declares {entry['tokens']}"
                )
        self.sizes = np.array([len(s) for s in self.shards], dtype=np.int64)
        self.total_tokens = int(self.sizes.sum())

    def sample_batch(self, batch_size: int, seq_len: int, rng: np.random.Generator,
                     device: str | torch.device = "cpu"):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        usable = self.sizes - (seq_len + 1)
        positions = np.clip(usable + 1, 0, None)  # count of valid crop offsets per shard
        if positions.sum() == 0:
            raise ValueError("no shard is long enough for this seq_len")
        p = positions.astype(np.float64) / positions.sum()
        shard_ids = rng.choice(len(self.shards), size=batch_size, p=p)
        xs = np.empty((batch_size, seq_len + 1), dtype=np.int64)
        for i, si in enumerate(shard_ids):
            off = int(rng.integers(0, positions[si]))
            xs[i] = self.shards[si][off : off + seq_len + 1].astype(np.int64)
        t = torch.from_numpy(xs)
        x, y = t[:, :-1].contiguous(), t[:, 1:].contiguous()
        if str(device) != "cpu":
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
        return x, y
