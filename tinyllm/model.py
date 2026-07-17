"""TinyLLM: a from-scratch decoder-only transformer.

RMSNorm pre-norm, RoPE, grouped-query attention with QK-norm, SwiGLU,
tied embeddings, no biases anywhere.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def precompute_rope(
    head_dim: int, max_seq_len: int, theta: float, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cos/sin tables, each (max_seq_len, head_dim // 2), float32."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate (B, H, T, head_dim) by the tables' T positions (NeoX half-split)."""
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    out = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.to(x.dtype)


class KVCache:
    """Preallocated per-layer K/V cache for autoregressive decoding. Usage
    contract: one multi-token prefill from an empty cache (pos == 0), then
    exactly one token per forward call. A multi-token forward with pos > 0
    would attend without any causal mask and leak future tokens within the
    chunk; Attention asserts against it.
    """

    def __init__(self, n_layer, batch_size, n_kv_head, max_seq_len, head_dim, device, dtype):
        self.k = torch.zeros(n_layer, batch_size, n_kv_head, max_seq_len, head_dim,
                             device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.pos = 0

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        t = k.size(2)
        self.k[layer, :, :, self.pos : self.pos + t] = k
        self.v[layer, :, :, self.pos : self.pos + t] = v
        return self.k[layer, :, :, : self.pos + t], self.v[layer, :, :, : self.pos + t]

    def advance(self, t: int) -> None:
        self.pos += t


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head, self.n_kv_head, self.head_dim = cfg.n_head, cfg.n_kv_head, cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.n_head * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_head * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_head * cfg.head_dim, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)

    def forward(self, x, cos, sin, cache: "KVCache | None" = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            assert T == 1 or cache.pos == 0, (
                "chunked prefill unsupported: q_len > 1 requires an empty cache"
            )
            k, v = cache.update(self.layer_idx, k, v)
        rep = self.n_head // self.n_kv_head
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=q.size(2) == k.size(2))
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        return self.wo(y)
