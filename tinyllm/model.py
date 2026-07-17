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


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, layer_idx)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache: KVCache | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class TinyLLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying
        cos, sin = precompute_rope(cfg.head_dim, cfg.seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        std = 0.02 / math.sqrt(2 * cfg.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.wo.weight, mean=0.0, std=std)
            nn.init.normal_(block.mlp.w_down.weight, mean=0.0, std=std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.weight.numel()
        return n

    def forward(self, idx, targets=None, cache: KVCache | None = None):
        B, T = idx.shape
        pos0 = cache.pos if cache is not None else 0
        cos = self.rope_cos[pos0 : pos0 + T]
        sin = self.rope_sin[pos0 : pos0 + T]
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x, cos, sin, cache)
        if cache is not None:
            cache.advance(T)
        x = self.final_norm(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
            return logits, loss
        if cache is not None:
            return self.lm_head(x[:, -1:, :]), None
        return self.lm_head(x), None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, eos_id=None):
        self.eval()
        B, T = idx.shape
        total = T + max_new_tokens
        assert total <= self.cfg.seq_len, "generation would exceed context length"
        cache = KVCache(
            self.cfg.n_layer, B, self.cfg.n_kv_head, total, self.cfg.head_dim,
            idx.device, self.embed.weight.dtype,
        )
        logits, _ = self(idx, cache=cache)
        for _ in range(max_new_tokens):
            logits_last = logits[:, -1, :]
            if temperature == 0.0:
                next_tok = logits_last.argmax(-1, keepdim=True)
            else:
                logits_last = logits_last / temperature
                if top_k is not None:
                    kth = torch.topk(logits_last, min(top_k, logits_last.size(-1))).values[:, -1:]
                    logits_last = logits_last.masked_fill(logits_last < kth, float("-inf"))
                if top_p is not None:
                    sorted_logits, sorted_idx = torch.sort(logits_last, descending=True)
                    probs = F.softmax(sorted_logits, dim=-1)
                    mask = probs.cumsum(-1) - probs > top_p
                    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
                    logits_last = torch.full_like(logits_last, float("-inf")).scatter(
                        1, sorted_idx, sorted_logits
                    )
                next_tok = torch.multinomial(F.softmax(logits_last, dim=-1), num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
            if eos_id is not None and (next_tok == eos_id).all():
                break
            logits, _ = self(next_tok, cache=cache)
        return idx
