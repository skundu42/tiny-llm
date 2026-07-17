import torch

from tinyllm.model import RMSNorm, apply_rope, precompute_rope


def test_rmsnorm_unit_rms():
    torch.manual_seed(0)
    x = torch.randn(4, 16, 64) * 3.0
    y = RMSNorm(64)(x)
    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rmsnorm_dtype_preserved():
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    assert RMSNorm(8)(x).dtype == torch.bfloat16


def test_rope_preserves_norm():
    torch.manual_seed(0)
    cos, sin = precompute_rope(64, 128, 10000.0)
    x = torch.randn(2, 4, 128, 64)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-4)


def test_rope_relative_property():
    """q.k after RoPE depends only on relative distance."""
    torch.manual_seed(0)
    cos, sin = precompute_rope(64, 256, 10000.0)
    q = torch.randn(1, 1, 1, 64)
    k = torch.randn(1, 1, 1, 64)

    def score(qi: int, kj: int) -> float:
        qr = apply_rope(q, cos[qi : qi + 1], sin[qi : qi + 1])
        kr = apply_rope(k, cos[kj : kj + 1], sin[kj : kj + 1])
        return (qr * kr).sum().item()

    assert abs(score(5, 9) - score(105, 109)) < 1e-3
    assert abs(score(5, 9) - score(9, 5)) > 1e-4  # direction matters


def test_rope_position_zero_is_identity():
    cos, sin = precompute_rope(64, 8, 10000.0)
    x = torch.randn(1, 2, 1, 64)
    assert torch.allclose(apply_rope(x, cos[:1], sin[:1]), x, atol=1e-6)


import torch.nn.functional as F

from tinyllm.config import ModelConfig
from tinyllm.model import Attention, KVCache


def _small_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=256, n_layer=2, n_head=4, n_kv_head=2, d_model=64, d_ff=128, seq_len=32
    )


def reference_attention(q, k, v):
    """Naive causal attention: q,k,v are (B, H, T, hd) with equal T."""
    B, H, T, hd = q.shape
    scores = (q @ k.transpose(-2, -1)) / (hd**0.5)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    return scores.softmax(-1) @ v


def test_sdpa_matches_reference():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 16, 32) for _ in range(3))
    ours = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = reference_attention(q, k, v)
    assert torch.allclose(ours, ref, atol=1e-5)


def test_attention_shapes_and_grad():
    torch.manual_seed(0)
    cfg = _small_cfg()
    from tinyllm.model import precompute_rope

    attn = Attention(cfg, layer_idx=0)
    cos, sin = precompute_rope(cfg.head_dim, cfg.seq_len, cfg.rope_theta)
    x = torch.randn(2, 16, cfg.d_model, requires_grad=True)
    y = attn(x, cos[:16], sin[:16])
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None


def test_kv_cache_matches_full_forward():
    """Prefill+decode through the cache must equal one full no-cache pass."""
    torch.manual_seed(0)
    cfg = _small_cfg()
    from tinyllm.model import precompute_rope

    attn = Attention(cfg, layer_idx=0).eval()
    cos, sin = precompute_rope(cfg.head_dim, cfg.seq_len, cfg.rope_theta)
    x = torch.randn(1, 10, cfg.d_model)

    with torch.no_grad():
        full = attn(x, cos[:10], sin[:10])

        cache = KVCache(1, 1, cfg.n_kv_head, 32, cfg.head_dim, x.device, x.dtype)
        pre = attn(x[:, :7], cos[:7], sin[:7], cache=cache)
        cache.advance(7)
        outs = [pre]
        for t in range(7, 10):
            o = attn(x[:, t : t + 1], cos[t : t + 1], sin[t : t + 1], cache=cache)
            cache.advance(1)
            outs.append(o)
    stepped = torch.cat(outs, dim=1)
    assert torch.allclose(full, stepped, atol=1e-5)


def test_attention_is_causal():
    torch.manual_seed(0)
    cfg = _small_cfg()
    from tinyllm.model import precompute_rope

    attn = Attention(cfg, layer_idx=0).eval()
    cos, sin = precompute_rope(cfg.head_dim, cfg.seq_len, cfg.rope_theta)
    x = torch.randn(1, 12, cfg.d_model)
    x2 = x.clone()
    x2[:, 6:] = torch.randn(1, 6, cfg.d_model)
    with torch.no_grad():
        y, y2 = attn(x, cos[:12], sin[:12]), attn(x2, cos[:12], sin[:12])
    assert torch.allclose(y[:, :6], y2[:, :6], atol=1e-5)
