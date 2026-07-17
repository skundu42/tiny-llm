import types

import pytest
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


def test_kv_cache_rejects_chunked_decode():
    cfg = _small_cfg()
    attn = Attention(cfg, layer_idx=0).eval()
    cos, sin = precompute_rope(cfg.head_dim, cfg.seq_len, cfg.rope_theta)
    cache = KVCache(1, 1, cfg.n_kv_head, 32, cfg.head_dim, torch.device("cpu"), torch.float32)
    cache.advance(1)
    with pytest.raises(ValueError, match="chunked prefill"):
        attn(torch.randn(1, 2, cfg.d_model), cos[1:3], sin[1:3], cache=cache)


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


from tinyllm.model import TinyLLM


def test_param_counts_exact():
    from tinyllm.config import MODEL_PRESETS

    with torch.device("meta"):
        d26 = TinyLLM(MODEL_PRESETS["d26"])
        smoke = TinyLLM(MODEL_PRESETS["smoke"])
    assert d26.num_params() == 489_297_408
    assert smoke.num_params() == 13_111_296


def test_tied_embeddings():
    model = TinyLLM(_small_cfg())
    assert model.lm_head.weight is model.embed.weight


def test_forward_loss_near_uniform_at_init():
    import math

    torch.manual_seed(0)
    cfg = _small_cfg()
    model = TinyLLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 17))
    logits, loss = model(x[:, :-1], targets=x[:, 1:])
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 1.0


def test_model_is_causal():
    torch.manual_seed(0)
    cfg = _small_cfg()
    model = TinyLLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 12))
    x2 = x.clone()
    x2[:, 6:] = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        y, _ = model(x)
        y2, _ = model(x2)
    assert torch.allclose(y[:, :6], y2[:, :6], atol=1e-4)


def test_generate_greedy_matches_uncached_argmax():
    torch.manual_seed(0)
    cfg = _small_cfg()
    model = TinyLLM(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 5))

    out = model.generate(prompt.clone(), max_new_tokens=8, temperature=0.0)

    seq = prompt.clone()
    with torch.no_grad():
        for _ in range(8):
            logits, _ = model(seq)
            seq = torch.cat([seq, logits[:, -1:].argmax(-1)], dim=1)
    assert torch.equal(out, seq)


def test_generate_respects_eos():
    torch.manual_seed(0)
    cfg = _small_cfg()
    model = TinyLLM(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 3))
    greedy = model.generate(prompt.clone(), max_new_tokens=10, temperature=0.0)
    eos = greedy[0, 3].item()  # first generated token
    out = model.generate(prompt.clone(), max_new_tokens=10, temperature=0.0, eos_id=eos)
    assert out.shape[1] == 4  # stops immediately after emitting eos


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"max_new_tokens": -1}, "max_new_tokens"),
        ({"temperature": -1.0}, "temperature"),
        ({"temperature": float("nan")}, "temperature"),
        ({"top_k": 0}, "top_k"),
        ({"top_k": 1.5}, "top_k"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_p": "0.9"}, "top_p"),
        ({"top_p": 1.1}, "top_p"),
        ({"eos_id": 256}, "eos_id"),
    ],
)
def test_generate_rejects_invalid_controls(kwargs, message):
    model = TinyLLM(_small_cfg())
    prompt = torch.ones((1, 1), dtype=torch.long)
    controls = dict(kwargs)
    max_new_tokens = controls.pop("max_new_tokens", 1)
    with pytest.raises(ValueError, match=message):
        model.generate(prompt, max_new_tokens=max_new_tokens, **controls)


@pytest.mark.parametrize(
    "prompt",
    [
        torch.ones((1, 1), dtype=torch.float32),
        torch.tensor([[-1]], dtype=torch.long),
        torch.tensor([[256]], dtype=torch.long),
    ],
)
def test_generate_rejects_invalid_prompt_tokens(prompt):
    with pytest.raises(ValueError, match="idx"):
        TinyLLM(_small_cfg()).generate(prompt, max_new_tokens=1)


def test_generate_zero_tokens_avoids_forward_and_preserves_mode():
    model = TinyLLM(_small_cfg()).train()
    prompt = torch.ones((1, 3), dtype=torch.long)
    calls = 0

    def count_calls(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    handle = model.register_forward_hook(count_calls)
    out = model.generate(prompt, max_new_tokens=0)
    handle.remove()
    assert torch.equal(out, prompt)
    assert calls == 0
    assert model.training


def test_generate_does_not_decode_after_final_token():
    model = TinyLLM(_small_cfg()).eval()
    prompt = torch.ones((1, 3), dtype=torch.long)
    calls = 0

    def count_calls(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    handle = model.register_forward_hook(count_calls)
    model.generate(prompt, max_new_tokens=1, temperature=0.0)
    handle.remove()
    assert calls == 1  # prefill only; the sampled token needs no unused decode
    assert not model.training


def test_generate_tracks_eos_per_batch_row():
    model = TinyLLM(_small_cfg()).eval()
    calls = 0

    def scripted_forward(self, idx, targets=None, cache=None):
        nonlocal calls
        logits = torch.full((2, 1, self.cfg.vocab_size), -100.0)
        if calls == 0:
            logits[0, 0, 7] = 100.0  # row 0 finishes immediately
            logits[1, 0, 8] = 100.0
        else:
            logits[0, 0, 9] = 100.0  # must be replaced by EOS for finished row 0
            logits[1, 0, 7] = 100.0  # row 1 now finishes
        calls += 1
        return logits, None

    model.forward = types.MethodType(scripted_forward, model)
    prompt = torch.ones((2, 1), dtype=torch.long)
    out = model.generate(prompt, max_new_tokens=4, temperature=0.0, eos_id=7)
    assert out.tolist() == [[1, 7, 7], [1, 8, 7]]
    assert calls == 2


def test_residual_projections_scaled_init():
    cfg = _small_cfg()
    model = TinyLLM(cfg)
    expected_std = 0.02 / (2 * cfg.n_layer) ** 0.5
    assert abs(model.blocks[0].attn.wo.weight.std().item() - expected_std) < expected_std * 0.2
    assert abs(model.blocks[0].mlp.w_down.weight.std().item() - expected_std) < expected_std * 0.2
