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
