import torch

from tinylm.config import ModelConfig, TrainConfig
from tinylm.model import TinyLM
from tinylm.muon import Muon, build_optimizers, zeropower_via_newtonschulz5


def test_newton_schulz_orthogonalizes():
    torch.manual_seed(0)
    G = torch.randn(64, 32)
    X = zeropower_via_newtonschulz5(G, steps=5).float()
    svals = torch.linalg.svdvals(X)
    assert svals.max() < 1.6 and svals.min() > 0.3  # quintic NS: svals ~ [0.7, 1.2]


def test_newton_schulz_handles_wide_matrices():
    torch.manual_seed(0)
    G = torch.randn(32, 64)
    X = zeropower_via_newtonschulz5(G, steps=5)
    assert X.shape == G.shape


def test_newton_schulz_rejects_non_matrix():
    import pytest

    with pytest.raises(ValueError, match="two-dimensional"):
        zeropower_via_newtonschulz5(torch.randn(8))


def test_muon_converges_on_regression():
    torch.manual_seed(0)
    W = torch.nn.Parameter(torch.zeros(16, 8))
    W_true = torch.randn(16, 8)
    x = torch.randn(256, 8)
    y = x @ W_true.T
    opt = Muon([W], lr=0.02, momentum=0.95)
    first = None
    for _ in range(300):
        loss = ((x @ W.T - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first or loss.item()
    assert loss.item() < first / 10


def test_build_optimizers_split():
    cfg = ModelConfig(vocab_size=256, n_layer=2, n_head=4, n_kv_head=2,
                      d_model=64, d_ff=128, seq_len=32)
    model = TinyLM(cfg)
    muon, adamw = build_optimizers(model, TrainConfig())
    n_muon = sum(len(g["params"]) for g in muon.param_groups)
    assert n_muon == cfg.n_layer * 7  # wq wk wv wo gate up down per layer
    embed_group, norm_group = adamw.param_groups
    assert len(embed_group["params"]) == 1
    # per layer: attn_norm, mlp_norm, q_norm, k_norm; plus final_norm
    assert len(norm_group["params"]) == cfg.n_layer * 4 + 1
    assert embed_group["weight_decay"] > 0 and norm_group["weight_decay"] == 0.0
    for opt in (muon, adamw):
        for g in opt.param_groups:
            assert "initial_lr" in g
    # tied lm_head must not be double-counted
    total = n_muon + len(embed_group["params"]) + len(norm_group["params"])
    assert total == len(list(model.parameters()))


def test_muon_state_dict_roundtrip():
    torch.manual_seed(0)
    W = torch.nn.Parameter(torch.randn(8, 8))
    opt = Muon([W], lr=0.02)
    (W.sum()).backward()
    opt.step()
    sd = opt.state_dict()
    opt2 = Muon([W], lr=0.02)
    opt2.load_state_dict(sd)
    buf = list(opt2.state.values())[0]["momentum_buffer"]
    assert torch.allclose(buf, list(opt.state.values())[0]["momentum_buffer"])
