import torch

from tinyllm.config import ModelConfig
from tinyllm.eval_hellaswag import ending_losses
from tinyllm.model import TinyLLM


def test_ending_losses_prefers_repeated_pattern():
    """A model overfit to a cyclic sequence must assign lower loss to the
    continuation that follows the cycle than to a broken one."""
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=32, n_layer=2, n_head=2, n_kv_head=1,
                      d_model=64, d_ff=128, seq_len=256)
    model = TinyLLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    data = torch.arange(64).remainder(8).repeat(4)[None, :]  # 0..7 cycle
    for _ in range(150):
        logits, loss = model(data[:, :-1], targets=data[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()

    ctx = [0, 1, 2, 3]
    good = [4, 5, 6, 7]
    bad = [7, 2, 0, 5]
    losses = ending_losses(model, ctx, [good, bad], device="cpu")
    assert losses[0][1] < losses[1][1]  # mean loss lower for the true continuation
    assert losses[0][0] < losses[1][0]  # sum loss too


def test_ending_losses_lengths_dont_crash():
    cfg = ModelConfig(vocab_size=32, n_layer=1, n_head=2, n_kv_head=1,
                      d_model=32, d_ff=64, seq_len=64)
    model = TinyLLM(cfg)
    out = ending_losses(model, [1, 2], [[3], [4, 5, 6]], device="cpu")
    assert len(out) == 2
