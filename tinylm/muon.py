"""Muon: MomentUm Orthogonalized by Newton-Schulz (Jordan et al., 2024).

Orthogonalizes the momentum-smoothed gradient of each 2-D weight matrix with
a quintic Newton-Schulz iteration, then takes an SGD-like step. Hidden
matrices only; embeddings and norm gains use AdamW (see build_optimizers).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import TrainConfig


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate UV^T for G = USV^T via quintic Newton-Schulz in bfloat16."""
    if G.ndim != 2:
        raise ValueError(f"G must be two-dimensional, got shape {tuple(G.shape)}")
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.mT
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5) -> None:
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                # `g.lerp_` mutates `p.grad` in place (nesterov branch only; the
                # else branch reassigns `g` to `buf` without touching `p.grad`).
                # Safe here because: grad clipping already ran before `step()`,
                # so there's no later reader of the pre-mutation gradient; Muon
                # and AdamW own disjoint parameter sets, so no other optimizer
                # reads this `p.grad`; and `zero_grad(set_to_none=True)` follows
                # this step, discarding the buffer rather than reusing it. This
                # would break under DDP's `gradient_as_bucket_view=True`, which
                # aliases `p.grad` to a reduction bucket; mutating it in place
                # would corrupt the bucket instead of just the local gradient.
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(g, alpha=-group["lr"] * scale)


def build_optimizers(model: nn.Module, cfg: TrainConfig) -> list[torch.optim.Optimizer]:
    """Muon for hidden 2-D matrices; AdamW for the (tied) embedding and norm gains."""
    hidden, embed, norms = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "embed" not in name:
            hidden.append(p)
        elif p.ndim == 2:
            embed.append(p)
        else:
            norms.append(p)
    muon = Muon(hidden, lr=cfg.muon_lr, momentum=cfg.muon_momentum)
    adamw = torch.optim.AdamW(
        [
            {"params": embed, "weight_decay": cfg.adamw_wd},
            {"params": norms, "weight_decay": 0.0},
        ],
        lr=cfg.adamw_lr,
        betas=cfg.adamw_betas,
    )
    for opt in (muon, adamw):
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
    return [muon, adamw]
