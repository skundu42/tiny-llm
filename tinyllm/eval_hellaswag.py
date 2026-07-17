"""HellaSwag continuation-likelihood eval (acc and length-normalized acc_norm)."""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from .tokenizer import BPETokenizer


@torch.no_grad()
def ending_losses(model, ctx_ids, ending_ids_list, device) -> list[tuple[float, float]]:
    """Per ending: (sum, mean) cross-entropy over the ending tokens only."""
    out = []
    for ending in ending_ids_list:
        ids = torch.tensor([ctx_ids + ending], device=device)
        logits, _ = model(ids[:, :-1])
        targets = ids[:, 1:]
        losses = F.cross_entropy(
            logits.float().transpose(1, 2), targets, reduction="none"
        )[0]
        tail = losses[len(ctx_ids) - 1 :]  # positions predicting ending tokens
        out.append((tail.sum().item(), tail.mean().item()))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="HellaSwag eval")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    from datasets import load_dataset

    from .sample import load_model

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, device)
    tok = BPETokenizer.load(args.tokenizer)

    ds = load_dataset("hellaswag", split="validation", streaming=True)
    n = acc = acc_norm = 0
    for ex in ds:
        if n >= args.limit:
            break
        ctx_ids = tok.encode(ex["ctx"])
        endings = [tok.encode(" " + e) for e in ex["endings"]]
        if any(len(ctx_ids) + len(e) > cfg.seq_len for e in endings):
            continue
        losses = ending_losses(model, ctx_ids, endings, device)
        label = int(ex["label"])
        acc += int(min(range(4), key=lambda i: losses[i][0]) == label)
        acc_norm += int(min(range(4), key=lambda i: losses[i][1]) == label)
        n += 1
        if n % 100 == 0:
            print(f"{n}: acc {acc / n:.4f} | acc_norm {acc_norm / n:.4f}")
    print(f"final ({n} examples): acc {acc / n:.4f} | acc_norm {acc_norm / n:.4f}")


if __name__ == "__main__":
    main()
