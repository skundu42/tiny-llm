"""Generate text from a trained checkpoint."""
from __future__ import annotations

import argparse

import torch

from .config import ModelConfig
from .model import TinyLLM
from .tokenizer import BPETokenizer


def load_model(ckpt_path: str, device: str) -> tuple[TinyLLM, ModelConfig]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["model_cfg"])
    model = TinyLLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def main() -> None:
    p = argparse.ArgumentParser(description="Sample from TinyLLM")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model, _ = load_model(args.ckpt, device)
    tok = BPETokenizer.load(args.tokenizer)

    ids = [tok.eot_id] + tok.encode(args.prompt)
    idx = torch.tensor([ids], device=device)
    out = model.generate(idx, args.max_new_tokens, temperature=args.temperature,
                         top_k=args.top_k, top_p=args.top_p, eos_id=tok.eot_id)
    print(tok.decode(out[0].tolist()[1:]))


if __name__ == "__main__":
    main()
