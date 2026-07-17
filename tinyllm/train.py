"""Pretraining loop: bf16 autocast, grad accumulation, WSD schedule,
Muon+AdamW, DDP via torchrun, atomic checkpoints with bit-exact resume."""
from __future__ import annotations

import argparse
import csv
import os
import time
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import MODEL_PRESETS, TRAIN_PRESETS, ModelConfig, TrainConfig
from .data import TokenShards
from .model import TinyLLM
from .muon import build_optimizers

PEAK_FLOPS = {"NVIDIA H100": 989e12, "NVIDIA A100": 312e12}


def lr_scale(step: int, total: int, warmup: int, decay_frac: float) -> float:
    """Warmup-stable-decay multiplier in (0, 1]."""
    if step < warmup:
        return (step + 1) / warmup
    decay_start = int(total * (1 - decay_frac))
    if step < decay_start:
        return 1.0
    return max(1e-8, (total - step) / (total - decay_start))


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _peak_flops(device: str) -> float | None:
    """Dense bf16 peak FLOP/s for the current GPU, matched by substring against
    `PEAK_FLOPS`. Returns None off-CUDA or for an unrecognized GPU, in which
    case MFU is not computable."""
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name()
    for key, peak in PEAK_FLOPS.items():
        if key in name:
            return peak
    return None


def _autocast_ctx(device: str, dtype: str):
    if dtype == "auto":
        dtype = "bf16" if device.startswith("cuda") else "fp32"
    if dtype == "bf16":
        return torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16)
    return nullcontext()


def _save_ckpt(path: str, raw_model, optimizers, step, model_cfg, train_cfg) -> None:
    tmp = path + ".tmp"
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizers": [o.state_dict() for o in optimizers],
            "step": step,
            "model_cfg": asdict(model_cfg),
            "train_cfg": asdict(train_cfg),
        },
        tmp,
    )
    os.replace(tmp, path)


def run(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    tokenizer_path: str | None = None,
    resume: bool = False,
    device: str | None = None,
) -> dict:
    tc, mc = train_cfg, model_cfg

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
    else:
        rank, world = 0, 1
        device = device or _pick_device()
    master = rank == 0

    torch.manual_seed(tc.seed + rank)

    micro_tokens = tc.micro_batch_size * mc.seq_len
    assert tc.batch_tokens % (world * micro_tokens) == 0, (
        f"batch_tokens={tc.batch_tokens} must be divisible by "
        f"world_size*micro_batch_size*seq_len={world * micro_tokens}"
    )
    grad_accum = tc.batch_tokens // (world * micro_tokens)

    train_data = TokenShards(tc.data_dir, "train")
    val_data = TokenShards(tc.data_dir, "val")

    raw_model = TinyLLM(mc).to(device)
    model = raw_model
    use_compile = tc.compile and device.startswith("cuda")
    if tc.compile and not use_compile and master:
        print("torch.compile disabled: not on CUDA")
    if use_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[local_rank] if device.startswith("cuda") else None)

    optimizers = build_optimizers(raw_model, tc)

    start_step = 0
    ckpt_path = os.path.join(tc.out_dir, "ckpt_last.pt")
    if resume:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        for opt, sd in zip(optimizers, ckpt["optimizers"]):
            opt.load_state_dict(sd)
        start_step = ckpt["step"]
        if master:
            print(f"resumed from {ckpt_path} at step {start_step}")

    tokenizer = None
    if tokenizer_path:
        from .tokenizer import BPETokenizer

        tokenizer = BPETokenizer.load(tokenizer_path)

    # MFU estimate (standard transformer FLOPs/token accounting): 6*N for the
    # matmuls (forward + backward, N = non-embedding params) plus the
    # attention term 12*n_layer*d_model*seq_len. Only meaningful on CUDA with a
    # recognized GPU in PEAK_FLOPS; otherwise mfu is left uncomputable (None).
    non_embed_params = raw_model.num_params(non_embedding=True)
    flops_per_token = 6 * non_embed_params + 12 * mc.n_layer * mc.d_model * mc.seq_len
    peak_flops = _peak_flops(device)

    if master:
        os.makedirs(tc.out_dir, exist_ok=True)
        log_path = os.path.join(tc.out_dir, "log.csv")
        new_log = not os.path.exists(log_path)
        log_file = open(log_path, "a", newline="")
        log = csv.writer(log_file)
        if new_log:
            log.writerow(["step", "split", "loss", "lr_scale", "tok_per_sec", "mfu"])
        wandb_run = None
        if tc.wandb_project:
            import wandb

            wandb_run = wandb.init(project=tc.wandb_project,
                                   config={**asdict(mc), **asdict(tc)})
        print(f"params: {raw_model.num_params():,} | device {device} | world {world} "
              f"| grad_accum {grad_accum} | steps {tc.total_steps}")

    ctx = _autocast_ctx(device, tc.dtype)
    history: list[tuple[int, float]] = []
    final_val = float("nan")
    t_last = time.time()

    @torch.no_grad()
    def eval_val() -> float:
        model.eval()
        losses = []
        for k in range(tc.val_batches):
            rng = np.random.default_rng([tc.seed, 999, k])
            x, y = val_data.sample_batch(tc.micro_batch_size, mc.seq_len, rng, device)
            with ctx:
                _, loss = model(x, targets=y)
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    model.train()
    for step in range(start_step, tc.total_steps):
        scale = lr_scale(step, tc.total_steps, tc.warmup_steps, tc.decay_frac)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * scale

        loss_accum = 0.0
        for micro in range(grad_accum):
            rng = np.random.default_rng([tc.seed, rank, step, micro])
            x, y = train_data.sample_batch(tc.micro_batch_size, mc.seq_len, rng, device)
            sync = (not ddp) or micro == grad_accum - 1
            sync_ctx = nullcontext() if sync else model.no_sync()
            with sync_ctx, ctx:
                _, loss = model(x, targets=y)
            (loss / grad_accum).backward()
            loss_accum += loss.item() / grad_accum

        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), tc.grad_clip)
        for opt in optimizers:
            opt.step()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        if master and (step % tc.log_every == 0 or step == tc.total_steps - 1):
            now = time.time()
            tok_s = tc.batch_tokens * tc.log_every / max(now - t_last, 1e-9)
            t_last = now
            mfu = flops_per_token * tok_s / peak_flops if peak_flops else None
            mfu_str = f"{mfu:.4f}" if mfu is not None else ""
            mfu_print = f" | mfu {mfu * 100:.1f}%" if mfu is not None else ""
            print(f"step {step:6d} | loss {loss_accum:.4f} | lr× {scale:.3f} | "
                  f"{tok_s:,.0f} tok/s{mfu_print}")
            log.writerow([step, "train", f"{loss_accum:.6f}", f"{scale:.4f}",
                          f"{tok_s:.0f}", mfu_str])
            log_file.flush()
            if tc.wandb_project and wandb_run:
                wandb_log = {"train/loss": loss_accum, "lr_scale": scale,
                             "tok_per_sec": tok_s}
                if mfu is not None:
                    wandb_log["mfu"] = mfu
                wandb_run.log(wandb_log, step=step)
            history.append((step, loss_accum))

        if (step + 1) % tc.val_every == 0 or step == tc.total_steps - 1:
            vl = eval_val()
            final_val = vl
            if master:
                print(f"step {step:6d} | val loss {vl:.4f}")
                log.writerow([step, "val", f"{vl:.6f}", "", "", ""])
                log_file.flush()
                if tc.wandb_project and wandb_run:
                    wandb_run.log({"val/loss": vl}, step=step)

        if master and ((step + 1) % tc.ckpt_every == 0 or step == tc.total_steps - 1):
            _save_ckpt(ckpt_path, raw_model, optimizers, step + 1, mc, tc)

        if (master and tokenizer is not None
                and (step + 1) % tc.sample_every == 0):
            prompt = torch.tensor([[tokenizer.eot_id]], device=device)
            out = raw_model.generate(prompt, max_new_tokens=100, temperature=0.8, top_k=50)
            print("sample:", tokenizer.decode(out[0].tolist()[1:]))
            model.train()

    if master:
        log_file.close()
    if ddp:
        dist.destroy_process_group()
    return {"history": history, "final_val_loss": final_val, "step": tc.total_steps}


def main() -> None:
    p = argparse.ArgumentParser(description="Pretrain TinyLLM")
    p.add_argument("--config", choices=sorted(MODEL_PRESETS), default="smoke")
    p.add_argument("--data-dir")
    p.add_argument("--out-dir")
    p.add_argument("--steps", type=int)
    p.add_argument("--device")
    p.add_argument("--tokenizer")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp32"])
    p.add_argument("--wandb", dest="wandb_project")
    args = p.parse_args()

    mc = MODEL_PRESETS[args.config]
    tc = TRAIN_PRESETS[args.config]
    if args.data_dir:
        tc.data_dir = args.data_dir
    if args.out_dir:
        tc.out_dir = args.out_dir
    if args.steps:
        tc.total_steps = args.steps
    if args.no_compile:
        tc.compile = False
    if args.dtype:
        tc.dtype = args.dtype
    if args.wandb_project:
        tc.wandb_project = args.wandb_project

    run(mc, tc, tokenizer_path=args.tokenizer, resume=args.resume, device=args.device)


if __name__ == "__main__":
    main()
