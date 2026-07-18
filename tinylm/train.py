"""Pretraining loop: bf16 autocast, grad accumulation, WSD schedule,
Muon+AdamW, DDP via torchrun, atomic checkpoints with bit-exact resume."""
from __future__ import annotations

import argparse
import csv
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, replace

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import MODEL_PRESETS, TRAIN_PRESETS, ModelConfig, TrainConfig
from .data import TokenShards
from .model import TinyLM
from .muon import build_optimizers

PEAK_FLOPS = {"NVIDIA H100": 989e12, "NVIDIA A100": 312e12}


def lr_scale(step: int, total: int, warmup: int, decay_frac: float) -> float:
    """Warmup-stable-decay multiplier in (0, 1]."""
    if total <= 0:
        raise ValueError("total must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if not 0.0 <= decay_frac <= 1.0:
        raise ValueError("decay_frac must be between 0 and 1")
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


def _synchronize(device: str) -> None:
    """Wait for queued accelerator work at a measurement boundary."""
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elif device.startswith("mps"):
        torch.mps.synchronize()


def _distributed_mean(value: torch.Tensor, world: int) -> torch.Tensor:
    """Return a detached scalar averaged across ranks (or unchanged locally)."""
    result = value.detach().clone()
    if world > 1:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result /= world
    return result


def _flops_per_token(model: TinyLM, cfg: ModelConfig) -> int:
    """Approximate training FLOPs/token, including the tied output projection."""
    return 6 * model.num_params() + 12 * cfg.n_layer * cfg.d_model * cfg.seq_len


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
    tc.validate()  # CLI overrides mutate an already-created preset.

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

    ckpt_path = os.path.join(tc.out_dir, "ckpt_last.pt")
    log_path = os.path.join(tc.out_dir, "log.csv")
    output_conflict = master and not resume and (
        os.path.exists(ckpt_path) or os.path.exists(log_path)
    )
    if ddp:
        conflict = torch.tensor(int(output_conflict), device=device)
        dist.broadcast(conflict, src=0)
        output_conflict = bool(conflict.item())
    if output_conflict:
        if ddp:
            dist.destroy_process_group()
        raise FileExistsError(
            f"{tc.out_dir} already contains training output; "
            "use --resume or choose a fresh --out-dir"
        )

    torch.manual_seed(tc.seed + rank)

    micro_tokens = tc.micro_batch_size * mc.seq_len
    if tc.batch_tokens % (world * micro_tokens) != 0:
        raise ValueError(
            f"batch_tokens={tc.batch_tokens} must be divisible by "
            f"world_size*micro_batch_size*seq_len={world * micro_tokens}"
        )
    grad_accum = tc.batch_tokens // (world * micro_tokens)

    train_data = TokenShards(tc.data_dir, "train")
    val_data = TokenShards(tc.data_dir, "val")

    raw_model = TinyLM(mc).to(device)
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
    if resume:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        if ckpt.get("model_cfg") != asdict(mc):
            raise ValueError("checkpoint model config does not match the requested model config")
        if len(ckpt.get("optimizers", ())) != len(optimizers):
            raise ValueError("checkpoint optimizer count does not match this training run")
        raw_model.load_state_dict(ckpt["model"])
        for opt, sd in zip(optimizers, ckpt["optimizers"]):
            opt.load_state_dict(sd)
        start_step = ckpt["step"]
        if start_step > tc.total_steps:
            raise ValueError(
                f"checkpoint step {start_step} exceeds total_steps={tc.total_steps}"
            )
        if master:
            print(f"resumed from {ckpt_path} at step {start_step}")

    tokenizer = None
    if tokenizer_path:
        from .tokenizer import BPETokenizer

        tokenizer = BPETokenizer.load(tokenizer_path)

    # MFU estimate (standard transformer FLOPs/token accounting): 6*N for the
    # matmuls (forward + backward) plus the attention term. N includes the tied
    # embedding because the same matrix is used by the dense output projection.
    flops_per_token = _flops_per_token(raw_model, mc)
    peak_flops = _peak_flops(device)

    if master:
        os.makedirs(tc.out_dir, exist_ok=True)
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
    train_seconds = 0.0
    steps_since_log = 0

    @torch.no_grad()
    def eval_val() -> float:
        model.eval()
        loss_sum = torch.zeros((), device=device)
        for k in range(tc.val_batches):
            seed = [tc.seed, 999, k] if world == 1 else [tc.seed, 999, rank, k]
            rng = np.random.default_rng(seed)
            x, y = val_data.sample_batch(tc.micro_batch_size, mc.seq_len, rng, device)
            with ctx:
                _, loss = model(x, targets=y)
            loss_sum += loss.detach()
        model.train()
        local_mean = loss_sum / tc.val_batches
        return _distributed_mean(local_mean, world).item()

    model.train()
    for step in range(start_step, tc.total_steps):
        should_log = step % tc.log_every == 0 or step == tc.total_steps - 1
        should_val = (step + 1) % tc.val_every == 0 or step == tc.total_steps - 1
        should_ckpt = master and (
            (step + 1) % tc.ckpt_every == 0 or step == tc.total_steps - 1
        )
        should_sample = (
            master and tokenizer is not None and (step + 1) % tc.sample_every == 0
        )
        step_started = time.perf_counter()
        scale = lr_scale(step, tc.total_steps, tc.warmup_steps, tc.decay_frac)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * scale

        loss_accum = torch.zeros((), device=device)
        for micro in range(grad_accum):
            rng = np.random.default_rng([tc.seed, rank, step, micro])
            x, y = train_data.sample_batch(tc.micro_batch_size, mc.seq_len, rng, device)
            sync = (not ddp) or micro == grad_accum - 1
            sync_ctx = nullcontext() if sync else model.no_sync()
            with sync_ctx, ctx:
                _, loss = model(x, targets=y)
            (loss / grad_accum).backward()
            loss_accum += loss.detach() / grad_accum

        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), tc.grad_clip)
        for opt in optimizers:
            opt.step()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        # Synchronize only where post-step work would otherwise absorb queued
        # training kernels. Regular steps remain asynchronous for throughput.
        if should_log or should_val or should_ckpt or should_sample:
            _synchronize(device)
        train_seconds += time.perf_counter() - step_started
        steps_since_log += 1

        if should_log:
            logged_loss = _distributed_mean(loss_accum, world).item()
            if master:
                tok_s = tc.batch_tokens * steps_since_log / max(train_seconds, 1e-9)
                mfu = flops_per_token * tok_s / peak_flops if peak_flops else None
                mfu_str = f"{mfu:.4f}" if mfu is not None else ""
                mfu_print = f" | mfu {mfu * 100:.1f}%" if mfu is not None else ""
                print(f"step {step:6d} | loss {logged_loss:.4f} | lr× {scale:.3f} | "
                      f"{tok_s:,.0f} tok/s{mfu_print}")
                log.writerow([step, "train", f"{logged_loss:.6f}", f"{scale:.4f}",
                              f"{tok_s:.0f}", mfu_str])
                log_file.flush()
                if tc.wandb_project and wandb_run:
                    wandb_log = {"train/loss": logged_loss, "lr_scale": scale,
                                 "tok_per_sec": tok_s}
                    if mfu is not None:
                        wandb_log["mfu"] = mfu
                    wandb_run.log(wandb_log, step=step)
                history.append((step, logged_loss))
            train_seconds = 0.0
            steps_since_log = 0

        if should_val:
            vl = eval_val()
            final_val = vl
            if master:
                print(f"step {step:6d} | val loss {vl:.4f}")
                log.writerow([step, "val", f"{vl:.6f}", "", "", ""])
                log_file.flush()
                if tc.wandb_project and wandb_run:
                    wandb_run.log({"val/loss": vl}, step=step)

        if should_ckpt:
            _save_ckpt(ckpt_path, raw_model, optimizers, step + 1, mc, tc)

        if should_sample:
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
    p = argparse.ArgumentParser(description="Pretrain TinyLM")
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

    mc = replace(MODEL_PRESETS[args.config])
    tc = replace(TRAIN_PRESETS[args.config])
    if args.data_dir:
        tc.data_dir = args.data_dir
    if args.out_dir:
        tc.out_dir = args.out_dir
    if args.steps is not None:
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
