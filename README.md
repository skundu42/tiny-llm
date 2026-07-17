# tiny-llm

A ~500M-parameter decoder-only language model, pretrained entirely from scratch in Python/PyTorch. The transformer, the byte-level BPE tokenizer, the data pipeline, the Muon+AdamW optimizer, and the training loop are all implemented in this repo, with no `transformers` and no pre-built model components. Development and correctness verification happen locally on Apple Silicon (CPU/MPS); the real pretraining run targets a rented cloud GPU.

## Architecture

`tinyllm/model.py` implements every layer from scratch: RMSNorm pre-norm, RoPE, grouped-query attention with QK-norm, SwiGLU, tied embeddings, no biases anywhere. Two presets ship in `tinyllm/config.py`:

| Hyperparameter | `d26` (target) | `smoke` (local dev) |
|---|---|---|
| Parameters | 489,297,408 total (≈447M non-embedding) | 13,111,296 |
| Layers | 26 | 6 |
| d_model | 1280 | 256 |
| Attention heads | 20 (head_dim 64) | 4 |
| KV heads (GQA) | 4 | 2 |
| FFN | SwiGLU, hidden 3456 | SwiGLU, hidden 768 |
| Norm | RMSNorm, pre-norm + QK-norm | same |
| Positional encoding | RoPE, θ = 10,000 | same |
| Context length | 2048 | 512 |
| Vocab | 32,768 (byte-level BPE, trained by us) | same |
| Embeddings | Tied input/output | same |
| Biases | None | None |

Optimization uses a **Muon + AdamW hybrid**: Muon (momentum + Newton-Schulz orthogonalization) for all 2-D hidden weight matrices, AdamW for embeddings and RMSNorm gains, under a warmup-stable-decay (WSD) LR schedule. Training runs in bf16 autocast, with `torch.compile` on CUDA and plain DDP via `torchrun` for multi-GPU; single-GPU is the degenerate case of the same code path.

## Quickstart

```bash
uv sync
uv run pytest -q                    # 58 tests, a few seconds
```

The test suite covers every component (tokenizer, model, Muon, data sharding, training loop, eval) against hand-derived reference implementations, and needs no downloaded data.

To exercise the full training loop end-to-end on a laptop, use the `smoke` config (13.1M params, 6 layers) against a small locally-prepared shard set; this trains on Apple Silicon MPS (or CPU) in roughly 15-30 minutes:

```bash
uv run python scripts/prepare_data.py --data-dir data/smoke \
    --max-tokens 60000000 --val-tokens 2000000 --shard-tokens 25000000
uv run python -m tinyllm.train --config smoke --tokenizer tokenizer/tokenizer.json
```

(`prepare_data.py` needs a trained tokenizer first; see below.)

## Tokenizer training

`scripts/train_tokenizer.py` streams `HuggingFaceFW/fineweb-edu` (`sample-10BT`) text via `datasets`, trains our from-scratch byte-level BPE (`tinyllm.tokenizer.BPETokenizer`) to a 32,768 vocab, saves it, and reports bytes/token compression on 100 held-out documents:

```bash
uv run python scripts/train_tokenizer.py \
    --out tokenizer/tokenizer.json \
    --max-bytes 250_000_000 \
    --vocab-size 32768 \
    --num-proc 8 \
    --min-word-freq 2
```

Every flag has a sane default (see `--help`); this is the one that produces the canonical tokenizer checked into cloud runs.

## Data preparation

`scripts/prepare_data.py` streams the same FineWeb-Edu subset, encodes it with the trained tokenizer, and writes uint16 token shards (`tinyllm.data.ShardWriter`) plus a JSON index: validation shard(s) filled first, then train shards, with `tok.eot_id` appended after every document:

```bash
uv run python scripts/prepare_data.py \
    --tokenizer tokenizer/tokenizer.json \
    --data-dir data/fineweb-edu \
    --val-tokens 10_000_000 \
    --shard-tokens 100_000_000 \
    --max-tokens 0
```

`--max-tokens 0` (the default) consumes the entire `sample-10BT` subset (~10B tokens, roughly Chinchilla-optimal for the `d26` model size); pass a smaller value for a quick local subset. By default the script takes a **fast path**: it exports the trained BPE merges to a HuggingFace `tokenizers` object (`export_fast()`), verifies it is token-identical to our own encoder on the first 200 documents (`verify_fast`), and then bulk-encodes in batches of 512 docs; pass `--slow` to force the pure-Python encoder instead (much slower, used only as a fallback/cross-check).

Both `train_tokenizer.py` and `prepare_data.py` also accept `--dataset`, `--dataset-config`, `--split`, and `--text-key`, so any HuggingFace text dataset can stand in for FineWeb-Edu; the docs site has a full runbook (Runbooks > Train on a custom dataset) including a laptop-scale TinyStories example.

## Local smoke runbook

1. `uv sync`
2. `uv run pytest -q`: confirm the suite is green.
3. Train a tokenizer on a small byte budget (a few minutes): `uv run python scripts/train_tokenizer.py --max-bytes 20_000_000`.
4. Prepare a small shard set: `uv run python scripts/prepare_data.py --data-dir data/smoke --max-tokens 60000000 --val-tokens 2000000 --shard-tokens 25000000`.
5. Train: `uv run python -m tinyllm.train --config smoke --tokenizer tokenizer/tokenizer.json`; logs to `out/smoke/log.csv`, checkpoints to `out/smoke/ckpt_last.pt`, periodic sample generations printed to stdout.
6. Sample from the checkpoint: `uv run python -m tinyllm.sample --ckpt out/smoke/ckpt_last.pt --tokenizer tokenizer/tokenizer.json --prompt "Once upon a time"`.

This whole loop runs on CPU or Apple Silicon MPS and needs no GPU: `torch.compile` is automatically disabled off-CUDA (no `--no-compile` flag needed), and precision falls back to fp32 when `--dtype` is left at its `auto` default.

## Cloud runbook

The real pretraining run, `d26`, ~10B FineWeb-Edu tokens (approximately Chinchilla-optimal for 489M params), is sized for a rented GPU box:

| Setup | Wall time | Approx. cost |
|---|---|---|
| 1× H100 | ≈20 h | ≈$50-60 (at $2.5-3/h) |
| 8× H100 (DDP) | ≈2.5 h | similar total spend, higher hourly rate |

Steps on a fresh Ubuntu GPU box (Lambda Labs, RunPod, etc.):

1. Copy the repo (and, if you already have one, `tokenizer/tokenizer.json`) onto the box.
2. Run the bootstrap script: idempotent, installs `uv` if missing and runs `uv sync`, then prints the exact next commands (it skips the tokenizer step if `tokenizer/tokenizer.json` is already present):
   ```bash
   bash scripts/cloud_setup.sh
   ```
3. If no tokenizer was shipped with the repo, train one (~30-60 min CPU):
   ```bash
   uv run python scripts/train_tokenizer.py
   ```
4. Prepare the full 10B-token dataset (~1-3 h CPU, ~20 GB disk):
   ```bash
   uv run python scripts/prepare_data.py
   ```
5. Launch training inside `tmux` so it survives a dropped SSH connection:
   ```bash
   tmux new -s train
   uv run python -m tinyllm.train --config d26 --tokenizer tokenizer/tokenizer.json
   # or, on an 8-GPU box:
   uv run torchrun --standalone --nproc_per_node=8 -m tinyllm.train --config d26 --tokenizer tokenizer/tokenizer.json
   ```
   Detach with `Ctrl-b d`; reattach any time with `tmux attach -t train`.
6. If the box dies or the session is interrupted, resume from the last atomic checkpoint (bit-exact):
   ```bash
   uv run python -m tinyllm.train --config d26 --tokenizer tokenizer/tokenizer.json --resume
   ```
7. Monitor progress via `out/d26/log.csv` (or Weights & Biases, if `--wandb <project>` was passed): step, train/val loss, LR scale, tokens/sec, and MFU (populated on CUDA against a recognized GPU).
8. Once trained, evaluate and sample from the final checkpoint:
   ```bash
   uv run python -m tinyllm.eval_hellaswag --ckpt out/d26/ckpt_last.pt --tokenizer tokenizer/tokenizer.json --limit 1000
   uv run python -m tinyllm.sample --ckpt out/d26/ckpt_last.pt --tokenizer tokenizer/tokenizer.json --prompt "Once upon a time"
   ```

## Documentation site

A more detailed, browsable writeup (architecture deep-dives, tokenizer theory, training internals, runbooks, and a per-module API reference) lives in `docs-site/` (a Fumadocs/Next.js site, MDX content, pnpm-managed):

```bash
cd docs-site
pnpm install
pnpm dev      # local preview
pnpm build    # static build
```
