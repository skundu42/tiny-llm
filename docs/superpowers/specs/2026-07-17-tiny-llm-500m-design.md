# tiny-llm — a ~500M-parameter LLM pretrained from scratch

**Date:** 2026-07-17
**Status:** Approved (design), pending spec review

## Goal

Build a ~500M-parameter decoder-only language model entirely from scratch in Python/PyTorch — model code, tokenizer, data pipeline, optimizer, and training loop, with no HuggingFace `transformers` and no pre-built model components. Development and verification happen locally on an Apple M3 Pro (18 GB); the real pretraining run (~10B tokens, Chinchilla-optimal) targets a rented cloud GPU (single H100 ≈ 20 h ≈ $50–60, or 8×H100 ≈ 2.5 h).

"Best architecture" is interpreted as the current consensus recipe for strong small dense models (Qwen3-0.6B / OLMo-2 / nanochat lineage), not exotic speedrun tricks.

## Non-goals

- Fine-tuning, instruction tuning, RLHF, or chat formatting (future work).
- Inference serving / quantization / export formats.
- MoE, multimodality, long context (>2048).
- Training to competitive-benchmark quality beyond what ~10B tokens gives.

## Model architecture (~489M params)

Decoder-only transformer, every layer hand-written in PyTorch:

| Hyperparameter | Value |
|---|---|
| Parameters | ≈489M total (≈447M non-embedding) |
| Layers | 26 |
| d_model | 1280 |
| Attention heads | 20 (head_dim 64) |
| KV heads (GQA) | 4 |
| FFN | SwiGLU, hidden 3456 |
| Norm | RMSNorm, pre-norm |
| QK-norm | Yes (RMSNorm on q and k per head, Qwen3-style) |
| Positional encoding | RoPE, θ = 10,000 |
| Context length | 2048 |
| Vocab | 32,768 (byte-level BPE, trained by us) |
| Embeddings | Tied input/output |
| Biases | None anywhere |
| Init | normal(0, 0.02); residual output projections scaled by 1/√(2·n_layers) |

Attention uses `F.scaled_dot_product_attention` (FlashAttention kernel) in the model; a hand-written reference attention lives in the test suite and must match SDPA numerically.

Two configs ship as presets: `d26` (the ~489M target above) and `smoke` (~10M params: 6 layers, d_model 256) for local end-to-end verification on MPS/CPU.

## Tokenizer (from scratch)

- Byte-level BPE, vocab 32,768, one special token `<|endoftext|>` used as document separator.
- GPT-4-style regex pre-tokenization (via the `regex` module).
- Trainer: runs on a ~1 GB FineWeb-Edu sample, multiprocessing for pair counting, incremental merge updates. Saves vocab + merges as JSON.
- Encoder: pure-Python encode/decode with an LRU cache per pre-token; exact byte-level roundtrip guaranteed.
- Fast bulk-encode path (optional): export our learned merges to a HuggingFace `tokenizers` object used *only* for bulk corpus encoding, verified token-identical against our own encoder on a sample before use. Our implementation remains the source of truth.

## Data pipeline

- Source: `HuggingFaceFW/fineweb-edu`, `sample-10BT` subset, streamed via `datasets` (no full download to disk before processing).
- `scripts/prepare_data.py`: stream docs → our BPE encode (multiprocessing) → append `<|endoftext|>`-separated token stream into uint16 `.bin` shards (~100M tokens each) with a small JSON index. Last shard held out as validation.
- Loader: numpy memmap over shards; random crops of `seq_len+1` tokens; DDP-aware (rank-sharded RNG); no epoch bookkeeping needed at ~1 epoch.
- Data prep is CPU-bound and runs on the cloud box (or overnight locally); ~2–4 h with multiprocessing.

## Optimizer & training recipe

- **Muon + AdamW hybrid** (nanochat / modded-nanoGPT recipe): Muon (momentum + Newton-Schulz orthogonalization, 5 iterations, implemented from scratch ~40 lines) for all 2-D hidden weight matrices; AdamW (β=0.9/0.95, wd 0.1) for embeddings/unembedding and RMSNorm gains.
- LR schedule: warmup–stable–decay (WSD); short warmup, constant plateau, linear decay over the final ~20% of steps. Muon LR ≈ 0.02 (momentum 0.95), AdamW LR ≈ 3e-4; exact values tuned on the smoke config.
- Batch: ~524,288 tokens/step (grad accumulation; micro-batch sized to GPU memory) → ~19–20k steps for 10B tokens.
- bf16 autocast, grad clip 1.0, `torch.compile` (with `--no-compile` escape hatch for MPS/debug).
- Checkpointing: model + optimizer + dataloader state + step, atomic write, `--resume` restores bit-exact training.
- Logging: CSV always; wandb optional behind a flag. Periodic val loss, tokens/sec, MFU estimate, and sample generations.
- Distributed: plain DDP via `torchrun`; single-GPU is the degenerate case of the same code path. No FSDP (unnecessary at 500M).

## Documentation site (Fumadocs)

A detailed documentation site lives in `docs-site/` (Next.js + Fumadocs, MDX content, pnpm-managed), documenting the project for readers who want to understand *and reproduce* it:

- **Getting started**: install, quickstart, repo tour.
- **Architecture**: one page per component — transformer overview, RMSNorm, RoPE, GQA attention + QK-norm, SwiGLU, weight init/tying — each explaining the math, the from-scratch code, and why the design choice is current best practice.
- **Tokenizer**: byte-level BPE theory, training algorithm, regex pre-tokenization, file format.
- **Data pipeline**: FineWeb-Edu, sharding format, memmap loading.
- **Training**: Muon + AdamW hybrid (with Newton-Schulz explanation), WSD schedule, bf16/compile, DDP, checkpointing.
- **Evaluation & inference**: val loss, HellaSwag, KV-cache generation.
- **Runbooks**: local smoke train on Apple Silicon; full cloud pretrain (provisioning → data prep → launch → monitoring → cost).
- **API reference**: per-module reference for `tinyllm/*`.

Verification: `pnpm build` must succeed (static export not required; default Next build), all internal links valid, code snippets in docs lifted from the real source.

## Evaluation & generation

- Val loss / perplexity on the held-out FineWeb-Edu shard.
- HellaSwag accuracy (nanoGPT-style continuation-likelihood scoring) as an external benchmark; expected ~0.30–0.35 for a 500M/10B-token model.
- `sample.py`: autoregressive generation with KV cache, temperature, top-k, top-p.

## Repository layout

```
tiny-llm/
├── pyproject.toml            # uv-managed: torch, numpy, regex, datasets, tqdm, pytest; wandb optional
├── README.md                 # quickstart + cloud runbook
├── tinyllm/
│   ├── config.py             # ModelConfig/TrainConfig dataclasses + presets (d26, smoke)
│   ├── model.py              # RMSNorm, RoPE, GQA attention w/ QK-norm, SwiGLU, Block, TinyLLM
│   ├── tokenizer.py          # BPE: train / encode / decode / save / load / fast-export
│   ├── data.py               # shard writer + memmap loader (DDP-aware)
│   ├── muon.py               # Muon optimizer + param-group split builder
│   ├── train.py              # training loop (single-GPU and torchrun DDP)
│   ├── sample.py             # generation with KV cache
│   └── eval_hellaswag.py
├── scripts/
│   ├── train_tokenizer.py
│   ├── prepare_data.py
│   └── cloud_setup.sh        # provision a fresh GPU box: uv, deps, data, launch
└── tests/
    ├── test_tokenizer.py     # roundtrip on unicode/emoji/code, known-merge toy corpus, save/load
    ├── test_model.py         # param count ≈489M, RoPE properties, GQA, SDPA vs reference attention
    ├── test_data.py          # shard write/read roundtrip, DDP shard disjointness
    ├── test_muon.py          # Newton-Schulz orthogonality, converges on toy problem
    └── test_train.py         # overfit tiny batch (loss→~0), checkpoint-resume exactness
```

## Verification strategy

1. **Unit tests** (pytest, run on Mac): everything in `tests/` above.
2. **Smoke pretrain** (Mac, MPS): `smoke` config on a small prepared shard — loss curve must fall convincingly below its starting value and generations must show learned character/word structure.
3. **Cloud dress rehearsal**: first 15 minutes of the real config on the GPU box — check tokens/sec, MFU, loss trajectory against expectation before committing to the full run.
4. **Full run acceptance**: final val loss ≈ 2.7–3.0 (typical for this scale/recipe on FineWeb-Edu), HellaSwag above random (>0.28), and qualitatively coherent short generations.

## Cloud runbook (README)

- Provider suggestions: Lambda / RunPod / Vast, single H100 80GB (default) or 8×H100.
- `cloud_setup.sh`: clone repo → `uv sync` → train or download tokenizer → `prepare_data.py` → `train.py` under `tmux`, checkpoints synced to persistent storage.
- Budget: ~20 h single-H100 ≈ $50–60 at ~$2.5–3/h; multi-GPU via `torchrun --nproc_per_node=8` unchanged code.

## Key decisions log

- **Cloud pretrain, not local**: M3 Pro would need ~2 months for 10B tokens. (User-selected.)
- **PyTorch over MLX/NumPy**: portable to CUDA, still fully from-scratch at the layer level. (User-selected.)
- **Own BPE + FineWeb-Edu**: end-to-end from scratch; strongest open small-model pretraining corpus. (User-selected.)
- **Muon+AdamW over pure AdamW**: ~1.5–2× data-efficiency at identical compute, validated at this exact scale by nanochat (561M). (User-selected.)
- **GQA + QK-norm + SwiGLU + RMSNorm + RoPE + tied embeddings**: the current small-model consensus stack; each piece is standard, stable, and cheap to implement from scratch.
- **SDPA kernel inside from-scratch attention**: writing the *math* from scratch but using PyTorch's fused kernel is the only way the cloud run is affordable; the manual implementation ships in tests.
