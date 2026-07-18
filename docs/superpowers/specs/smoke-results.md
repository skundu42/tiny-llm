# Smoke run results (measured)

Local end-to-end verification of the full pipeline on an **Apple M3 Pro (18 GB), PyTorch MPS backend**, 2026-07-17.
These are real measurements from the committed code, not estimates.

## Tokenizer (`scripts/train_tokenizer.py`)

| Metric | Value |
|---|---|
| Vocab size | 32,768 (32,511 merges + 256 byte tokens + `<\|endoftext\|>`) |
| Training corpus | ~250 MB of FineWeb-Edu `sample-10BT` (streamed) |
| Workers | 9 processes |
| Compression (100 held-out docs) | **4.517 bytes/token** |
| Sample sentence compression | 5.00 bytes/token |
| `eot_id` | 32767 |

## Data preparation (`scripts/prepare_data.py`)

| Metric | Value |
|---|---|
| Total tokens | **60,003,098** |
| Validation | 2,004,611 (`val_00000.bin`, `val_00001.bin`) |
| Train | 57,998,487 (`train_00000.bin`…`train_00002.bin`) |
| Shard size | 25M tokens (uint16) |
| Fast tokenizer | verified token-identical on 200 docs before use |

## Smoke pretrain (`tinylm.train --config smoke --device mps`)

Model: 13,111,296 params (6 layers, d_model 256, 4 heads, 2 KV heads, ctx 512), grad_accum 4, 600 steps.

| Step | Train loss | Val loss |
|---|---|---|
| 0 | 10.4399 | |
| 99 | | 6.7274 |
| 100 | 6.6888 | |
| 199 | | 6.2567 |
| 200 | 6.1561 | |
| 299 | | 5.9739 |
| 300 | 5.8912 | |
| 399 | | 5.7768 |
| 400 | 5.6471 | |
| 499 | | 5.6177 |
| 500 | 5.5151 | |
| 599 | 5.5300 | **5.5012** |

- **Initial loss 10.4399 ≈ ln(32768) = 10.3972**; confirms correct uniform initialization.
- Final val loss **5.50**, comfortably below the plan's ≤6.0 acceptance gate.
- Val tracks train throughout, so no overfitting; unsurprising, since 600 steps × 16,384 tokens/step = 9,830,400 tokens is only **0.169 epochs** over the 57,998,487-token train split (a full epoch would need ~3,540 steps).
- Throughput: **~11,000–13,400 tok/s** on MPS; 600 steps ≈ 15 min wall clock.
- LR schedule behaved as designed: warmup to 1.0 by step 20, stable plateau, decay engaged from step ~480 (lr× 0.833 → 0.008 at step 599).

## Resume verification

`--steps 620 --resume` → `resumed from out/smoke/ckpt_last.pt at step 600`, continued to step 619, val loss **5.4914**. Checkpoint/resume works on a real run, not just in tests.

## Sample generations

At step 400 (temperature 0.8, top-k 50, unconditional):

> You need to know how to make sure that you need to be able to get down the next day!
> A. The following you
> - It's important to do that this is the same. When it's an example of what you are going to a different way.

At step 600 via `tinylm.sample`, prompt `"The"` (seed 1):

> The 'the last in the first years of the past was the most common in the world. The best way to be the most diverse and highly difficult, while there was a lot of the most important for future lives. In a very high level of work, the world's most important is the same as

Prompt `"Photosynthesis is"` (seed 2):

> Photosynthesis is a problem that has been published as a new disease in the past 20 years. There are also some studies, which are called "A disease" and is found on these problems in people by the first 20 years in the United States. The most common number of people is about 1.

**Interpretation:** at 13M params and 600 steps (~10M tokens seen), the model has learned English morphology, syntax, punctuation, and quotation/list formatting: it produces grammatical, fluent sentences. It has *not* learned factual grounding (note "Photosynthesis is a problem that has been published as a new disease"), which is exactly what this scale and token budget predict. The d26 (489M) config at 10B tokens is where semantic coherence is expected.
