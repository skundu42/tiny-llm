"""Stream a HuggingFace dataset (FineWeb-Edu by default), encode with the
(verified) fast tokenizer, write shards."""
import argparse
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

from tinyllm.data import ShardWriter
from tinyllm.tokenizer import BPETokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    p.add_argument("--data-dir", default="data/fineweb-edu")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu",
                   help="HuggingFace dataset repo id")
    p.add_argument("--dataset-config", default="sample-10BT",
                   help="dataset config name; pass '' for datasets without one")
    p.add_argument("--split", default="train")
    p.add_argument("--text-key", default="text",
                   help="column holding the document text")
    p.add_argument("--val-tokens", type=int, default=10_000_000)
    p.add_argument("--shard-tokens", type=int, default=100_000_000)
    p.add_argument("--max-tokens", type=int, default=0, help="0 = entire subset")
    p.add_argument("--slow", action="store_true", help="use the pure-Python encoder")
    args = p.parse_args()
    if args.val_tokens <= 0:
        p.error("--val-tokens must be positive")
    if args.shard_tokens <= 0:
        p.error("--shard-tokens must be positive")
    if args.max_tokens < 0:
        p.error("--max-tokens must be non-negative")
    if args.max_tokens and args.max_tokens <= args.val_tokens:
        p.error("--max-tokens must exceed --val-tokens so the train split is non-empty")

    tok = BPETokenizer.load(args.tokenizer)
    if tok.vocab_size > 65_536:
        p.error("tokenizer vocabulary exceeds the uint16 shard limit of 65536")
    eot = tok.eot_id

    ds = load_dataset(args.dataset, name=args.dataset_config or None,
                      split=args.split, streaming=True)
    docs = (ex[args.text_key] for ex in ds)

    fast = None
    if not args.slow:
        fast = tok.export_fast()
        sample = list(itertools.islice(docs, 200))
        tok.verify_fast(fast, sample)
        print("fast tokenizer verified token-identical on 200 docs")
        docs = itertools.chain(sample, docs)

    val_w = ShardWriter(args.data_dir, "val", min(args.shard_tokens, args.val_tokens))
    train_w = ShardWriter(args.data_dir, "train", args.shard_tokens)
    total = 0
    t0 = time.time()
    next_report = 100_000_000

    def batches(it, size=512):
        while True:
            b = list(itertools.islice(it, size))
            if not b:
                return
            yield b

    for batch in batches(docs):
        if fast is not None:
            encs = [e.ids for e in fast.encode_batch(batch)]
        else:
            encs = [tok.encode(t) for t in batch]
        for ids in encs:
            ids.append(eot)
            w = val_w if val_w.total_written < args.val_tokens else train_w
            w.write(ids)
            total += len(ids)
        if total >= next_report:
            rate = total / (time.time() - t0)
            print(f"{total/1e9:.2f}B tokens ({rate/1e6:.2f}M tok/s)")
            next_report += 100_000_000
        if args.max_tokens and total >= args.max_tokens:
            break

    val_w.close()
    train_w.close()
    print(f"done: {total:,} tokens ({val_w.total_written:,} val, "
          f"{train_w.total_written:,} train) in {args.data_dir}")


if __name__ == "__main__":
    main()
