"""Train the 32k byte-level BPE tokenizer on a HuggingFace dataset
(FineWeb-Edu by default)."""
import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

from tinylm.tokenizer import BPETokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="tokenizer/tokenizer.json")
    p.add_argument("--max-bytes", type=int, default=250_000_000)
    p.add_argument("--vocab-size", type=int, default=32768)
    p.add_argument("--num-proc", type=int, default=max(1, (os.cpu_count() or 1) - 2))
    p.add_argument("--min-word-freq", type=int, default=2)
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu",
                   help="HuggingFace dataset repo id")
    p.add_argument("--dataset-config", default="sample-10BT",
                   help="dataset config name; pass '' for datasets without one")
    p.add_argument("--split", default="train")
    p.add_argument("--text-key", default="text",
                   help="column holding the document text")
    args = p.parse_args()
    if not 257 <= args.vocab_size <= 65_536:
        p.error("--vocab-size must be between 257 and 65536 for uint16 shards")
    if args.max_bytes <= 0:
        p.error("--max-bytes must be positive")
    if args.num_proc <= 0:
        p.error("--num-proc must be positive")
    if args.min_word_freq <= 0:
        p.error("--min-word-freq must be positive")

    ds = load_dataset(args.dataset, name=args.dataset_config or None,
                      split=args.split, streaming=True)
    it = iter(ds)

    def texts():
        seen = 0
        for ex in it:
            t = ex[args.text_key]
            seen += len(t.encode("utf-8"))
            yield t
            if seen >= args.max_bytes:
                return

    print(f"training vocab={args.vocab_size} on ~{args.max_bytes/1e6:.0f}MB, "
          f"num_proc={args.num_proc}")
    tok = BPETokenizer.train(texts(), vocab_size=args.vocab_size,
                             num_proc=args.num_proc,
                             min_word_freq=args.min_word_freq, verbose=True)
    parent = os.path.dirname(args.out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tok.save(args.out)
    print(f"saved {args.out}")

    holdout = [ex[args.text_key] for ex in itertools.islice(it, 100)]
    nbytes = sum(len(t.encode("utf-8")) for t in holdout)
    ntok = sum(len(tok.encode(t)) for t in holdout)
    if ntok:
        print(f"compression on holdout: {nbytes / ntok:.3f} bytes/token")
    else:
        print("compression on holdout: unavailable (holdout produced no tokens)")


if __name__ == "__main__":
    main()
