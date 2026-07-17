"""Train the canonical 32k byte-level BPE tokenizer on FineWeb-Edu text."""
import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset

from tinyllm.tokenizer import BPETokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="tokenizer/tokenizer.json")
    p.add_argument("--max-bytes", type=int, default=250_000_000)
    p.add_argument("--vocab-size", type=int, default=32768)
    p.add_argument("--num-proc", type=int, default=max(1, os.cpu_count() - 2))
    p.add_argument("--min-word-freq", type=int, default=2)
    args = p.parse_args()

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    it = iter(ds)

    def texts():
        seen = 0
        for ex in it:
            t = ex["text"]
            seen += len(t.encode("utf-8"))
            yield t
            if seen >= args.max_bytes:
                return

    print(f"training vocab={args.vocab_size} on ~{args.max_bytes/1e6:.0f}MB, "
          f"num_proc={args.num_proc}")
    tok = BPETokenizer.train(texts(), vocab_size=args.vocab_size,
                             num_proc=args.num_proc,
                             min_word_freq=args.min_word_freq, verbose=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tok.save(args.out)
    print(f"saved {args.out}")

    holdout = [ex["text"] for ex in itertools.islice(it, 100)]
    nbytes = sum(len(t.encode("utf-8")) for t in holdout)
    ntok = sum(len(tok.encode(t)) for t in holdout)
    print(f"compression on holdout: {nbytes / ntok:.3f} bytes/token")


if __name__ == "__main__":
    main()
