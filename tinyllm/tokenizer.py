"""Byte-level BPE tokenizer, written from scratch.

Training counts unique pre-tokens (words) once each with a frequency weight,
then runs merges with an incrementally-maintained pair-count index, so 32k
merges over ~1 GB of text stay tractable in pure Python.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from functools import lru_cache
from multiprocessing import Pool

import regex as re

# GPT-4-style pre-tokenization pattern. Possessive quantifiers are replaced by
# greedy ones so the pattern behaves identically under Python `regex` and the
# Rust engine used by HF `tokenizers` (see export_fast in Task 3).
SPLIT_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]"
    r"|\s+(?!\S)"
    r"|\s+"
)

ENDOFTEXT = "<|endoftext|>"

_PATTERN = re.compile(SPLIT_PATTERN)


def _count_words(text: str) -> Counter[bytes]:
    """Pre-tokenize one document and count utf-8 word occurrences."""
    return Counter(m.group().encode("utf-8") for m in _PATTERN.finditer(text))


@lru_cache(maxsize=1)
def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte -> printable-unicode-char table."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class BPETokenizer:
    def __init__(
        self,
        merges: list[tuple[int, int]] | None = None,
        special_tokens: dict[str, int] | None = None,
    ) -> None:
        self.merges = [tuple(m) for m in (merges or [])]
        self.special_tokens = dict(special_tokens or {})
        self._build()

    def _build(self) -> None:
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self._special_of_id = {v: k for k, v in self.special_tokens.items()}
        self._encode_word = lru_cache(maxsize=1 << 16)(self._encode_word_uncached)

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special_tokens)

    @property
    def eot_id(self) -> int:
        return self.special_tokens[ENDOFTEXT]

    # ------------------------------------------------------------- training

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        num_proc: int = 1,
        min_word_freq: int = 1,
        verbose: bool = False,
    ) -> "BPETokenizer":
        assert vocab_size > 257, "need room for 256 bytes + <|endoftext|>"
        n_merges = vocab_size - 256 - 1

        word_counts: Counter[bytes] = Counter()
        if num_proc > 1:
            with Pool(num_proc) as pool:
                for c in pool.imap_unordered(_count_words, texts, chunksize=64):
                    word_counts.update(c)
        else:
            for t in texts:
                word_counts.update(_count_words(t))
        if min_word_freq > 1:
            word_counts = Counter(
                {w: c for w, c in word_counts.items() if c >= min_word_freq}
            )

        words: list[list[int]] = [list(w) for w in word_counts]
        counts: list[int] = list(word_counts.values())

        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_to_words: dict[tuple[int, int], set[int]] = {}
        for wi, w in enumerate(words):
            c = counts[wi]
            for pair in zip(w, w[1:]):
                pair_counts[pair] += c
                pair_to_words.setdefault(pair, set()).add(wi)

        merges: list[tuple[int, int]] = []
        for step in range(n_merges):
            if not pair_counts:
                break
            # highest count wins; ties broken by smallest pair ids => deterministic
            best = max(pair_counts.items(), key=lambda kv: (kv[1], (-kv[0][0], -kv[0][1])))[0]
            new_id = 256 + step
            merges.append(best)
            for wi in list(pair_to_words.get(best, ())):
                w, c = words[wi], counts[wi]
                for pair in zip(w, w[1:]):
                    pair_counts[pair] -= c
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    s = pair_to_words.get(pair)
                    if s is not None:
                        s.discard(wi)
                merged: list[int] = []
                i = 0
                while i < len(w):
                    if i < len(w) - 1 and w[i] == best[0] and w[i + 1] == best[1]:
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(w[i])
                        i += 1
                words[wi] = merged
                for pair in zip(merged, merged[1:]):
                    pair_counts[pair] += c
                    pair_to_words.setdefault(pair, set()).add(wi)
            if verbose and (step + 1) % 1000 == 0:
                print(f"merge {step + 1}/{n_merges}")

        return cls(merges, {ENDOFTEXT: 256 + len(merges)})

    # ------------------------------------------------------- encode / decode

    def _encode_word_uncached(self, word: bytes) -> tuple[int, ...]:
        ids = list(word)
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            best = min(pairs, key=lambda p: self.merge_ranks.get(p, float("inf")))
            if best not in self.merge_ranks:
                break
            new_id = 256 + self.merge_ranks[best]
            out: list[int] = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == best[0] and ids[i + 1] == best[1]:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        return tuple(ids)

    def encode(self, text: str) -> list[int]:
        """Encode text to token ids. Special tokens are never produced."""
        ids: list[int] = []
        for m in _PATTERN.finditer(text):
            ids.extend(self._encode_word(m.group().encode("utf-8")))
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        parts: list[bytes] = []
        for i in ids:
            if i in self._special_of_id:
                parts.append(self._special_of_id[i].encode("utf-8"))
            else:
                parts.append(self.vocab[i])
        return b"".join(parts).decode("utf-8", errors="replace")

    # ----------------------------------------------------------- persistence

    def save(self, path: str) -> None:
        data = {
            "version": 1,
            "pattern": SPLIT_PATTERN,
            "merges": self.merges,
            "special_tokens": self.special_tokens,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path) as f:
            data = json.load(f)
        pattern = data.get("pattern")
        if pattern is not None and pattern != SPLIT_PATTERN:
            raise ValueError(
                "tokenizer was saved with a different pre-tokenization pattern; "
                "loading it under today's SPLIT_PATTERN would silently mis-encode. "
                f"saved: {pattern!r} current: {SPLIT_PATTERN!r}"
            )
        return cls([tuple(m) for m in data["merges"]], data["special_tokens"])

    # ------------------------------------------------------------ fast export

    def export_fast(self):
        """Build a HF `tokenizers.Tokenizer` with identical merges.

        Used only to accelerate bulk corpus encoding; call verify_fast on a
        sample before trusting it. Our pure-Python encoder is ground truth.
        """
        from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

        b2u = _bytes_to_unicode()
        to_str = lambda bs: "".join(b2u[b] for b in bs)  # noqa: E731
        vocab = {to_str(tok_bytes): i for i, tok_bytes in self.vocab.items()}
        merges = [(to_str(self.vocab[a]), to_str(self.vocab[b])) for a, b in self.merges]
        fast = Tokenizer(models.BPE(vocab=vocab, merges=merges))
        fast.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        fast.decoder = decoders.ByteLevel()
        return fast

    def verify_fast(self, fast, texts: Iterable[str]) -> None:
        """Assert `fast` produces byte-identical ids to our encoder."""
        for t in texts:
            if fast.encode(t).ids != self.encode(t):
                raise ValueError(f"fast tokenizer mismatch on: {t[:80]!r}")
