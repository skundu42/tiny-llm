import pytest

from tinyllm.tokenizer import ENDOFTEXT, BPETokenizer

CORPUS = [
    "The quick brown fox jumps over the lazy dog. " * 20,
    "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)\n" * 10,
    "Numbers 123 4567 89, punctuation!? (brackets) [more] {braces}. " * 15,
    "Unicode: héllo wörld 🌍 日本語のテキスト émojis 🎉🎊 " * 10,
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. " * 5,
    "Pack my box with five dozen liquor jugs; how vexingly quick daft zebras jump! Sphinx of black quartz, judge my vow. Grumpy wizards make toxic brew for the evil queen and jack. " * 5,
]

ROUNDTRIP_STRINGS = [
    "hello world",
    "héllo wörld 🌍",
    "def f(x):\n    return x * 2",
    "日本語のテキスト",
    "  leading spaces\t\ttabs\n\nnewlines  ",
    "<|endoftext|>",
    "",
]


@pytest.fixture(scope="module")
def tok() -> BPETokenizer:
    return BPETokenizer.train(CORPUS, vocab_size=512)


def test_vocab_size_and_eot(tok):
    assert tok.vocab_size == 512
    assert tok.eot_id == 511
    assert tok.special_tokens == {ENDOFTEXT: 511}
    assert len(tok.merges) == 512 - 256 - 1


@pytest.mark.parametrize("s", ROUNDTRIP_STRINGS)
def test_roundtrip(tok, s):
    assert tok.decode(tok.encode(s)) == s


def test_encode_never_emits_special(tok):
    assert tok.eot_id not in tok.encode("<|endoftext|>")


def test_decode_special(tok):
    assert tok.decode([tok.eot_id]) == ENDOFTEXT


def test_first_merge_is_most_frequent_pair():
    t = BPETokenizer.train(["aaabdaaabac"], vocab_size=259)
    assert t.merges[0] == (97, 97)  # 'aa' dominates


def test_deterministic_training():
    a = BPETokenizer.train(CORPUS, vocab_size=400)
    b = BPETokenizer.train(CORPUS, vocab_size=400)
    assert a.merges == b.merges


def test_compression(tok):
    text = CORPUS[0]
    assert len(tok.encode(text)) < len(text.encode("utf-8"))


def test_save_load_roundtrip(tok, tmp_path):
    p = str(tmp_path / "tok.json")
    tok.save(p)
    tok2 = BPETokenizer.load(p)
    assert tok2.merges == tok.merges
    assert tok2.special_tokens == tok.special_tokens
    s = "The quick brown fox! 🌍"
    assert tok2.encode(s) == tok.encode(s)


def test_min_word_freq_filters_rare_words():
    texts = ["common common common rare"]
    t = BPETokenizer.train(texts, vocab_size=300, min_word_freq=2)
    # 'rare' appeared once -> excluded from training, but still encodable at byte level
    assert t.decode(t.encode("rare")) == "rare"


def test_fast_export_parity(tok):
    fast = tok.export_fast()
    samples = ROUNDTRIP_STRINGS + CORPUS
    for s in samples:
        assert fast.encode(s).ids == tok.encode(s), f"mismatch on {s!r}"


def test_verify_fast_passes(tok):
    tok.verify_fast(tok.export_fast(), CORPUS)


def test_verify_fast_raises_on_mismatch(tok):
    class Bogus:
        def encode(self, s):
            class R:
                ids = [0]
            return R()
    import pytest
    with pytest.raises(ValueError):
        tok.verify_fast(Bogus(), ["hello"])
