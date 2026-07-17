from tinyllm.config import MODEL_PRESETS, TRAIN_PRESETS, ModelConfig


def test_presets_exist():
    assert set(MODEL_PRESETS) == {"d26", "smoke"} == set(TRAIN_PRESETS)


def test_d26_shape():
    c = MODEL_PRESETS["d26"]
    assert (c.n_layer, c.d_model, c.n_head, c.n_kv_head) == (26, 1280, 20, 4)
    assert (c.d_ff, c.seq_len, c.vocab_size) == (3456, 2048, 32768)
    assert c.head_dim == 64


def test_smoke_shape():
    c = MODEL_PRESETS["smoke"]
    assert (c.n_layer, c.d_model, c.head_dim) == (6, 256, 64)
    assert c.vocab_size == 32768


def test_invalid_heads_rejected():
    import pytest
    with pytest.raises(AssertionError):
        ModelConfig(n_head=7, n_kv_head=3)


def test_batch_math_divisible():
    t = TRAIN_PRESETS["d26"]
    m = MODEL_PRESETS["d26"]
    assert t.batch_tokens % (t.micro_batch_size * m.seq_len) == 0
    ts, ms = TRAIN_PRESETS["smoke"], MODEL_PRESETS["smoke"]
    assert ts.batch_tokens % (ts.micro_batch_size * ms.seq_len) == 0
