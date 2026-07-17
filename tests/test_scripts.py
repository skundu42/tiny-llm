import json
import sys

import pytest

from scripts import prepare_data, train_tokenizer


def test_train_tokenizer_accepts_bare_output_path(tmp_path, monkeypatch):
    docs = iter([{"text": "abc"}, {"text": "held out"}])
    monkeypatch.setattr(train_tokenizer, "load_dataset", lambda *args, **kwargs: docs)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_tokenizer.py",
            "--out",
            "tokenizer.json",
            "--max-bytes",
            "3",
            "--vocab-size",
            "257",
            "--num-proc",
            "1",
            "--min-word-freq",
            "1",
        ],
    )

    train_tokenizer.main()

    data = json.loads((tmp_path / "tokenizer.json").read_text())
    assert data["special_tokens"] == {"<|endoftext|>": 256}


@pytest.mark.parametrize(
    "args, message",
    [
        (["--shard-tokens", "0"], "--shard-tokens must be positive"),
        (["--val-tokens", "0"], "--val-tokens must be positive"),
        (["--max-tokens", "-1"], "--max-tokens must be non-negative"),
        (
            ["--max-tokens", "100", "--val-tokens", "100"],
            "--max-tokens must exceed --val-tokens",
        ),
    ],
)
def test_prepare_data_rejects_invalid_budgets_before_io(args, message, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["prepare_data.py", *args])
    monkeypatch.setattr(
        prepare_data,
        "load_dataset",
        lambda *args, **kwargs: pytest.fail("dataset should not be loaded"),
    )

    with pytest.raises(SystemExit):
        prepare_data.main()

    assert message in capsys.readouterr().err
