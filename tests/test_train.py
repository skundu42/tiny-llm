import numpy as np
import torch

from tinyllm.config import ModelConfig, TrainConfig
from tinyllm.data import ShardWriter
from tinyllm.train import lr_scale, run


def _micro_cfgs(tmp_path, steps=60):
    mcfg = ModelConfig(vocab_size=64, n_layer=2, n_head=2, n_kv_head=1,
                       d_model=64, d_ff=128, seq_len=64)
    tcfg = TrainConfig(
        data_dir=str(tmp_path / "data"), out_dir=str(tmp_path / "out"),
        batch_tokens=512, micro_batch_size=4, total_steps=steps, warmup_steps=5,
        muon_lr=0.02, adamw_lr=1e-3, compile=False, dtype="fp32",
        log_every=10, val_every=10_000, val_batches=2, ckpt_every=5,
        sample_every=10_000,
    )
    return mcfg, tcfg


def _write_cyclic_data(tmp_path):
    d = tmp_path / "data"
    tokens = np.tile(np.arange(32), 4000).tolist()  # perfectly predictable cycle
    for split in ("train", "val"):
        w = ShardWriter(str(d), split, shard_tokens=100_000)
        w.write(tokens)
        w.close()


def test_lr_scale_wsd():
    assert lr_scale(0, 100, 10, 0.2) == 0.1
    assert lr_scale(9, 100, 10, 0.2) == 1.0
    assert lr_scale(50, 100, 10, 0.2) == 1.0
    assert lr_scale(80, 100, 10, 0.2) == 1.0
    assert 0.0 < lr_scale(99, 100, 10, 0.2) <= 0.06
    scales = [lr_scale(s, 100, 10, 0.2) for s in range(100)]
    assert max(scales) == 1.0 and min(scales) > 0.0


def test_overfits_cyclic_data(tmp_path):
    _write_cyclic_data(tmp_path)
    mcfg, tcfg = _micro_cfgs(tmp_path, steps=80)
    result = run(mcfg, tcfg, device="cpu")
    first = result["history"][0][1]
    last = result["history"][-1][1]
    assert last < 0.5, f"final loss {last} too high"
    assert last < first / 5


def test_resume_is_bit_exact(tmp_path):
    _write_cyclic_data(tmp_path)
    mcfg, tcfg = _micro_cfgs(tmp_path, steps=10)

    run(mcfg, tcfg, device="cpu")
    full = torch.load(tmp_path / "out" / "ckpt_last.pt", weights_only=False)

    import shutil
    shutil.rmtree(tmp_path / "out")
    mcfg2, tcfg2 = _micro_cfgs(tmp_path, steps=5)
    run(mcfg2, tcfg2, device="cpu")
    mcfg3, tcfg3 = _micro_cfgs(tmp_path, steps=10)
    run(mcfg3, tcfg3, device="cpu", resume=True)
    resumed = torch.load(tmp_path / "out" / "ckpt_last.pt", weights_only=False)

    assert full["step"] == resumed["step"] == 10
    for k in full["model"]:
        assert torch.equal(full["model"][k], resumed["model"][k]), f"mismatch in {k}"


def test_checkpoint_contains_configs(tmp_path):
    _write_cyclic_data(tmp_path)
    mcfg, tcfg = _micro_cfgs(tmp_path, steps=6)
    run(mcfg, tcfg, device="cpu")
    ckpt = torch.load(tmp_path / "out" / "ckpt_last.pt", weights_only=False)
    assert ckpt["model_cfg"]["n_layer"] == 2
    assert ckpt["train_cfg"]["total_steps"] == 6
    assert len(ckpt["optimizers"]) == 2
