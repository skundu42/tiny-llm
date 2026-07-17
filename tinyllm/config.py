"""Model and training configuration presets."""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    n_layer: int = 26
    n_head: int = 20
    n_kv_head: int = 4
    d_model: int = 1280
    d_ff: int = 3456
    seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        assert self.d_model % self.n_head == 0, "d_model must divide evenly into heads"
        assert self.n_head % self.n_kv_head == 0, "n_head must be a multiple of n_kv_head"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


@dataclass
class TrainConfig:
    # data
    data_dir: str = "data/fineweb-edu"
    # batch: tokens per optimizer step, summed across all ranks
    batch_tokens: int = 524_288
    micro_batch_size: int = 16          # sequences per micro-step per rank
    # schedule
    total_steps: int = 19_073           # ~10B tokens / 524288
    warmup_steps: int = 250
    decay_frac: float = 0.2             # final fraction of steps: linear decay to 0
    # optimizer
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    adamw_lr: float = 6e-4
    adamw_betas: tuple[float, float] = (0.9, 0.95)
    adamw_wd: float = 0.1               # embeddings only; norm gains get 0
    grad_clip: float = 1.0
    # runtime
    dtype: str = "auto"                 # auto | bf16 | fp32  (auto: bf16 on cuda, fp32 elsewhere)
    compile: bool = True                # auto-disabled off-cuda with a warning
    seed: int = 1337
    # logging / eval / checkpoints
    out_dir: str = "out/d26"
    log_every: int = 10
    val_every: int = 250
    val_batches: int = 20
    ckpt_every: int = 1000
    sample_every: int = 1000
    wandb_project: str = ""             # empty = wandb disabled


MODEL_PRESETS: dict[str, ModelConfig] = {
    "d26": ModelConfig(),
    "smoke": ModelConfig(n_layer=6, n_head=4, n_kv_head=2, d_model=256, d_ff=768, seq_len=512),
}

TRAIN_PRESETS: dict[str, TrainConfig] = {
    "d26": TrainConfig(),
    "smoke": TrainConfig(
        data_dir="data/smoke",
        batch_tokens=16_384,
        micro_batch_size=8,
        total_steps=600,
        warmup_steps=20,
        compile=False,
        out_dir="out/smoke",
        val_every=100,
        val_batches=8,
        ckpt_every=200,
        sample_every=200,
    ),
}
