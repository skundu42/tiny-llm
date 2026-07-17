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
        positive_ints = {
            "vocab_size": self.vocab_size,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_kv_head": self.n_kv_head,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "seq_len": self.seq_len,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {self.rope_theta}")
        if self.norm_eps <= 0:
            raise ValueError(f"norm_eps must be positive, got {self.norm_eps}")
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must divide evenly into heads")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be a multiple of n_kv_head")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


@dataclass
class TrainConfig:
    # data
    data_dir: str = "data/fineweb-edu"
    # batch: tokens per optimizer step, summed across all ranks
    batch_tokens: int = 524_288
    micro_batch_size: int = 8           # sequences per micro-step per rank
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

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        positive_ints = {
            "batch_tokens": self.batch_tokens,
            "micro_batch_size": self.micro_batch_size,
            "total_steps": self.total_steps,
            "log_every": self.log_every,
            "val_every": self.val_every,
            "val_batches": self.val_batches,
            "ckpt_every": self.ckpt_every,
            "sample_every": self.sample_every,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not 0.0 <= self.decay_frac <= 1.0:
            raise ValueError("decay_frac must be between 0 and 1")
        if self.muon_lr < 0 or self.adamw_lr < 0:
            raise ValueError("learning rates must be non-negative")
        if not 0.0 <= self.muon_momentum < 1.0:
            raise ValueError("muon_momentum must be in [0, 1)")
        if any(not 0.0 <= beta < 1.0 for beta in self.adamw_betas):
            raise ValueError("AdamW betas must be in [0, 1)")
        if self.adamw_wd < 0:
            raise ValueError("adamw_wd must be non-negative")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")
        if self.dtype not in {"auto", "bf16", "fp32"}:
            raise ValueError(f"unsupported dtype: {self.dtype}")


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
