#!/usr/bin/env bash
# Bootstrap a fresh Ubuntu GPU box (e.g. Lambda/RunPod H100) for tiny-llm.
set -euo pipefail

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv sync

echo "== setup complete. Suggested run sequence =="
if [ ! -f tokenizer/tokenizer.json ]; then
  echo "uv run python scripts/train_tokenizer.py            # ~30-60 min CPU"
fi
cat <<'EOF'
uv run python scripts/prepare_data.py                        # ~1-3 h CPU, ~20 GB disk
tmux new -s train
uv run python -m tinyllm.train --config d26 --tokenizer tokenizer/tokenizer.json
# multi-GPU instead:
# uv run torchrun --standalone --nproc_per_node=8 -m tinyllm.train --config d26 --tokenizer tokenizer/tokenizer.json
EOF
