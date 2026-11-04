#!/usr/bin/env bash

# Train the self-contained MLX experiment on Apple Silicon.
# This is intentionally separate from runcpu.sh, which runs the original
# PyTorch CPU/MPS pretraining + SFT pipeline.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "This runner requires macOS on Apple Silicon (arm64) and MLX." >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    cat >&2 <<'EOF'
uv is required for the nanochat MLX environment.

Install it with:
  curl -LsSf https://astral.sh/uv/install.sh | sh

Then restart the terminal and run this script again.
EOF
    exit 1
fi

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"

PRESET="${NANOCHAT_MLX_PRESET:-hour}"
ARGS=(--preset "$PRESET")
if [[ -n "${NANOCHAT_MLX_MINUTES:-}" ]]; then
    ARGS+=(--minutes "$NANOCHAT_MLX_MINUTES")
fi
if [[ -n "${NANOCHAT_MLX_ITERATIONS:-}" ]]; then
    ARGS+=(--iterations "$NANOCHAT_MLX_ITERATIONS")
fi
if [[ -n "${NANOCHAT_MLX_OUT:-}" ]]; then
    ARGS+=(--out "$NANOCHAT_MLX_OUT")
fi

cd "$ROOT"
uv sync --extra mlx
exec uv run --extra mlx python scripts/mlx_train.py "${ARGS[@]}"
