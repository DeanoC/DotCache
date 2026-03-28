#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/env_cuda.sh"

exec .venv/bin/python benchmarks/bench_qwen35_deltanet_statecache_readout.py \
  --model-id Qwen/Qwen3.5-0.8B \
  --backend torch_cuda \
  --device cuda \
  --bits 8 \
  --state-stage post_update_m0 \
  --renorm-interval 0 \
  --max-new-tokens 4 \
  --repeat-counts \
  --target-prompt-lengths 512 1024 \
  --continue-on-error \
  "$@"
