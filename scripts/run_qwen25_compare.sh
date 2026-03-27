#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

exec .venv/bin/python benchmarks/bench_qwen2_compare.py \
  --model-id Qwen/Qwen2.5-3B-Instruct \
  --backend torch_mps \
  --device mps \
  --max-new-tokens 4 \
  --repeat-counts \
  --target-prompt-lengths 1024 2048 \
  --continue-on-error \
  "$@"
