#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

exec .venv/bin/python benchmarks/bench_qwen2_compare.py \
  --model-id Qwen/Qwen2.5-7B-Instruct \
  --backend torch_cuda \
  --device cuda \
  --default-mode-k M3 \
  --default-mode-v M0 \
  --max-new-tokens 4 \
  --repeat-counts \
  --target-prompt-lengths 1024 2048 4096 \
  --continue-on-error \
  "$@"
