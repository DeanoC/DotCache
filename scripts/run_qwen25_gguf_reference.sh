#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

exec .venv/bin/python benchmarks/bench_gguf_external.py \
  --model-id Qwen/Qwen2.5-3B-Instruct-GGUF \
  --tokenizer-model-id Qwen/Qwen2.5-3B-Instruct \
  --max-new-tokens 4 \
  --target-prompt-lengths 1024 2048 \
  --continue-on-error \
  "$@"
