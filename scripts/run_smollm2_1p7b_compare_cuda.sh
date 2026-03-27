#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

exec .venv/bin/python benchmarks/bench_llama_compare.py \
  --model-id HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --backend torch_cuda \
  --device cuda \
  --default-mode-k M0 \
  --default-mode-v M0 \
  --max-new-tokens 4 \
  --repeat-counts \
  --target-prompt-lengths 1024 2048 \
  --continue-on-error \
  "$@"
