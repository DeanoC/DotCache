#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

exec .venv/bin/python benchmarks/bench_llama_compare.py \
  --model-id HuggingFaceTB/SmolLM2-360M-Instruct \
  --backend torch_mps \
  --device mps \
  --max-new-tokens 4 \
  --repeat-counts \
  --target-prompt-lengths 256 512 1024 1536 2048 \
  --continue-on-error \
  "$@"
