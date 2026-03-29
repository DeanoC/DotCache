#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

exec .venv/bin/python benchmarks/bench_turboquant_external.py \
  --model-id HuggingFaceTB/SmolLM2-360M-Instruct-GGUF \
  --tokenizer-model-id HuggingFaceTB/SmolLM2-360M-Instruct \
  --target-prompt-lengths 4096 16384 32768 \
  --max-new-tokens 4 \
  --configs q8_0 turbo3_uniform turbo3_la1 turbo3_la5 \
  --continue-on-error \
  "$@"
