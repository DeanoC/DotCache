#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

exec .venv/bin/python benchmarks/bench_turboquant_external.py \
  --model-id TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  --tokenizer-model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --target-prompt-lengths 4096 16384 32768 \
  --max-new-tokens 4 \
  --configs q8_0 turbo3_uniform turbo3_la1 turbo3_la5 \
  --continue-on-error \
  "$@"
