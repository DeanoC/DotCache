#!/usr/bin/env bash
set -euo pipefail

MODEL_PRESET="${1:-tinyllama}"

case "$MODEL_PRESET" in
  tinyllama)
    MODEL_ID="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    COMPARE_ARGS=(--target-prompt-lengths 289 --max-new-tokens 4)
    LOSS_ARGS=(--sequence-length 320 --prefix-length 288 --eval-steps 32)
    ;;
  smollm2)
    MODEL_ID="HuggingFaceTB/SmolLM2-360M-Instruct"
    COMPARE_ARGS=(--target-prompt-lengths 1024 --max-new-tokens 4)
    LOSS_ARGS=(--sequence-length 1040 --prefix-length 1024 --eval-steps 16)
    ;;
  *)
    echo "usage: bash scripts/run_turbo3_mps_suite.sh [tinyllama|smollm2]" >&2
    exit 1
    ;;
esac

COMMON_ARGS=(
  --backend torch_mps
  --device mps
  --model-id "$MODEL_ID"
  --default-mode-k T3
  --default-mode-v T3
  --quant-scheme-k turbo3
  --quant-scheme-v turbo3
)

.venv/bin/python benchmarks/bench_llama_compare.py "${COMMON_ARGS[@]}" "${COMPARE_ARGS[@]}"
.venv/bin/python benchmarks/bench_llama_loss.py "${COMMON_ARGS[@]}" "${LOSS_ARGS[@]}"
