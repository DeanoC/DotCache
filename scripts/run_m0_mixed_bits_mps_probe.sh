#!/usr/bin/env bash
set -euo pipefail

MODEL_PRESET="${1:-tinyllama}"
shift || true

case "$MODEL_PRESET" in
  tinyllama)
    MODEL_ID="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    PROMPT_LEN=577
    ;;
  smollm2)
    MODEL_ID="HuggingFaceTB/SmolLM2-360M-Instruct"
    PROMPT_LEN=1024
    ;;
  *)
    echo "usage: bash scripts/run_m0_mixed_bits_mps_probe.sh [tinyllama|smollm2] [extra bench args...]" >&2
    exit 1
    ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

exec .venv/bin/python benchmarks/bench_llama_mixed_bits_profile.py \
  --model-id "$MODEL_ID" \
  --backend torch_mps \
  --device mps \
  --bits-k 4 \
  --bits-v-options 4 3 \
  --max-new-tokens 4 \
  --target-prompt-lengths "$PROMPT_LEN" \
  "$@"
