#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-google/gemma-4-E2B}"
PROMPT="${PROMPT:-Cache locality on Apple Silicon is}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-180}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmarks/results/gemma4_apple_smoke_$(date +%Y%m%d)}"

mkdir -p "$OUTPUT_DIR"

.venv/bin/python scripts/run_gemma4_apple_smoke.py \
  --model-id "$MODEL_ID" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --prompt "$PROMPT" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
  --output-dir "$OUTPUT_DIR"

echo "Wrote Gemma 4 Apple smoke results to $OUTPUT_DIR"
