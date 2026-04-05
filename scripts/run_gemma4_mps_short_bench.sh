#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-benchmarks/results/gemma4_mps_short_bench_$(date +%Y%m%d)}"

.venv/bin/python scripts/run_gemma4_profile_sweep.py \
  --model-id google/gemma-4-E2B \
  --backend torch_mps \
  --device mps \
  --torch-dtype bfloat16 \
  --profiles balanced \
  --target-prompt-lengths 32 \
  --max-new-tokens-list 1 \
  --output-dir "$OUTPUT_DIR" \
  "$@"
