#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_PATH="${1:-$ROOT_DIR/benchmarks/results/qwen35_cuda_shortlist_large_context_probe.jsonl}"

export PYTHONPATH="$ROOT_DIR"
export PATH="$ROOT_DIR/.venv/bin:$PATH"

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/run_qwen35_cuda_shortlist_probe.py" \
  --layer-profile "$ROOT_DIR/configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_third_pass.yaml" \
  --contexts 32768 49152 \
  --cases exact shortlist_base shortlist_l23_ctx \
  --timeout-seconds 900 \
  --profile-backend \
  --output "$OUTPUT_PATH"
