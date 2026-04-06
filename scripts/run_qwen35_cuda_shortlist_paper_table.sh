#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_PATH="${1:-$ROOT_DIR/benchmarks/results/qwen35_cuda_shortlist_probe.jsonl}"

export PYTHONPATH="$ROOT_DIR"
export PATH="$ROOT_DIR/.venv/bin:$PATH"

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/run_qwen35_cuda_shortlist_probe.py" \
  --contexts 4096 8192 16384 \
  --cases exact shortlist_base shortlist_l23_ctx \
  --profile-backend \
  --output "$OUTPUT_PATH"
