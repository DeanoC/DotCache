#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
usage: bash scripts/run_qwen35_27b_longbench_pack.sh <output-dir> [extra pack args...]

Runs the generic LongBench pack compare for Qwen3.5-27B.
Default pack is `mini`; pass `--pack medium` or `--pack full` for broader coverage.
EOF
  exit 1
}

if [[ $# -lt 1 ]]; then
  usage
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$1"
shift

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/run_qwen35_longbench_pack.py" \
  --model-id Qwen/Qwen3.5-27B \
  --backend torch_cuda \
  --device cuda \
  --torch-dtype float16 \
  --selector-artifact "$ROOT_DIR/benchmarks/results/qwen35_selector_qwen35_27b_suite_20260404/serving_selector_artifact/linear_selector_model.json" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
