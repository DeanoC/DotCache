#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
usage: bash scripts/run_qwen_results_matrix.sh <output-dir> [extra runner args...]

Runs the checked-in Qwen results matrix manifest and then renders the combined report.
EOF
  exit 1
}

if [[ $# -lt 1 ]]; then
  usage
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
OUTPUT_DIR="$1"
shift

"$PYTHON_BIN" "$ROOT_DIR/scripts/run_qwen_results_matrix.py" \
  --output-dir "$OUTPUT_DIR" \
  "$@"

"$PYTHON_BIN" "$ROOT_DIR/scripts/report_qwen_results_matrix.py" \
  --manifest "$ROOT_DIR/configs/benchmark_matrices/qwen_results_matrix_v1.json" \
  --output-dir "$OUTPUT_DIR" \
  --markdown-output "$OUTPUT_DIR/qwen_results_matrix.md" \
  --json-output "$OUTPUT_DIR/qwen_results_matrix.json"

echo "$OUTPUT_DIR/qwen_results_matrix.md"
echo "$OUTPUT_DIR/qwen_results_matrix.json"
