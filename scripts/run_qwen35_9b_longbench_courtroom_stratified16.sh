#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
usage: bash scripts/run_qwen35_9b_longbench_courtroom_stratified16.sh <output-dir> [extra pack args...]

Runs the paper-facing stratified LongBench courtroom for Qwen3.5-9B:
- 16 rows per dataset across the full 21-dataset LongBench suite
- comparison cases: exact, systems, streaming_sink_recent, quest_like
- quality-check disabled on the main run for speed
EOF
  exit 1
}

if [[ $# -lt 1 ]]; then
  usage
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$1"
shift

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" && -x /venv/main/bin/python3 ]]; then
  PYTHON_BIN="/venv/main/bin/python3"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'fatal: missing python interpreter\n' >&2
  exit 1
fi

PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" "$ROOT_DIR/scripts/run_qwen35_longbench_pack.py" \
  --model-id Qwen/Qwen3.5-9B \
  --backend torch_cuda \
  --device cuda \
  --torch-dtype float16 \
  --selector-artifact "$ROOT_DIR/benchmarks/results/qwen35_selector_qwen35_9b_suite_20260401/serving_selector_artifact/linear_selector_model.json" \
  --pack stratified_16 \
  --comparison-case-preset paper_headline \
  --no-quality-check \
  --output-dir "$OUTPUT_DIR" \
  "$@"
