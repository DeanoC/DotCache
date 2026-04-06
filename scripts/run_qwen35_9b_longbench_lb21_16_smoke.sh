#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
usage: bash scripts/run_qwen35_9b_longbench_lb21_16_smoke.sh <output-dir> [extra pack args...]

Runs the frozen LB21-16 smoke pass for Qwen3.5-9B:
- fixed 2 rows per dataset from the immutable LB21-16 manifest
- comparison cases: exact, systems, streaming_sink_recent, quest_like
- quality-check enabled
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
  --prompt-pack "$ROOT_DIR/configs/prompt_packs/longbench_lb21_16_smoke_v1.json" \
  --comparison-case-preset paper_headline \
  --quality-check \
  --output-dir "$OUTPUT_DIR" \
  "$@"
