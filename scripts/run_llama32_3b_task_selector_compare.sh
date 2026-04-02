#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <output-dir> [extra runner args...]" >&2
  exit 1
fi

OUTPUT_DIR="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

JSONL_PATH="${OUTPUT_DIR}/llama32_3b_task_selector_compare.jsonl"
MARKDOWN_PATH="${OUTPUT_DIR}/task_selector_compare.md"
JSON_PATH="${OUTPUT_DIR}/task_selector_compare.json"

mkdir -p "${OUTPUT_DIR}"

./.venv/bin/python scripts/run_llama_task_selector_compare.py \
  --model-id meta-llama/Llama-3.2-3B-Instruct \
  --backend torch_cuda \
  --device cuda \
  --torch-dtype float16 \
  --selector-artifact benchmarks/results/llama32_selector_suite_20260401/serving_selector_artifact/linear_selector_model.json \
  --prompt-lengths 1024 2048 \
  --warmup-runs 1 \
  --measured-runs 5 \
  --output "${JSONL_PATH}" \
  "$@"

./.venv/bin/python scripts/report_llama_task_selector_compare.py \
  --input "${JSONL_PATH}" \
  --markdown-output "${MARKDOWN_PATH}" \
  --json-output "${JSON_PATH}"

echo "Wrote JSONL: ${JSONL_PATH}"
echo "Wrote report: ${MARKDOWN_PATH}"
