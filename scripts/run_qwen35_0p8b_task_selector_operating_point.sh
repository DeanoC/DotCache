#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <output-dir> <selector-artifact> <reference-jsonl> [extra runner args...]" >&2
  exit 1
fi

OUTPUT_DIR="$1"
SELECTOR_ARTIFACT="$2"
REFERENCE_JSONL="$3"
shift 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

JSONL_PATH="${OUTPUT_DIR}/qwen35_0p8b_task_selector_compare.jsonl"
MARKDOWN_PATH="${OUTPUT_DIR}/task_selector_compare.md"
JSON_PATH="${OUTPUT_DIR}/task_selector_compare.json"

mkdir -p "${OUTPUT_DIR}"

./.venv/bin/python scripts/run_qwen35_task_selector_compare.py \
  --model-id Qwen/Qwen3.5-0.8B \
  --backend torch_mps \
  --device mps \
  --torch-dtype float16 \
  --selector-artifact "${SELECTOR_ARTIFACT}" \
  --profiles quality systems \
  --prompt-lengths 512 1024 \
  --warmup-runs 0 \
  --measured-runs 1 \
  --max-new-tokens-retrieval 64 \
  --max-new-tokens-reasoning 512 \
  --max-new-tokens-instruction 32 \
  --output "${JSONL_PATH}" \
  "$@"

./.venv/bin/python scripts/report_qwen35_task_selector_compare.py \
  --input "${JSONL_PATH}" \
  --reference-input "${REFERENCE_JSONL}" \
  --markdown-output "${MARKDOWN_PATH}" \
  --json-output "${JSON_PATH}"

echo "Wrote JSONL: ${JSONL_PATH}"
echo "Wrote report: ${MARKDOWN_PATH}"
