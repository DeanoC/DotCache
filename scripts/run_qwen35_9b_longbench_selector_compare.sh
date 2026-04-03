#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$ROOT_DIR/benchmarks/results/qwen35_9b_longbench_selector_compare_20260403}"
shift || true

JSONL_PATH="$OUTPUT_DIR/qwen35_9b_longbench_selector_compare.jsonl"
MARKDOWN_PATH="$OUTPUT_DIR/longbench_selector_compare.md"
JSON_PATH="$OUTPUT_DIR/longbench_selector_compare.json"

export PYTHONPATH="$ROOT_DIR"
export PATH="$ROOT_DIR/.venv/bin:$PATH"

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/run_qwen35_longbench_selector_compare.py" \
  --model-id Qwen/Qwen3.5-9B \
  --backend torch_cuda \
  --device cuda \
  --torch-dtype float16 \
  --selector-artifact "$ROOT_DIR/benchmarks/results/qwen35_selector_qwen35_9b_suite_20260401/serving_selector_artifact/linear_selector_model.json" \
  --prompt-pack "$ROOT_DIR/configs/prompt_packs/qwen35_cuda_longbench_qa_pack_v1.json" \
  --max-prompt-tokens 4096 8192 \
  --warmup-runs 1 \
  --measured-runs 5 \
  --quality-check \
  --output "$JSONL_PATH" \
  "$@"

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/report_qwen35_longbench_selector_compare.py" \
  --input "$JSONL_PATH" \
  --markdown-output "$MARKDOWN_PATH" \
  --json-output "$JSON_PATH" \
  >/dev/null

printf 'Wrote:\n- %s\n- %s\n- %s\n' "$JSONL_PATH" "$MARKDOWN_PATH" "$JSON_PATH"
