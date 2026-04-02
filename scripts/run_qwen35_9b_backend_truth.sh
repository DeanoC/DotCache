#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
usage: bash scripts/run_qwen35_9b_backend_truth.sh <output-dir> [extra bench args...]

Runs the narrow backend-truth experiment for Qwen3.5-9B:
  1. exact DotCache lane
  2. shortlist-base DotCache lane
  3. learned-selector DotCache lane
  4. backend comparison report

The learned-selector artifact defaults to:
  benchmarks/results/qwen35_selector_qwen35_9b_suite_20260401/serving_selector_artifact/linear_selector_model.json
EOF
  exit 1
}

if [[ $# -lt 1 ]]; then
  usage
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
OUTPUT_DIR="$1"
shift

LEARNED_SELECTOR_ARTIFACT="${LEARNED_SELECTOR_ARTIFACT:-$REPO_ROOT/benchmarks/results/qwen35_selector_qwen35_9b_suite_20260401/serving_selector_artifact/linear_selector_model.json}"
MODEL_ID="${QWEN35_BACKEND_TRUTH_MODEL_ID:-Qwen/Qwen3.5-9B}"
MAX_NEW_TOKENS="${QWEN35_BACKEND_TRUTH_MAX_NEW_TOKENS:-8}"
TOKENS_PER_PAGE="${QWEN35_BACKEND_TRUTH_TOKENS_PER_PAGE:-16}"
PROMPT_LENGTHS=(${QWEN35_BACKEND_TRUTH_PROMPT_LENGTHS:-"1024 2048"})

mkdir -p "$OUTPUT_DIR"

COMMON_ARGS=(
  --model-id "$MODEL_ID"
  --backend torch_cuda
  --device cuda
  --torch-dtype float16
  --target-prompt-lengths "${PROMPT_LENGTHS[@]}"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --tokens-per-page "$TOKENS_PER_PAGE"
  --continue-on-error
  --profile-backend
)

"$PYTHON_BIN" "$REPO_ROOT/benchmarks/bench_qwen35_attention_subset_dotcache_serving.py" \
  "${COMMON_ARGS[@]}" \
  "$@" \
  > "$OUTPUT_DIR/qwen35_9b_dotcache_exact_serving.jsonl"

"$PYTHON_BIN" "$REPO_ROOT/benchmarks/bench_qwen35_attention_subset_dotcache_serving.py" \
  "${COMMON_ARGS[@]}" \
  --execution-recent-window 1024 \
  --execution-sink-window 256 \
  --execution-relevance-top-k 4 \
  --execution-relevance-mode envelope \
  "$@" \
  > "$OUTPUT_DIR/qwen35_9b_dotcache_shortlist_base_serving.jsonl"

"$PYTHON_BIN" "$REPO_ROOT/benchmarks/bench_qwen35_attention_subset_dotcache_serving.py" \
  "${COMMON_ARGS[@]}" \
  --learned-page-selector-path "$LEARNED_SELECTOR_ARTIFACT" \
  --learned-page-selector-prompt-family cache \
  --learned-page-selector-prompt-variant locality \
  --learned-page-selector-scope KV \
  "$@" \
  > "$OUTPUT_DIR/qwen35_9b_dotcache_learned_selector_serving.jsonl"

"$PYTHON_BIN" "$REPO_ROOT/scripts/report_qwen35_backend_truth.py" \
  --exact "$OUTPUT_DIR/qwen35_9b_dotcache_exact_serving.jsonl" \
  --shortlist "$OUTPUT_DIR/qwen35_9b_dotcache_shortlist_base_serving.jsonl" \
  --learned "$OUTPUT_DIR/qwen35_9b_dotcache_learned_selector_serving.jsonl" \
  --markdown-output "$OUTPUT_DIR/backend_truth_report.md" \
  --json-output "$OUTPUT_DIR/backend_truth_report.json"

echo "$OUTPUT_DIR/backend_truth_report.md"
echo "$OUTPUT_DIR/backend_truth_report.json"
