#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
usage: bash scripts/run_qwen35_9b_selector_quality_compare.sh <output-dir> [extra bench args...]

Runs a Qwen3.5-9B quality-oriented comparison across:
  1. exact DotCache
  2. learned selector with the quality profile
  3. learned selector with the systems profile
  4. compact quality-vs-speed report

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
MODEL_ID="${QWEN35_SELECTOR_QUALITY_MODEL_ID:-Qwen/Qwen3.5-9B}"
MAX_NEW_TOKENS="${QWEN35_SELECTOR_QUALITY_MAX_NEW_TOKENS:-8}"
TOKENS_PER_PAGE="${QWEN35_SELECTOR_QUALITY_TOKENS_PER_PAGE:-16}"
PROMPT_LENGTHS_STRING="${QWEN35_SELECTOR_QUALITY_PROMPT_LENGTHS:-1024 2048}"
read -r -a PROMPT_LENGTHS <<< "$PROMPT_LENGTHS_STRING"

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
  --quality-check
)

"$PYTHON_BIN" "$REPO_ROOT/benchmarks/bench_qwen35_attention_subset_dotcache_serving.py" \
  "${COMMON_ARGS[@]}" \
  "$@" \
  > "$OUTPUT_DIR/qwen35_9b_exact_quality.jsonl"

"$PYTHON_BIN" "$REPO_ROOT/benchmarks/bench_qwen35_attention_subset_dotcache_serving.py" \
  "${COMMON_ARGS[@]}" \
  --learned-page-selector-path "$LEARNED_SELECTOR_ARTIFACT" \
  --learned-page-selector-prompt-family cache \
  --learned-page-selector-prompt-variant locality \
  --learned-page-selector-profile quality \
  "$@" \
  > "$OUTPUT_DIR/qwen35_9b_learned_quality_profile.jsonl"

"$PYTHON_BIN" "$REPO_ROOT/benchmarks/bench_qwen35_attention_subset_dotcache_serving.py" \
  "${COMMON_ARGS[@]}" \
  --learned-page-selector-path "$LEARNED_SELECTOR_ARTIFACT" \
  --learned-page-selector-prompt-family cache \
  --learned-page-selector-prompt-variant locality \
  --learned-page-selector-profile systems \
  "$@" \
  > "$OUTPUT_DIR/qwen35_9b_learned_systems_profile.jsonl"

"$PYTHON_BIN" "$REPO_ROOT/scripts/report_qwen35_selector_quality_compare.py" \
  --exact "$OUTPUT_DIR/qwen35_9b_exact_quality.jsonl" \
  --quality "$OUTPUT_DIR/qwen35_9b_learned_quality_profile.jsonl" \
  --systems "$OUTPUT_DIR/qwen35_9b_learned_systems_profile.jsonl" \
  --markdown-output "$OUTPUT_DIR/selector_quality_compare.md" \
  --json-output "$OUTPUT_DIR/selector_quality_compare.json"

echo "$OUTPUT_DIR/selector_quality_compare.md"
echo "$OUTPUT_DIR/selector_quality_compare.json"
