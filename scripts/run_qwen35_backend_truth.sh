#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
usage: bash scripts/run_qwen35_backend_truth.sh <output-dir> [extra sweep args...]

Runs the narrow backend-truth experiment for Qwen3.5 0.8B:
  1. dense / statecache / exact DotCache sweep
  2. shortlist-base DotCache lane
  3. learned-selector DotCache lane
  4. backend comparison report

The learned-selector artifact defaults to:
  benchmarks/results/qwen35_selector_qwen35_4b_suite_20260401_longer/serving_selector_artifact/linear_selector_model.json
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

LEARNED_SELECTOR_ARTIFACT="${LEARNED_SELECTOR_ARTIFACT:-$REPO_ROOT/benchmarks/results/qwen35_selector_qwen35_4b_suite_20260401_longer/serving_selector_artifact/linear_selector_model.json}"

mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" "$REPO_ROOT/scripts/run_qwen35_serving_sweep.py" \
  --contexts 1024 2048 \
  --max-new-tokens 4 \
  --output-dir "$OUTPUT_DIR" \
  --skip-turboquant \
  --dotcache-profile-backend \
  "$@"

"$PYTHON_BIN" "$REPO_ROOT/benchmarks/bench_qwen35_attention_subset_dotcache_serving.py" \
  --model-id Qwen/Qwen3.5-0.8B \
  --backend torch_cuda \
  --device cuda \
  --torch-dtype float16 \
  --layer-profile "$REPO_ROOT/configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_third_pass.yaml" \
  --repeat-counts \
  --target-prompt-lengths 1024 2048 \
  --max-new-tokens 4 \
  --continue-on-error \
  --profile-backend \
  --execution-recent-window 1024 \
  --execution-sink-window 256 \
  --execution-relevance-top-k 4 \
  --execution-relevance-mode envelope \
  > "$OUTPUT_DIR/qwen35_0p8b_dotcache_shortlist_base_serving.jsonl"

"$PYTHON_BIN" "$REPO_ROOT/benchmarks/bench_qwen35_attention_subset_dotcache_serving.py" \
  --model-id Qwen/Qwen3.5-0.8B \
  --backend torch_cuda \
  --device cuda \
  --torch-dtype float16 \
  --layer-profile "$REPO_ROOT/configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_third_pass.yaml" \
  --repeat-counts \
  --target-prompt-lengths 1024 2048 \
  --max-new-tokens 4 \
  --continue-on-error \
  --profile-backend \
  --learned-page-selector-path "$LEARNED_SELECTOR_ARTIFACT" \
  --learned-page-selector-prompt-family cache \
  --learned-page-selector-prompt-variant locality \
  > "$OUTPUT_DIR/qwen35_0p8b_dotcache_learned_selector_serving.jsonl"

"$PYTHON_BIN" "$REPO_ROOT/scripts/report_qwen35_backend_truth.py" \
  --exact "$OUTPUT_DIR/qwen35_0p8b_dotcache_serving_sweep.jsonl" \
  --shortlist "$OUTPUT_DIR/qwen35_0p8b_dotcache_shortlist_base_serving.jsonl" \
  --learned "$OUTPUT_DIR/qwen35_0p8b_dotcache_learned_selector_serving.jsonl" \
  --markdown-output "$OUTPUT_DIR/backend_truth_report.md" \
  --json-output "$OUTPUT_DIR/backend_truth_report.json"

echo "$OUTPUT_DIR/backend_truth_report.md"
echo "$OUTPUT_DIR/backend_truth_report.json"
