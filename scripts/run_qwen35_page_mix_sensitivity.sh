#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
usage: bash scripts/run_qwen35_page_mix_sensitivity.sh <output-dir> [extra bench args...]

Runs a learned-selector page-mix sensitivity sweep for Qwen3.5 0.8B by:
  1. generating bias-shifted selector artifacts
  2. benchmarking each variant on 1024 and 2048 prompt lengths
  3. writing a compact sensitivity report

This wrapper pins the serving benchmark to `--learned-page-selector-profile quality`
so the generated bias-shifted artifacts are measured directly, without stacking the
default Qwen `systems` profile on top.

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
TARGET_CANDIDATE="${TARGET_CANDIDATE:-M3/affine/4/float16}"
OFFSET_STRING="${SELECTOR_LOGIT_OFFSETS:--2.0 -1.0 -0.5 0.0 0.5 1.0 2.0}"
read -r -a OFFSETS <<< "$OFFSET_STRING"

ARTIFACT_DIR="$OUTPUT_DIR/artifacts"
RESULTS_DIR="$OUTPUT_DIR/results"
mkdir -p "$ARTIFACT_DIR" "$RESULTS_DIR"

MANIFEST_PATH="$("$PYTHON_BIN" "$REPO_ROOT/scripts/build_selector_logit_sweep.py" \
  --artifact "$LEARNED_SELECTOR_ARTIFACT" \
  --output-dir "$ARTIFACT_DIR" \
  --target-candidate "$TARGET_CANDIDATE" \
  --offsets "${OFFSETS[@]}")"

export MANIFEST_PATH
export RESULTS_DIR
while IFS=$'\t' read -r variant artifact_path; do
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
    --learned-page-selector-path "$artifact_path" \
    --learned-page-selector-prompt-family cache \
    --learned-page-selector-prompt-variant locality \
    --learned-page-selector-profile quality \
    "$@" \
    > "$RESULTS_DIR/$variant.jsonl"
done < <(
  "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

manifest = json.loads(Path(os.environ["MANIFEST_PATH"]).read_text(encoding="utf-8"))
for entry in manifest.get("variants", []):
    print(f"{entry['variant']}\t{entry['artifact_path']}")
PY
)

"$PYTHON_BIN" "$REPO_ROOT/scripts/report_qwen35_page_mix_sensitivity.py" \
  --manifest "$MANIFEST_PATH" \
  --results-dir "$RESULTS_DIR" \
  --markdown-output "$OUTPUT_DIR/page_mix_sensitivity_report.md" \
  --json-output "$OUTPUT_DIR/page_mix_sensitivity_report.json"

echo "$OUTPUT_DIR/page_mix_sensitivity_report.md"
echo "$OUTPUT_DIR/page_mix_sensitivity_report.json"
