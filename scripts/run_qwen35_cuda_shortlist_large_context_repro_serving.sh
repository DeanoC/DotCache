#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$ROOT_DIR/benchmarks/results/qwen35_cuda_shortlist_large_context_repro_serving}"
REPEATS="${REPEATS:-3}"

export PYTHONPATH="$ROOT_DIR"
export PATH="$ROOT_DIR/.venv/bin:$PATH"

mkdir -p "$OUTPUT_DIR"

run_probe() {
  local mode="$1"
  local repeat_index="$2"
  local output_path="$OUTPUT_DIR/${mode}_repeat${repeat_index}.jsonl"
  rm -f "$output_path"
  if [[ "$mode" == "forced_grouped" ]]; then
    DOTCACHE_QWEN35_FORCE_GROUPED_BATCHING=1 \
      "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/run_qwen35_cuda_shortlist_probe.py" \
      --layer-profile "$ROOT_DIR/configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_third_pass.yaml" \
      --contexts 32768 49152 \
      --cases shortlist_base shortlist_l23_ctx \
      --timeout-seconds 900 \
      --profile-backend \
      --output "$output_path"
  else
    "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/run_qwen35_cuda_shortlist_probe.py" \
      --layer-profile "$ROOT_DIR/configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_third_pass.yaml" \
      --contexts 32768 49152 \
      --cases shortlist_base shortlist_l23_ctx \
      --timeout-seconds 900 \
      --profile-backend \
      --output "$output_path"
  fi
}

for repeat_index in $(seq 1 "$REPEATS"); do
  run_probe default "$repeat_index"
done

for repeat_index in $(seq 1 "$REPEATS"); do
  run_probe forced_grouped "$repeat_index"
done
