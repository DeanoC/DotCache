#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_PATH="${1:-$ROOT_DIR/benchmarks/results/qwen35_cuda_needle_protocol.jsonl}"

export PYTHONPATH="$ROOT_DIR"
export PATH="$ROOT_DIR/.venv/bin:$PATH"

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/run_qwen35_cuda_needle_probe.py" \
  --layer-profile "$ROOT_DIR/configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_third_pass.yaml" \
  --contexts 32768 49152 \
  --cases exact shortlist_base shortlist_l23_ctx \
  --timeout-seconds 1200 \
  --max-new-tokens 12 \
  --needle-position-fraction 0.5 \
  --evaluation-split held_out \
  --evaluation-lane systems \
  --evaluation-prompt-family needle_in_a_haystack \
  --evaluation-prompt-suite-name qwen35_cuda_needle_in_a_haystack_v1 \
  --evaluation-prompt-count 1 \
  --evaluation-batch-size 1 \
  --evaluation-notes "Named task-style Needle-in-a-Haystack prompt with exact-match retrieval scoring; single prompt per context/case in this first wiring pass." \
  --profile-backend \
  --output "$OUTPUT_PATH"
