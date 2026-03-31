#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_PATH="${1:-$ROOT_DIR/benchmarks/results/qwen35_cuda_shortlist_large_context_quality_tail_forced_grouped.jsonl}"

export PYTHONPATH="$ROOT_DIR"
export PATH="$ROOT_DIR/.venv/bin:$PATH"
export DOTCACHE_QWEN35_FORCE_GROUPED_BATCHING=1

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/run_qwen35_cuda_shortlist_probe.py" \
  --layer-profile "$ROOT_DIR/configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_third_pass.yaml" \
  --contexts 32768 49152 \
  --cases exact shortlist_base shortlist_l23_ctx \
  --timeout-seconds 900 \
  --quality-check \
  --quality-mode loss_tail \
  --quality-eval-steps 4 \
  --evaluation-split held_out \
  --evaluation-lane quality \
  --evaluation-prompt-family synthetic_exact_length_filler \
  --evaluation-prompt-suite-name qwen35_cuda_shortlist_large_context_forced_grouped_quality_tail_synthetic \
  --evaluation-prompt-count 1 \
  --evaluation-batch-size 1 \
  --evaluation-notes "Synthetic exact-length filler only; useful for held-out lane discipline, not publication-grade final quality claims." \
  --profile-backend \
  --output "$OUTPUT_PATH"
