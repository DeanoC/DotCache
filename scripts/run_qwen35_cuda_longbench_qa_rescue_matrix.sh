#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_PATH="${1:-$ROOT_DIR/benchmarks/results/qwen35_cuda_longbench_qa_rescue_matrix_v1.jsonl}"
SUMMARY_PATH="${2:-$ROOT_DIR/benchmarks/results/qwen35_cuda_longbench_qa_rescue_matrix_v1_summary.md}"
PROMPT_PACK="${3:-$ROOT_DIR/configs/prompt_packs/qwen35_cuda_longbench_qa_pack_v1.json}"

export PYTHONPATH="$ROOT_DIR"
export PATH="$ROOT_DIR/.venv/bin:$PATH"

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/run_qwen35_cuda_longbench_qa_probe.py" \
  --layer-profile "$ROOT_DIR/configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_third_pass.yaml" \
  --quality-layer-profile "$ROOT_DIR/configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_shortlist_quality.yaml" \
  --cases exact shortlist_base shortlist_l23_ctx shortlist_topk8 shortlist_quality_profile \
  --prompt-pack "$PROMPT_PACK" \
  --timeout-seconds 1800 \
  --evaluation-split held_out \
  --evaluation-lane systems \
  --evaluation-prompt-family longbench_qa \
  --evaluation-prompt-suite-name qwen35_cuda_longbench_qa_rescue_matrix_v1 \
  --evaluation-prompt-count 4 \
  --evaluation-batch-size 1 \
  --evaluation-notes "LongBench QA rescue matrix with baseline shortlist, layer-23 context override, top-k=8 widening, and the existing quality-biased shortlist profile. Rows also record cleaned-answer diagnostic scores for chat artifact spillover." \
  --profile-backend \
  --output "$OUTPUT_PATH"

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/summarize_qwen35_cuda_longbench_qa_pack.py" \
  "$OUTPUT_PATH" \
  --markdown-output "$SUMMARY_PATH" \
  >/dev/null
