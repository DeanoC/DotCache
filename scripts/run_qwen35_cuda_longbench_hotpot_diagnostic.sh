#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_PATH="${1:-$ROOT_DIR/benchmarks/results/qwen35_cuda_longbench_hotpot_diagnostic_v1.jsonl}"
SUMMARY_PATH="${2:-$ROOT_DIR/benchmarks/results/qwen35_cuda_longbench_hotpot_diagnostic_v1_summary.md}"
PROMPT_PACK="${3:-$ROOT_DIR/configs/prompt_packs/qwen35_cuda_longbench_hotpot_case_order_v1.json}"

export PYTHONPATH="$ROOT_DIR"
export PATH="$ROOT_DIR/.venv/bin:$PATH"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

EXACT_PATH="$TMP_DIR/exact.jsonl"
DIAGNOSTIC_PATH="$TMP_DIR/diagnostic.jsonl"

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/run_qwen35_cuda_longbench_qa_probe.py" \
  --layer-profile "$ROOT_DIR/configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_third_pass.yaml" \
  --cases exact \
  --prompt-pack "$PROMPT_PACK" \
  --timeout-seconds 1800 \
  --evaluation-split held_out \
  --evaluation-lane diagnostic \
  --evaluation-prompt-family longbench_qa \
  --evaluation-prompt-suite-name qwen35_cuda_longbench_hotpot_diagnostic_v1 \
  --evaluation-prompt-count 1 \
  --evaluation-batch-size 1 \
  --evaluation-notes "Hotpot-focused LongBench reference row for shortlist retrieval diagnostics. Exact row is included for output and F1 comparison; shortlist rows below add scorer-diagnostic traces." \
  --profile-backend \
  --output "$EXACT_PATH"

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/run_qwen35_cuda_longbench_qa_probe.py" \
  --layer-profile "$ROOT_DIR/configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_third_pass.yaml" \
  --quality-layer-profile "$ROOT_DIR/configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_shortlist_quality.yaml" \
  --cases shortlist_base shortlist_l23_ctx shortlist_topk8 shortlist_quality_profile \
  --prompt-pack "$PROMPT_PACK" \
  --timeout-seconds 1800 \
  --evaluation-split held_out \
  --evaluation-lane diagnostic \
  --evaluation-prompt-family longbench_qa \
  --evaluation-prompt-suite-name qwen35_cuda_longbench_hotpot_diagnostic_v1 \
  --evaluation-prompt-count 1 \
  --evaluation-batch-size 1 \
  --evaluation-notes "Hotpot-focused LongBench shortlist scorer diagnostic for retrieval/selection analysis. Compare shortlist variants against exact-top page recall and repeated missed exact page ranges." \
  --profile-backend \
  --scorer-diagnostic \
  --output "$DIAGNOSTIC_PATH"

cat "$EXACT_PATH" "$DIAGNOSTIC_PATH" >"$OUTPUT_PATH"

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/summarize_qwen35_cuda_longbench_hotpot_diagnostic.py" \
  "$OUTPUT_PATH" \
  --markdown-output "$SUMMARY_PATH" \
  >/dev/null
