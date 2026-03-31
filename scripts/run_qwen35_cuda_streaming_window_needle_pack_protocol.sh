#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_PATH="${1:-$ROOT_DIR/benchmarks/results/qwen35_cuda_streaming_window_needle_pack_v1.jsonl}"
SUMMARY_PATH="${2:-$ROOT_DIR/benchmarks/results/qwen35_cuda_streaming_window_needle_pack_v1_summary.md}"
PROMPT_PACK="${3:-$ROOT_DIR/configs/prompt_packs/qwen35_cuda_needle_pack_v1.json}"

export PYTHONPATH="$ROOT_DIR"
export PATH="$ROOT_DIR/.venv/bin:$PATH"

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/run_qwen35_cuda_needle_probe.py" \
  --layer-profile "$ROOT_DIR/configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_third_pass.yaml" \
  --contexts 32768 49152 \
  --cases exact streaming_sink_recent shortlist_base shortlist_l23_ctx \
  --timeout-seconds 1200 \
  --max-new-tokens 12 \
  --prompt-pack "$PROMPT_PACK" \
  --evaluation-split held_out \
  --evaluation-lane systems \
  --evaluation-prompt-family needle_in_a_haystack \
  --evaluation-prompt-suite-name qwen35_cuda_streaming_window_needle_pack_v1 \
  --evaluation-prompt-count 4 \
  --evaluation-batch-size 1 \
  --evaluation-notes "Fixed four-prompt Needle pack with StreamingLLM-style sink-plus-recent baseline (256 sink, 1024 recent) alongside exact and DotCache shortlist cases." \
  --profile-backend \
  --output "$OUTPUT_PATH"

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/summarize_qwen35_cuda_needle_pack.py" \
  "$OUTPUT_PATH" \
  --markdown-output "$SUMMARY_PATH" \
  >/dev/null
