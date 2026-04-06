#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_PATH="${1:-$ROOT_DIR/benchmarks/results/qwen35_cuda_passkey_pack_protocol_v1.jsonl}"
SUMMARY_PATH="${2:-$ROOT_DIR/benchmarks/results/qwen35_cuda_passkey_pack_protocol_v1_summary.md}"
PROMPT_PACK="${3:-$ROOT_DIR/configs/prompt_packs/qwen35_cuda_passkey_pack_v1.json}"

export PYTHONPATH="$ROOT_DIR"
export PATH="$ROOT_DIR/.venv/bin:$PATH"

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/run_qwen35_cuda_passkey_probe.py" \
  --layer-profile "$ROOT_DIR/configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_third_pass.yaml" \
  --contexts 32768 49152 \
  --cases exact shortlist_base shortlist_l23_ctx \
  --timeout-seconds 1200 \
  --max-new-tokens 12 \
  --prompt-pack "$PROMPT_PACK" \
  --evaluation-split held_out \
  --evaluation-lane systems \
  --evaluation-prompt-family passkey_retrieval \
  --evaluation-prompt-suite-name qwen35_cuda_passkey_pack_v1 \
  --evaluation-prompt-count 4 \
  --evaluation-batch-size 1 \
  --evaluation-notes "Fixed four-prompt RULER-style passkey retrieval pack; exact plus DotCache shortlist cases with retrieval scoring and serving metrics." \
  --profile-backend \
  --output "$OUTPUT_PATH"

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/summarize_qwen35_cuda_passkey_pack.py" \
  "$OUTPUT_PATH" \
  --markdown-output "$SUMMARY_PATH" \
  >/dev/null
