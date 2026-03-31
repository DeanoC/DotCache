#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/env_cuda.sh"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
RESULTS_DIR="${DOTCACHE_RESULTS_DIR:-benchmarks/results/qwen35_rocm_890m_statecache_discovery_20260331}"

mkdir -p "$RESULTS_DIR"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <case> [extra benchmark args...]"
  exit 1
fi

CASE_NAME="$1"
shift

case "$CASE_NAME" in
  0p8b-readout-baseline)
    OUTPUT_PATH="${RESULTS_DIR}/qwen35_0p8b_readout_baseline_readoutonlym0.jsonl"
    exec "$PYTHON_BIN" benchmarks/bench_qwen35_deltanet_statecache_readout.py \
      --model-id Qwen/Qwen3.5-0.8B \
      --backend torch_cuda \
      --device cuda \
      --torch-dtype float16 \
      --weight-quantization none \
      --bits 8 \
      --state-stage readout_only_m0 \
      --renorm-interval 0 \
      --max-new-tokens 4 \
      --repeat-counts \
      --target-prompt-lengths 512 2048 8192 \
      --continue-on-error \
      "$@" | tee "$OUTPUT_PATH"
    ;;
  0p8b-readout-postupdate)
    OUTPUT_PATH="${RESULTS_DIR}/qwen35_0p8b_readout_postupdatem0.jsonl"
    exec "$PYTHON_BIN" benchmarks/bench_qwen35_deltanet_statecache_readout.py \
      --model-id Qwen/Qwen3.5-0.8B \
      --backend torch_cuda \
      --device cuda \
      --torch-dtype float16 \
      --weight-quantization none \
      --bits 8 \
      --state-stage post_update_m0 \
      --renorm-interval 0 \
      --max-new-tokens 4 \
      --repeat-counts \
      --target-prompt-lengths 512 2048 8192 \
      --continue-on-error \
      "$@" | tee "$OUTPUT_PATH"
    ;;
  0p8b-real-sweep)
    OUTPUT_PATH="${RESULTS_DIR}/qwen35_0p8b_real_sweep_prompt32_layers0_1_2_12_22.jsonl"
    exec "$PYTHON_BIN" benchmarks/bench_qwen35_statecache_real_sweep.py \
      --model-id Qwen/Qwen3.5-0.8B \
      --backend torch_cuda \
      --device cuda \
      --torch-dtype float16 \
      --prompt-length 32 \
      --max-new-tokens 4 \
      --layers 0 1 2 12 22 \
      --bits 8 4 3 \
      --renorm-intervals 0 2 4 8 \
      --state-kinds recurrent conv \
      --output-dir "${RESULTS_DIR}/real_sweep_outputs" \
      "$@" | tee "$OUTPUT_PATH"
    ;;
  4b-readout-fp16-m3early)
    OUTPUT_PATH="${RESULTS_DIR}/qwen35_4b_readout_fp16_postupdatem0_m3early.jsonl"
    exec "$PYTHON_BIN" benchmarks/bench_qwen35_deltanet_statecache_readout.py \
      --model-id Qwen/Qwen3.5-4B \
      --backend torch_cuda \
      --device cuda \
      --torch-dtype float16 \
      --weight-quantization none \
      --bits 8 \
      --state-stage post_update_m0 \
      --renorm-interval 0 \
      --recurrent-mode-override layer:0=M3 \
      --recurrent-mode-override layer:1=M3 \
      --recurrent-mode-override layer:2=M3 \
      --max-new-tokens 4 \
      --repeat-counts \
      --target-prompt-lengths 512 1024 \
      --continue-on-error \
      "$@" | tee "$OUTPUT_PATH"
    ;;
  *)
    echo "unknown case: $CASE_NAME" >&2
    exit 1
    ;;
esac
