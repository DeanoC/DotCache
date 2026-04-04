#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/env_cuda.sh"

MODE="${1:-serving}"
if [[ $# -gt 0 ]]; then
  shift
fi

PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

case "$MODE" in
  readout)
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
      "$@"
    ;;
  serving)
    exec "$PYTHON_BIN" benchmarks/bench_qwen35_deltanet_statecache_serving.py \
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
      "$@"
    ;;
  serving-long-horizon)
    exec "$PYTHON_BIN" benchmarks/bench_qwen35_deltanet_statecache_serving.py \
      --model-id Qwen/Qwen3.5-0.8B \
      --backend torch_cuda \
      --device cuda \
      --torch-dtype float16 \
      --weight-quantization none \
      --bits 8 \
      --state-stage post_update_m0 \
      --renorm-interval 0 \
      --recurrent-group-size-policy 890m_long_horizon_group_escape_v1 \
      --max-new-tokens 16 \
      --warmup-in-process-repeats 1 \
      --in-process-repeats 3 \
      --repeat-counts \
      --target-prompt-lengths 6144 \
      --continue-on-error \
      "$@"
    ;;
  *)
    echo "usage: $0 [readout|serving|serving-long-horizon] [extra benchmark args...]" >&2
    exit 1
    ;;
esac
