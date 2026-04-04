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
      --model-id Qwen/Qwen3.5-4B \
      --backend torch_cuda \
      --device cuda \
      --torch-dtype float16 \
      --weight-quantization bnb_8bit \
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
      "$@"
    ;;
  serving)
    exec "$PYTHON_BIN" benchmarks/bench_qwen35_deltanet_statecache_serving.py \
      --model-id Qwen/Qwen3.5-4B \
      --backend torch_cuda \
      --device cuda \
      --torch-dtype float16 \
      --weight-quantization bnb_8bit \
      --bits 8 \
      --state-stage post_update_m0 \
      --renorm-interval 0 \
      --recurrent-mode-override layer:0=M3 \
      --recurrent-mode-override layer:1=M3 \
      --recurrent-mode-override layer:2=M3 \
      --max-new-tokens 4 \
      --repeat-counts \
      --target-prompt-lengths 2048 4096 \
      --continue-on-error \
      "$@"
    ;;
  *)
    echo "usage: $0 [readout|serving] [extra benchmark args...]" >&2
    exit 1
    ;;
esac
