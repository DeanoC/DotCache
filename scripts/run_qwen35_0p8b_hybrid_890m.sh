#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/env_cuda.sh"

exec "$ROOT_DIR/.venv/bin/python" benchmarks/bench_qwen35_attention_subset_statecache_dotcache.py \
  --model-id Qwen/Qwen3.5-0.8B \
  --backend torch_cuda \
  --device cuda \
  --torch-dtype float16 \
  --layer-profile configs/layer_profiles/qwen35_0p8b_attention_subset_cuda_shortlist_baseline.yaml \
  --state-stage post_update_m0 \
  --state-bits 8 \
  --state-renorm-interval 0 \
  --max-new-tokens 4 \
  --repeat-counts \
  --target-prompt-lengths 512 2048 8192 \
  --continue-on-error \
  "$@"
