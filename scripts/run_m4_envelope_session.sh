#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python benchmarks/bench_decode_session.py \
  --backend torch_mps \
  --config configs/dotcache_m4_mps.yaml \
  --execution-sink-window 256 \
  --execution-recent-window 1024 \
  --execution-relevance-top-k 4 \
  --execution-relevance-mode envelope \
  "$@"
