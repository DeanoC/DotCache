#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python benchmarks/bench_decode_session.py \
  --backend torch_mps \
  --config configs/dotcache_m4_mps.yaml \
  --execution-profile m4_envelope_fast \
  "$@"
