#!/usr/bin/env bash
# 64K throughput sweep on pg19 prefill, the regime where the tiered
# architecture actually pays off (K_max=128 is 3.1% of 4096 blocks, so
# per-layer unions can realistically stay well under corpus size).
#
# Configs:
#   dense                — baseline (no certified, no cache)
#   certified cap=256    — small cache (6.3% corpus)
#   certified cap=512    — (12.5% corpus)
#   certified cap=1024   — (25% corpus)
#   certified cap=4096   — (= corpus)
#   certified cap=∞      — full mirror ceiling
#
# 3 repeats per config (lowered from 5 to keep wall time ~3h instead of 5h).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/env_cuda.sh"
source "$ROOT_DIR/.venv/bin/activate"

OUT_DIR="$ROOT_DIR/benchmarks/results/perf_tests_20260422/cache_sweep_64k"
mkdir -p "$OUT_DIR"
BRANCH="feature/interval-ellipsoidal-bounds"

REPEATS=3
CONTEXT=65536

run_one() {
  local tag="$1"
  local env_spec="$2"
  local configs="$3"
  local out="$OUT_DIR/64k_${tag}.json"
  if [[ -f "$out" ]]; then
    echo "skip $tag"
    return 0
  fi
  echo "=== 64K: $tag ==="
  eval $env_spec DOTCACHE_V_TOL=0.05 python "$ROOT_DIR/benchmarks/bench_throughput_8k.py" \
    --context-length "$CONTEXT" --decode-tokens 256 --warmup-tokens 16 --repeats "$REPEATS" \
    --configs $configs --prompt-source pg19 --output "$out"
  git add "$out"
  git commit -m "64k throughput: $tag" || true
  git push origin "$BRANCH" || true
}

run_one "dense"         ""                                 "dense"
run_one "cert_cap256"   "DOTCACHE_FP16_CACHE_BLOCKS=256"   "certified"
run_one "cert_cap512"   "DOTCACHE_FP16_CACHE_BLOCKS=512"   "certified"
run_one "cert_cap1024"  "DOTCACHE_FP16_CACHE_BLOCKS=1024"  "certified"
run_one "cert_cap4096"  "DOTCACHE_FP16_CACHE_BLOCKS=4096"  "certified"
run_one "cert_full"     ""                                 "certified"

echo "=== 64K throughput sweep complete ==="
