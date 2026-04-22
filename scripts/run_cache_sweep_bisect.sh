#!/usr/bin/env bash
# Binary-search bisection between cap=256 and cap=1024 to locate the knee.
# The 8K corpus is 512 blocks; the hypothesis is that the knee sits right
# at the corpus size (any cap >= 512 fits everything after warmup).
#
# Points (in search order — stops early if a sequential file already exists):
#   384  — below corpus (expected like 256)
#   512  — exactly at corpus (expected to jump to ~99% hit)
#   640  — above corpus (expected like 1024 capacity-wise)
#   768  — well above corpus (confirmation that it plateaus)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/env_cuda.sh"
source "$ROOT_DIR/.venv/bin/activate"

OUT_DIR="$ROOT_DIR/benchmarks/results/perf_tests_20260422/cache_sweep"
mkdir -p "$OUT_DIR"
BRANCH="feature/interval-ellipsoidal-bounds"
REPEATS=5

run_one() {
  local cap="$1"
  local tag="cap_${cap}"
  local out="$OUT_DIR/cache_sweep_${tag}.json"
  if [[ -f "$out" ]]; then
    echo "skip: $out already exists"
    return 0
  fi
  echo "=== cache sweep: $tag ==="
  DOTCACHE_FP16_CACHE_BLOCKS="$cap" DOTCACHE_V_TOL=0.05 python \
    "$ROOT_DIR/benchmarks/bench_throughput_8k.py" \
    --context-length 8192 --decode-tokens 256 --warmup-tokens 16 \
    --repeats "$REPEATS" --configs certified --output "$out"
  git add "$out" 2>/dev/null || true
  git commit -m "cache sweep: $tag" || true
  git push origin "$BRANCH" || true
}

run_one 384
run_one 512
run_one 640
run_one 768

echo "=== bisection complete ==="
