#!/usr/bin/env bash
# FP16 VRAM cache capacity sweep on the certified path @ 8K context.
# Runs the same throughput bench as Test 1 but only for the certified config,
# varying DOTCACHE_FP16_CACHE_BLOCKS across the sweep points. Each run
# auto-commits its JSON.
#
# Sweep points (5 total):
#   0      — pure H2D floor: every top-K block is paged in every step
#   64     — current paper operating point (~12.5% of the 512-block corpus)
#   256    — 50% of the corpus
#   1024   — above total blocks → effectively full-mirror ceiling
#   unset  — legacy full-mirror mode (same ceiling confirmed a different way)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/env_cuda.sh"
source "$ROOT_DIR/.venv/bin/activate"

OUT_DIR="$ROOT_DIR/benchmarks/results/perf_tests_20260422/cache_sweep"
mkdir -p "$OUT_DIR"
BRANCH="feature/interval-ellipsoidal-bounds"

REPEATS=5

run_one() {
  local tag="$1"
  local env_spec="$2"
  local out="$OUT_DIR/cache_sweep_${tag}.json"
  if [[ -f "$out" ]]; then
    echo "skip: $out already exists"
    return 0
  fi
  echo "=== cache sweep: $tag ==="
  eval $env_spec DOTCACHE_V_TOL=0.05 python "$ROOT_DIR/benchmarks/bench_throughput_8k.py" \
    --context-length 8192 --decode-tokens 256 --warmup-tokens 16 --repeats "$REPEATS" \
    --configs certified \
    --output "$out"
  git add "$out" 2>/dev/null || true
  git commit -m "cache sweep: $tag" || true
  git push origin "$BRANCH" || true
}

run_one "cap_0"       "DOTCACHE_FP16_CACHE_BLOCKS=0"
run_one "cap_64"      "DOTCACHE_FP16_CACHE_BLOCKS=64"
run_one "cap_256"     "DOTCACHE_FP16_CACHE_BLOCKS=256"
run_one "cap_1024"    "DOTCACHE_FP16_CACHE_BLOCKS=1024"
run_one "full_mirror" ""

echo "=== cache sweep complete ==="
