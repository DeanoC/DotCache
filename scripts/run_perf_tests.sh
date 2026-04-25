#!/usr/bin/env bash
# Orchestrate Tests 1/2/3 sequentially (GPU contention-free). Each test
# auto-commits + pushes its output JSON so a pod reset loses at most one.
# Runs the paper-matching bounded FP16 cache configuration: the VRAM
# scratch holds at most DOTCACHE_FP16_CACHE_BLOCKS blocks at a time, with
# LRU eviction and H2D on miss.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/env_cuda.sh"
source "$ROOT_DIR/.venv/bin/activate"

# Paper's "smallish transparent VRAM cache" size, in blocks (16 tokens each).
# 8K context = 512 total blocks; 64 ≈ 12.5% (fits top-K*=128 loosely when
# attention is concentrated across heads, forces misses otherwise).
export DOTCACHE_FP16_CACHE_BLOCKS="${DOTCACHE_FP16_CACHE_BLOCKS:-64}"

OUT_DIR="$ROOT_DIR/benchmarks/results/perf_tests_20260422"
mkdir -p "$OUT_DIR"

BRANCH="feature/interval-ellipsoidal-bounds"

# --- Test 3: 8K certified quality + pagein telemetry ---
# Scaled-down sample counts: enough for telemetry convergence (~1-2k decode
# steps per benchmark) without running the full arXiv v1 sweep again.
if [[ ! -f "$OUT_DIR/test3_niah_8k_paper.pagein.json" ]]; then
  echo "=== Test 3: NIAH 8K certified + telemetry ==="
  DOTCACHE_V_TOL=0.05 python "$ROOT_DIR/benchmarks/paper/niah.py" \
    --contexts 8192 --needles 3 \
    --output "$OUT_DIR/test3_niah_8k_paper.json" \
    --tau-cov 0.995 --k-min 2 --k-max 128 \
    --ranking-fallback --ranking-r 1 \
    --eps-guard 0.01 \
    --exploration-rate 0.02 --rung1-threshold 0.02 --rung1-multiplier 2.0 \
    --pagein-telemetry
  git add "$OUT_DIR/test3_niah_8k_paper.json" "$OUT_DIR/test3_niah_8k_paper.pagein.json" 2>/dev/null || true
  git commit -m "perf Test 3: NIAH 8K certified + pagein telemetry" || true
  git push origin "$BRANCH" || true
fi

if [[ ! -f "$OUT_DIR/test3_pg19_8k_paper.pagein.json" ]]; then
  echo "=== Test 3: PG-19 8K certified + telemetry ==="
  DOTCACHE_V_TOL=0.05 python "$ROOT_DIR/benchmarks/paper/pg19_perplexity.py" \
    --context 8192 --num-chunks 1 --eval-start 0.8 \
    --output "$OUT_DIR/test3_pg19_8k_paper.json" \
    --tau-cov 0.995 --k-min 2 --k-max 128 \
    --ranking-fallback --ranking-r 1 \
    --eps-guard 0.01 \
    --exploration-rate 0.02 --rung1-threshold 0.02 --rung1-multiplier 2.0 \
    --pagein-telemetry
  git add "$OUT_DIR/test3_pg19_8k_paper.json" "$OUT_DIR/test3_pg19_8k_paper.pagein.json" 2>/dev/null || true
  git commit -m "perf Test 3: PG-19 8K certified + pagein telemetry" || true
  git push origin "$BRANCH" || true
fi

if [[ ! -f "$OUT_DIR/test3_ruler_8k_paper.pagein.json" ]]; then
  echo "=== Test 3: RULER 8K certified + telemetry ==="
  DOTCACHE_V_TOL=0.05 python "$ROOT_DIR/benchmarks/paper/ruler.py" \
    --contexts 8192 --num-samples 10 \
    --output "$OUT_DIR/test3_ruler_8k_paper.json" \
    --tau-cov 0.995 --k-min 2 --k-max 128 \
    --ranking-fallback --ranking-r 1 \
    --eps-guard 0.01 \
    --exploration-rate 0.02 --rung1-threshold 0.02 --rung1-multiplier 2.0 \
    --pagein-telemetry
  git add "$OUT_DIR/test3_ruler_8k_paper.json" "$OUT_DIR/test3_ruler_8k_paper.pagein.json" 2>/dev/null || true
  git commit -m "perf Test 3: RULER 8K certified + pagein telemetry" || true
  git push origin "$BRANCH" || true
fi

# --- Test 1: decode throughput matrix ---
if [[ ! -f "$OUT_DIR/test1_throughput_paper.json" ]]; then
  echo "=== Test 1: throughput matrix (4 configs × 10 repeats) ==="
  python "$ROOT_DIR/benchmarks/bench_throughput_8k.py" \
    --context-length 8192 --decode-tokens 256 --warmup-tokens 16 --repeats 10 \
    --configs dense certified certified-no-fallback quantised-only \
    --output "$OUT_DIR/test1_throughput_paper.json"
  git add "$OUT_DIR/test1_throughput_paper.json"
  git commit -m "perf Test 1: throughput matrix (dense/cert/cert-no-fb/quant-only × 10)"
  git push origin "$BRANCH"
fi

# --- Test 2: 7-phase latency breakdown ---
if [[ ! -f "$OUT_DIR/test2_phase_breakdown_paper.json" ]]; then
  echo "=== Test 2: phase breakdown (500 certified decode steps) ==="
  python "$ROOT_DIR/benchmarks/bench_latency_breakdown_8k.py" \
    --context-length 8192 --decode-steps 500 --warmup-steps 16 \
    --output "$OUT_DIR/test2_phase_breakdown_paper.json"
  git add "$OUT_DIR/test2_phase_breakdown_paper.json"
  git commit -m "perf Test 2: 7-phase latency breakdown (500 cert decode steps)"
  git push origin "$BRANCH"
fi

echo "=== all three perf tests complete ==="
