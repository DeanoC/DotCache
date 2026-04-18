#!/bin/bash
# Paper-grade LongBench (EN subset): 50 samples × 13 subtasks × 2 regimes × 2 contexts.
# Projected ~9 h 15 min wall on RTX 5090 (see session_20260417d).
# Defer 16K to the larger machine per benchmark_plan.md.
set -e
cd /DotCache
PROFILE=configs/NousResearch_Meta-Llama-3.1-8B_calibrated.npz
OUTDIR=benchmarks/results/longbench_paper_20260418
N=50

run() {
  local tag=$1; shift
  echo "=== [$tag] start $(date -Iseconds) ==="
  python -u benchmarks/paper/longbench.py "$@" \
    --num-samples $N \
    --output "$OUTDIR/${tag}.json" \
    2>&1 | tee "$OUTDIR/${tag}.log"
  echo "=== [$tag] done  $(date -Iseconds) ==="
}

run eps0_4k        --contexts 4096 --default-epsilon 0
run calibrated_4k  --contexts 4096 --profile "$PROFILE"
run eps0_8k        --contexts 8192 --default-epsilon 0
run calibrated_8k  --contexts 8192 --profile "$PROFILE"

echo "=== ALL PAPER CONFIGS DONE ==="
