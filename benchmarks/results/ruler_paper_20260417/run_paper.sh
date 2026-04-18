#!/bin/bash
# Paper-grade RULER: 50 samples × 7 subtasks × 2 modes × 2 contexts.
# Estimated ~5 h wall on RTX 5090.
set -e
cd /DotCache
PROFILE=configs/NousResearch_Meta-Llama-3.1-8B_calibrated.npz
OUTDIR=benchmarks/results/ruler_paper_20260417
N=50

run() {
  local tag=$1; shift
  echo "=== [$tag] start $(date -Iseconds) ==="
  python benchmarks/paper/ruler.py "$@" \
    --output "$OUTDIR/${tag}.json" \
    2>&1 | tee "$OUTDIR/${tag}.log"
  echo "=== [$tag] done  $(date -Iseconds) ==="
}

run eps0_4k        --contexts 4096 --num-samples $N --default-epsilon 0
run calibrated_4k  --contexts 4096 --num-samples $N --profile "$PROFILE"
run eps0_8k        --contexts 8192 --num-samples $N --default-epsilon 0
run calibrated_8k  --contexts 8192 --num-samples $N --profile "$PROFILE"

echo "=== ALL PAPER CONFIGS DONE ==="
