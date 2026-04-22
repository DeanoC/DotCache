#!/bin/bash
# Runs remaining 3 post-fix RULER smokes sequentially.
set -e
cd /DotCache
PROFILE=configs/NousResearch_Meta-Llama-3.1-8B_calibrated.npz
OUTDIR=benchmarks/results/ruler_postfix_20260417

echo "=== [1/3] calibrated 4K ==="
python benchmarks/paper/ruler.py \
  --contexts 4096 --num-samples 10 \
  --profile "$PROFILE" \
  --output "$OUTDIR/calibrated_4k.json" \
  2>&1 | tee "$OUTDIR/calibrated_4k.log"

echo "=== [2/3] eps=0 8K ==="
python benchmarks/paper/ruler.py \
  --contexts 8192 --num-samples 10 \
  --default-epsilon 0 \
  --output "$OUTDIR/eps0_8k.json" \
  2>&1 | tee "$OUTDIR/eps0_8k.log"

echo "=== [3/3] calibrated 8K ==="
python benchmarks/paper/ruler.py \
  --contexts 8192 --num-samples 10 \
  --profile "$PROFILE" \
  --output "$OUTDIR/calibrated_8k.json" \
  2>&1 | tee "$OUTDIR/calibrated_8k.log"

echo "=== ALL DONE ==="
