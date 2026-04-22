#!/usr/bin/env bash
# NIAH 8K τ_cov sweep (100 trials × 3 values) to tighten the confidence
# interval on the -6.7pp NIAH 8K certified-vs-dense gap reported in the
# arXiv v1 SUMMARY. Runs three τ_cov settings sequentially; commits+pushes
# after each so a pod reset loses at most one run.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/env_cuda.sh"
source "$ROOT_DIR/.venv/bin/activate"

OUT_DIR="$ROOT_DIR/benchmarks/results/niah_8k_tau_sweep_20260422"
mkdir -p "$OUT_DIR"

BRANCH="feature/interval-ellipsoidal-bounds"

for TAU in 0.99 0.995 0.999; do
  TAG="${TAU//./}"
  OUT="$OUT_DIR/niah_8k_tau${TAG}_n100.json"
  LOG="$OUT_DIR/niah_8k_tau${TAG}_n100.log"
  if [[ -f "$OUT" ]]; then
    echo "skip: $OUT already exists"
    continue
  fi
  echo "=== NIAH 8K tau_cov=$TAU n=100 -> $OUT ==="
  DOTCACHE_V_TOL=0.05 python "$ROOT_DIR/benchmarks/paper/niah.py" \
    --contexts 8192 \
    --needles 10 \
    --output "$OUT" \
    --tau-cov "$TAU" \
    --k-min 2 \
    --k-max 128 \
    --ranking-fallback \
    --ranking-r 1 \
    --score-consistency-check \
    --eps-guard 0.01 \
    --exploration-rate 0.02 \
    --rung1-threshold 0.02 \
    --rung1-multiplier 2.0 \
    2>&1 | tee "$LOG"
  git add "$OUT" "$LOG"
  git commit -m "NIAH 8K tau_cov=$TAU sweep (100 trials)"
  git push origin "$BRANCH"
done

echo "=== all three runs complete ==="
