#!/usr/bin/env bash
# Short multi-context instrumented pass for the E_key / tail_mass
# histogram figure (reviewer Item 3). Five chunks per context at 8K and
# 16K. The 32K run supplies the wide-context data point; this script
# covers the smaller contexts so the histogram spans the full range.
#
# Expected wall: ~1h (8K: 5×7 min ≈ 35 min; 16K: 5×~18 min ≈ 90 min total).
set +e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
OUT_DIR="$ROOT_DIR/benchmarks/results/pg19_ekey_short_20260423"
mkdir -p "$OUT_DIR"

PY="$ROOT_DIR/.venv/bin/python"

banner() { echo; echo "=== $(date -u +%FT%TZ) :: $* ==="; }

exec > >(tee -a "$OUT_DIR/orchestrator.log") 2>&1

banner "E_key short pass start"

for ctx in 8192 16384; do
  out_json="$OUT_DIR/pg19_ctx${ctx}_ekey.json"
  out_log="$OUT_DIR/pg19_ctx${ctx}_ekey.log"
  banner "ctx=$ctx (5 chunks, instrumented)"
  # Generous timeout per context — 3h at 8K, 5h at 16K in principle.
  case "$ctx" in
    8192)  budget=10800 ;;
    16384) budget=18000 ;;
    *)     budget=10800 ;;
  esac
  timeout "$budget" "$PY" benchmarks/paper/pg19_perplexity.py \
    --context "$ctx" --num-chunks 5 \
    --tau-cov 0.995 --k-min 2 --k-max 128 \
    --ranking-fallback --ranking-r 1 \
    --epsilon-override 1e-4 \
    --output "$out_json" \
    > "$out_log" 2>&1
  ec=$?
  banner "ctx=$ctx exit=$ec"
  tail -15 "$out_log" | grep -v "Loading weights\|MatMul8bitLt" || true
done

banner "E_key short pass done"
echo "Results in $OUT_DIR"
