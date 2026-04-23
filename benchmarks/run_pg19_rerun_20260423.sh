#!/usr/bin/env bash
# PG-19 8K and 16K reruns with longer per-context budgets.
#
# The overnight 2026-04-23 run hit 5400s per context, which only got
# 10/20 chunks done at 8K and 5/20 at 16K. Re-running with the same
# flags but a 4h budget at 8K and an 8h budget at 16K to land 20-chunk
# CIs under PR #98's per_chunk_bpt_stats pipeline.
set +e  # continue past failures

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
OUT_DIR="$ROOT_DIR/benchmarks/results/pg19_rerun_20260423"
mkdir -p "$OUT_DIR"

PY="$ROOT_DIR/.venv/bin/python"

banner() { echo; echo "=== $(date -u +%FT%TZ) :: $* ==="; }

exec > >(tee -a "$OUT_DIR/orchestrator.log") 2>&1

banner "PG-19 rerun start"

# 8K: ~9 min/chunk observed → 20 chunks ≈ 3h, budget 4h.
banner "PG-19 ctx=8192 (20 chunks, 4h budget)"
timeout 14400 "$PY" benchmarks/paper/pg19_perplexity.py \
  --context 8192 --num-chunks 20 \
  --tau-cov 0.995 --k-min 2 --k-max 128 \
  --ranking-fallback --ranking-r 1 \
  --epsilon-override 1e-4 \
  --output "$OUT_DIR/pg19_ctx8192.json" \
  > "$OUT_DIR/pg19_ctx8192.log" 2>&1
ec_8k=$?
banner "PG-19 ctx=8192 exit=$ec_8k"
tail -30 "$OUT_DIR/pg19_ctx8192.log" | grep -v "Loading weights\|MatMul8bitLt" || true

# 16K: ~18 min/chunk observed → 20 chunks ≈ 6h, budget 8h.
banner "PG-19 ctx=16384 (20 chunks, 8h budget)"
timeout 28800 "$PY" benchmarks/paper/pg19_perplexity.py \
  --context 16384 --num-chunks 20 \
  --tau-cov 0.995 --k-min 2 --k-max 128 \
  --ranking-fallback --ranking-r 1 \
  --epsilon-override 1e-4 \
  --output "$OUT_DIR/pg19_ctx16384.json" \
  > "$OUT_DIR/pg19_ctx16384.log" 2>&1
ec_16k=$?
banner "PG-19 ctx=16384 exit=$ec_16k"
tail -30 "$OUT_DIR/pg19_ctx16384.log" | grep -v "Loading weights\|MatMul8bitLt" || true

banner "PG-19 rerun done (8K exit=$ec_8k, 16K exit=$ec_16k)"
echo "Results in $OUT_DIR"
