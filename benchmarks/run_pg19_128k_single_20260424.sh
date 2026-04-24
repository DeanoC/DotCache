#!/usr/bin/env bash
# PG-19 128K — single-chunk run with effectively no timeout.
#
# The earlier smoke confirmed dense prefill of 131K tokens is fine
# (22s) on this 96GB GPU; the 3h smoke budget timed out during the
# cert teacher-forced loop, which at ~180 ms/step × 65535 steps
# ≈ 3.3h per chunk. 24h budget here is effectively "no timeout" —
# real run is expected ~4h end-to-end.
#
# One chunk gives the paper a single validated 128K data point
# (no CI, but completes the figure / lets the abstract claim
# "validated through 128K including the regime where VRAM savings
# emerge").
set +e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
OUT_DIR="$ROOT_DIR/benchmarks/results/pg19_rerun_20260423"
mkdir -p "$OUT_DIR"

PY="$ROOT_DIR/.venv/bin/python"

banner() { echo; echo "=== $(date -u +%FT%TZ) :: $* ==="; }

exec > >(tee -a "$OUT_DIR/orchestrator_128k_single.log") 2>&1

banner "PG-19 ctx=131072 single-chunk start (24h ceiling, ~4h expected)"
timeout 86400 "$PY" benchmarks/paper/pg19_perplexity.py \
  --context 131072 --num-chunks 1 \
  --tau-cov 0.995 --k-min 2 --k-max 128 \
  --ranking-fallback --ranking-r 1 \
  --epsilon-override 1e-4 \
  --output "$OUT_DIR/pg19_ctx131072.json" \
  > "$OUT_DIR/pg19_ctx131072.log" 2>&1
ec=$?
banner "PG-19 ctx=131072 single-chunk exit=$ec"
tail -40 "$OUT_DIR/pg19_ctx131072.log" \
  | grep -v "Loading weights\|MatMul8bitLt" || true

banner "PG-19 128K single-chunk done (exit=$ec)"
echo "Result: $OUT_DIR/pg19_ctx131072.json"
