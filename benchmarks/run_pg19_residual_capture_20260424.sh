#!/usr/bin/env bash
# Short residual-telemetry capture for the certificate figure
# (reviewer Item 3, Option C). Five chunks at 16K with the newly
# instrumented kernel so each step records the observed per-block
# score residual |FP16 - INT8| alongside the analytical delta bound.
# ~1.5h wall. Run after the 128K single-chunk completes (single GPU).
set +e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
OUT_DIR="$ROOT_DIR/benchmarks/results/pg19_residual_capture_20260424"
mkdir -p "$OUT_DIR"

PY="$ROOT_DIR/.venv/bin/python"

banner() { echo; echo "=== $(date -u +%FT%TZ) :: $* ==="; }

exec > >(tee -a "$OUT_DIR/orchestrator.log") 2>&1

banner "PG-19 residual capture start (16K, 5 chunks, ~1.5h expected)"
timeout 10800 "$PY" benchmarks/paper/pg19_perplexity.py \
  --context 16384 --num-chunks 5 \
  --tau-cov 0.995 --k-min 2 --k-max 128 \
  --ranking-fallback --ranking-r 1 \
  --epsilon-override 1e-4 \
  --output "$OUT_DIR/pg19_ctx16384_residual.json" \
  > "$OUT_DIR/pg19_ctx16384_residual.log" 2>&1
ec=$?
banner "residual capture exit=$ec"
tail -20 "$OUT_DIR/pg19_ctx16384_residual.log" \
  | grep -v "Loading weights\|MatMul8bitLt" || true

banner "PG-19 residual capture done (exit=$ec)"
echo "Result: $OUT_DIR/pg19_ctx16384_residual.json"
