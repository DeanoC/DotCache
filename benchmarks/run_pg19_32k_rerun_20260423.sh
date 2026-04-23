#!/usr/bin/env bash
# PG-19 32K rerun with 20-chunk per-chunk CI pipeline.
#
# The arxiv_v1 sweep's 32K certified cell took ~6.5h wall (cell 22),
# but that used the older per-cell cadence without per_chunk_bpt_stats.
# Budget 12h here to leave comfortable headroom for the newer CI path.
#
# Launched separately from run_pg19_rerun_20260423.sh because 16K was
# already in-flight when we decided to add 32K.
set +e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
OUT_DIR="$ROOT_DIR/benchmarks/results/pg19_rerun_20260423"
mkdir -p "$OUT_DIR"

PY="$ROOT_DIR/.venv/bin/python"

banner() { echo; echo "=== $(date -u +%FT%TZ) :: $* ==="; }

exec > >(tee -a "$OUT_DIR/orchestrator_32k.log") 2>&1

banner "PG-19 ctx=32768 rerun start (20 chunks, 12h budget)"

# At 32K the certified cell in arxiv_v1 took ~6.5h, so 12h is generous
# even with the slightly heavier per-chunk CI pipeline.
timeout 43200 "$PY" benchmarks/paper/pg19_perplexity.py \
  --context 32768 --num-chunks 20 \
  --tau-cov 0.995 --k-min 2 --k-max 128 \
  --ranking-fallback --ranking-r 1 \
  --epsilon-override 1e-4 \
  --output "$OUT_DIR/pg19_ctx32768.json" \
  > "$OUT_DIR/pg19_ctx32768.log" 2>&1
ec=$?
banner "PG-19 ctx=32768 exit=$ec"
tail -30 "$OUT_DIR/pg19_ctx32768.log" | grep -v "Loading weights\|MatMul8bitLt" || true

banner "PG-19 32K rerun done (exit=$ec)"
echo "Results in $OUT_DIR"
