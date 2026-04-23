#!/usr/bin/env bash
# Orchestrates the runs that follow the current 16K job:
#   1. PG-19 32K rerun (20 chunks, 12h budget) — instrumented automatically
#      now that the telemetry wiring is in.
#   2. Short E_key pass at 8K + 16K (5 chunks each, ~2h wall) — populates
#      the histogram bins at smaller contexts so Fig-3 spans the range.
#   3. PG-19 128K smoke (1 chunk) → 5-chunk full — the paper's strongest
#      claim data point.
#
# Each stage runs sequentially on the single GPU.
set +e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

banner() { echo; echo "=== $(date -u +%FT%TZ) :: $* ==="; }

LOG="$ROOT_DIR/benchmarks/results/postpg19_chain_20260423.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

banner "post-PG-19 chain start"

banner "Stage 1/3 — PG-19 32K rerun"
bash benchmarks/run_pg19_32k_rerun_20260423.sh
ec32k=$?
banner "Stage 1 exit=$ec32k"

banner "Stage 2/3 — short E_key pass (8K + 16K)"
bash benchmarks/run_pg19_ekey_short_20260423.sh
ec_short=$?
banner "Stage 2 exit=$ec_short"

banner "Stage 3/3 — PG-19 128K smoke + 5-chunk full"
bash benchmarks/run_pg19_128k_20260423.sh
ec_128k=$?
banner "Stage 3 exit=$ec_128k"

banner "post-PG-19 chain done (32k=$ec32k, short=$ec_short, 128k=$ec_128k)"
