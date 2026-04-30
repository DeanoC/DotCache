#!/usr/bin/env bash
# PG-19 128K: smoke-first (1 chunk, ~2h) then 5-chunk full run (~10-13h).
#
# Validates the 128K regime for the paper. Memory envelope on this 96GB
# RTX PRO 6000 should be comfortable in principle (steady-state decode
# ~29GB, tiered construction transient ~37GB), but unchunked prefill of
# the 65K prefix via model(..., use_cache=True) is the unknown — it hasn't
# been exercised on Llama 3.1 8B INT8 at this scale on this hardware.
# arxiv_v1 only measured PG-19 up to 32K; bench_certified_64k chunks
# prefill above 32K precisely because of MLP activation spikes.
#
# Flow: smoke (1 chunk) → if exit=0 → full (5 chunks). Short-circuit
# on smoke failure so the big run doesn't start if prefill OOMs.
set +e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
OUT_DIR="$ROOT_DIR/benchmarks/results/pg19_rerun_20260423"
mkdir -p "$OUT_DIR"

PY="$ROOT_DIR/.venv/bin/python"

banner() { echo; echo "=== $(date -u +%FT%TZ) :: $* ==="; }

exec > >(tee -a "$OUT_DIR/orchestrator_128k.log") 2>&1

# ── Smoke: 1 chunk at 128K ─────────────────────────────────────
banner "PG-19 ctx=131072 SMOKE (1 chunk, 3h budget)"
timeout 10800 "$PY" benchmarks/paper/pg19_perplexity.py \
  --context 131072 --num-chunks 1 \
  --tau-cov 0.995 --k-min 2 --k-max 128 \
  --ranking-fallback --ranking-r 1 \
  --epsilon-override 1e-4 \
  --output "$OUT_DIR/pg19_ctx131072_smoke.json" \
  > "$OUT_DIR/pg19_ctx131072_smoke.log" 2>&1
smoke_ec=$?
banner "PG-19 ctx=131072 SMOKE exit=$smoke_ec"
tail -40 "$OUT_DIR/pg19_ctx131072_smoke.log" \
  | grep -v "Loading weights\|MatMul8bitLt" || true

if [[ "$smoke_ec" -ne 0 ]]; then
  banner "SMOKE FAILED — aborting full 128K run"
  echo "smoke log: $OUT_DIR/pg19_ctx131072_smoke.log"
  exit "$smoke_ec"
fi

banner "smoke passed — proceeding to 5-chunk full run"

# ── Full: 5 chunks at 128K ─────────────────────────────────────
# Budget 15h — 5 chunks × projected ~150 min/chunk = ~12.5h, plus
# model load + dense prefill overhead.
banner "PG-19 ctx=131072 FULL (5 chunks, 15h budget)"
timeout 54000 "$PY" benchmarks/paper/pg19_perplexity.py \
  --context 131072 --num-chunks 5 \
  --tau-cov 0.995 --k-min 2 --k-max 128 \
  --ranking-fallback --ranking-r 1 \
  --epsilon-override 1e-4 \
  --output "$OUT_DIR/pg19_ctx131072.json" \
  > "$OUT_DIR/pg19_ctx131072.log" 2>&1
full_ec=$?
banner "PG-19 ctx=131072 FULL exit=$full_ec"
tail -40 "$OUT_DIR/pg19_ctx131072.log" \
  | grep -v "Loading weights\|MatMul8bitLt" || true

banner "PG-19 128K done (smoke=$smoke_ec, full=$full_ec)"
echo "Results in $OUT_DIR"

# Propagate the full run's exit code so timeouts/crashes don't get
# silently logged as success in automation. The smoke-failure branch
# above already exits non-zero; this mirrors it for the full run.
exit "$full_ec"
