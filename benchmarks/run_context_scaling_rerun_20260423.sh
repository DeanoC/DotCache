#!/usr/bin/env bash
# Re-run the context-scaling sweep with the Paper-1 hybrid attend-all
# config (bench_certified_64k.py is patched in commit 33efcd2c).
#
# Output: benchmarks/results/context_scaling_rerun_20260423/
set +e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
OUT_DIR="$ROOT_DIR/benchmarks/results/context_scaling_rerun_20260423"
mkdir -p "$OUT_DIR"

PY="$ROOT_DIR/.venv/bin/python"

banner() { echo; echo "=== $(date -u +%FT%TZ) :: $* ==="; }

exec > >(tee -a "$OUT_DIR/orchestrator.log") 2>&1

banner "context-scaling Paper-1 rerun start"

# bench_certified_64k.py writes a fixed JSON path
# (benchmarks/results/certified_64k_int8model.json). Move the previous
# (Paper-2) JSON aside so the rerun's output is preserved separately and
# the old-vs-new can be diffed.
PREV_JSON="$ROOT_DIR/benchmarks/results/certified_64k_int8model.json"
if [[ -f "$PREV_JSON" ]]; then
  mv "$PREV_JSON" "$OUT_DIR/certified_64k_int8model.PAPER2_REJECTED.json"
  banner "moved previous Paper-2 JSON → $OUT_DIR/certified_64k_int8model.PAPER2_REJECTED.json"
fi

banner "running bench_certified_64k.py with Paper-1 config (8K..64K, 32 steps each)"
timeout 7200 "$PY" benchmarks/bench_certified_64k.py \
  > "$OUT_DIR/context_scaling.log" 2>&1
ec=$?
banner "bench_certified_64k.py exit=$ec"

# Move the new JSON into the rerun dir (the bench writes back to the
# canonical path).
if [[ -f "$PREV_JSON" ]]; then
  cp "$PREV_JSON" "$OUT_DIR/certified_64k_int8model.PAPER1.json"
  banner "copied Paper-1 JSON → $OUT_DIR/certified_64k_int8model.PAPER1.json"
fi

# Print the table tail so the user can see headline numbers without
# opening the JSON.
tail -50 "$OUT_DIR/context_scaling.log" \
  | grep -v "Loading weights\|MatMul8bitLt" || true

banner "context-scaling Paper-1 rerun done (exit=$ec)"
echo "Results in $OUT_DIR"
