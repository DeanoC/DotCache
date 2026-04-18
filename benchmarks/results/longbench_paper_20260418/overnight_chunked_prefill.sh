#!/bin/bash
# Overnight: after the LongBench paper chain finishes (PID 974861), run
# the chunked prefill validation smoke at 4K, 8K and 16K.  Each writes
# its own JSON so partial results survive a mid-run failure.
set -e
cd /DotCache
OUTDIR=benchmarks/results/longbench_paper_20260418

echo "=== waiting for LongBench chain (PID 974861) ==="
tail --pid=974861 -f /dev/null
echo "=== LongBench chain done $(date -Iseconds) ==="

echo "=== generating paper-grade summary.csv ==="
python -u benchmarks/paper/aggregate_longbench.py \
  --dir "$OUTDIR" --csv "$OUTDIR/summary.csv" 2>&1 | tee "$OUTDIR/aggregate_final.log"

echo "=== starting chunked prefill smoke ==="

run() {
  local tag=$1; local target=$2
  echo "=== chunked_prefill_smoke ${tag} start $(date -Iseconds) ==="
  python -u benchmarks/paper/chunked_prefill_smoke.py \
    --target-tokens $target \
    --output "$OUTDIR/chunked_prefill_${tag}.json" \
    2>&1 | tee "$OUTDIR/chunked_prefill_${tag}.log"
  echo "=== chunked_prefill_smoke ${tag} done $(date -Iseconds) ==="
}

# Start at 4K — baseline validation that chunked KV matches one-shot.
run 4k 4096
# 8K — should show noticeable peak reduction with chunk=1024.
run 8k 8192
# 16K — the real payoff: one-shot might OOM or peak near 26 GB, chunked should halve it.
run 16k 16384

echo "=== ALL CHUNKED PREFILL SMOKES DONE $(date -Iseconds) ==="
