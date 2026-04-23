#!/usr/bin/env bash
# Overnight orchestration: full perf suite then quality sweeps with CIs.
#
# Runs each stage sequentially, logs to a timestamped file under
# benchmarks/results/overnight_20260423/, and continues past failures.
# A final SUMMARY.md collates the headline numbers from each stage.
#
# Expected wall time: ~6 hours (dominated by PG-19 + RULER quality sweeps).
set +e  # continue past failures

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
OUT_DIR="$ROOT_DIR/benchmarks/results/overnight_20260423"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/orchestrator.log"

PY="$ROOT_DIR/.venv/bin/python"

banner() { echo; echo "=== $(date -u +%FT%TZ) :: $* ==="; }

exec > >(tee -a "$LOG") 2>&1

banner "overnight suite start"

# ── Stage 1: Perf ────────────────────────────────────────────────
#
# 1a. 64K full-mirror dense/slow/fast (already running in its own
#     task; this script is orchestrating the REST, so skip the first
#     stage here and consume its log when we write SUMMARY.md).
#
# 1b. Capacity sweep at 64K per-KV-group.
banner "1b: 64K cap sweep (per-KV-group, new LRU + kernel)"
for cap in 256 512 1024 2048; do
  out="$OUT_DIR/cap_${cap}_kvgroup.log"
  banner "  cap=$cap → $out"
  DOTCACHE_FP16_CACHE_BLOCKS="$cap" DOTCACHE_PER_KV_GROUP_TOPK=1 \
    timeout 1800 "$PY" "$ROOT_DIR/benchmarks/bench_decode_64k_no_timer.py" \
    --context-length 65536 --decode-steps 64 --warmup-steps 8 --modes fast \
    > "$out" 2>&1
  tail -3 "$out" | grep -v "Loading weights\|MatMul8bitLt" || true
done

# 1c. Context-scaling sweep (8K/16K/32K/48K/64K, dense vs cert).
banner "1c: context-scaling (8K..64K, 32 decode steps per cell)"
timeout 3600 "$PY" benchmarks/bench_certified_64k.py \
  > "$OUT_DIR/context_scaling.log" 2>&1
tail -20 "$OUT_DIR/context_scaling.log" | grep -v "Loading weights\|MatMul8bitLt" || true

# ── Stage 2: Value-error sweep at longer contexts ──────────────────
# Confirm the default flip to "tight" holds as contexts grow.
for ctx in 16384 32768; do
  out="$OUT_DIR/value_error_sweep_ctx${ctx}.log"
  banner "2: value-error sweep at ctx=$ctx → $out"
  timeout 1800 "$PY" benchmarks/sweep_value_error_bound.py \
    --context "$ctx" --decode-steps 16 --warmup-steps 0 \
    --output "$OUT_DIR/value_error_sweep_ctx${ctx}.json" \
    > "$out" 2>&1
  tail -20 "$out" | grep -v "Loading weights\|MatMul8bitLt" || true
done

# ── Stage 3: PG-19 perplexity with CIs, across contexts ────────────
# Uses the new per_chunk + per_chunk_bpt_stats pipeline from PR #98.
# 20 chunks × 3 contexts ≈ 60 chunks. Each chunk is ~30-60s for the
# certified path at 16K+; budget accordingly.
for ctx in 4096 8192 16384; do
  out_json="$OUT_DIR/pg19_ctx${ctx}.json"
  out_log="$OUT_DIR/pg19_ctx${ctx}.log"
  banner "3: PG-19 ppl at ctx=$ctx (20 chunks, tau_cov=0.995)"
  timeout 5400 "$PY" benchmarks/paper/pg19_perplexity.py \
    --context "$ctx" --num-chunks 20 \
    --tau-cov 0.995 --k-min 2 --k-max 128 \
    --ranking-fallback --ranking-r 1 \
    --epsilon-override 1e-4 \
    --output "$out_json" \
    > "$out_log" 2>&1
  tail -30 "$out_log" | grep -v "Loading weights\|MatMul8bitLt" || true
done

# ── Stage 4: RULER with CIs ─────────────────────────────────────────
# 20 samples × 7 subtasks × 2 contexts. The ruler.py harness runs
# dense + cert paired per sample, so we get the paired McNemar counts
# needed for the new paired diff CI in PR #98's aggregator.
for ctx in 4096 8192; do
  out_json="$OUT_DIR/ruler_ctx${ctx}.json"
  out_log="$OUT_DIR/ruler_ctx${ctx}.log"
  banner "4: RULER at ctx=$ctx (20 samples × 7 subtasks, calibrated cert)"
  timeout 7200 "$PY" benchmarks/paper/ruler.py \
    --contexts "$ctx" --num-samples 20 \
    --tau-cov 0.995 --k-min 2 --k-max 128 \
    --ranking-fallback --ranking-r 1 \
    --default-epsilon 1e-4 \
    --output "$out_json" \
    > "$out_log" 2>&1
  tail -30 "$out_log" | grep -v "Loading weights\|MatMul8bitLt" || true
done

# Run the RULER aggregator over all ruler JSONs we just produced,
# picking up the new Wilson + paired-diff CI columns from PR #98.
banner "4a: aggregate_ruler on the overnight JSONs"
# aggregate_ruler expects specific filenames — point it at our dir via
# symlinks with the expected tags.
AGG_DIR="$OUT_DIR/ruler_agg_input"
mkdir -p "$AGG_DIR"
for ctx_k in 4 8; do
  ctx=$((ctx_k * 1024))
  src="$OUT_DIR/ruler_ctx${ctx}.json"
  if [[ -f "$src" ]]; then
    # aggregate_ruler.CONFIGS uses eps0_XK and calibrated_XK names; our
    # run is calibrated, so map to calibrated_{4,8}k.json.
    ln -sf "$src" "$AGG_DIR/calibrated_${ctx_k}k.json"
  fi
done
timeout 120 "$PY" benchmarks/paper/aggregate_ruler.py \
  --dir "$AGG_DIR" \
  --csv "$OUT_DIR/ruler_summary.csv" \
  > "$OUT_DIR/ruler_summary.log" 2>&1
tail -50 "$OUT_DIR/ruler_summary.log" || true

# ── Final summary ─────────────────────────────────────────────────
banner "writing SUMMARY.md"
"$PY" - <<PY > "$OUT_DIR/SUMMARY.md" 2>&1 || true
import json
import re
from pathlib import Path

OUT = Path("$OUT_DIR")

def tail_match(path, pattern, limit=200, flags=0):
    """Return first regex match in the last \`limit\` lines of \`path\`, or None."""
    if not Path(path).exists():
        return None
    lines = Path(path).read_text().splitlines()
    blob = "\n".join(lines[-limit:])
    m = re.search(pattern, blob, flags)
    return m.groupdict() if m else None

def collect_cap_sweep():
    rows = []
    for cap in (256, 512, 1024, 2048):
        r = tail_match(
            OUT / f"cap_{cap}_kvgroup.log",
            r"cert FAST_ATTEND=1\s+mean\s+(?P<mean>[\d.]+) ms\s+p50\s+(?P<p50>[\d.]+) ms\s+p95\s+(?P<p95>[\d.]+) ms.*?tok/s\s+(?P<toks>[\d.]+)",
        )
        if r:
            rows.append(f"| {cap} | {r['mean']} | {r['p50']} | {r['p95']} | {r['toks']} |")
        else:
            rows.append(f"| {cap} | — | — | — | (run failed) |")
    return rows

def collect_value_error(ctx):
    # Prefer the JSON output; fall back to the multi-line log if the
    # JSON is missing. The bench writes loose/tight/ratio on separate
    # lines, so the regex must span newlines (re.DOTALL).
    j = OUT / f"value_error_sweep_ctx{ctx}.json"
    if j.exists():
        try:
            data = json.loads(j.read_text())
            s = data["summary"]
            return {
                "l": f"{s['loose_mean']:.4f}",
                "t": f"{s['tight_mean']:.4f}",
                "ratio": f"{s['ratio_mean']:.4f}",
            }
        except (KeyError, json.JSONDecodeError):
            pass
    return tail_match(
        OUT / f"value_error_sweep_ctx{ctx}.log",
        r"loose bound\s+mean=(?P<l>[\d.]+).*?tight bound\s+mean=(?P<t>[\d.]+).*?tight / loose ratio\s+mean=(?P<ratio>[\d.]+)",
        limit=60,
        flags=re.DOTALL,
    )

print("# Overnight 2026-04-23 — summary")
print()
print("> **Vocabulary note (Paper 1):** the certified path attends *every*")
print("> block. Top-K* blocks use FP16 keys; the tail uses INT8 keys. The")
print("> metric formerly logged as `skip_rate` is the **INT8-tail rate** —")
print("> the fraction of blocks served from the cheap INT8-key path, not a")
print("> drop rate. Logs from runs predating commit b7ec3165 still print the")
print("> word \"skip\"; read it as \"int8_tail\".")
print()
print("## Perf stage")
print()
print("### 1b. 64K cap sweep (per-KV-group)")
print()
print("| cap | mean ms/step | p50 ms/step | p95 ms/step | tok/s |")
print("|---|---:|---:|---:|---:|")
for row in collect_cap_sweep():
    print(row)
print()
print("### 1c. Context-scaling summary (tail of log)")
print()
log = OUT / "context_scaling.log"
if log.exists():
    lines = log.read_text().splitlines()
    # The bench prints a final table; dump the tail.
    print("\`\`\`")
    for line in lines[-50:]:
        if "Loading weights" in line or "MatMul8bitLt" in line:
            continue
        print(line)
    print("\`\`\`")
else:
    print("(context_scaling.log missing)")
print()
print("## Value-error sweep at longer contexts")
print()
print("| context | loose mean | tight mean | tight/loose |")
print("|---|---:|---:|---:|")
for ctx in (16384, 32768):
    r = collect_value_error(ctx)
    if r:
        print(f"| {ctx} | {r['l']} | {r['t']} | {r['ratio']} |")
    else:
        print(f"| {ctx} | — | — | (run failed) |")
print()
print("## Quality stage")
print()
print("### PG-19 perplexity (with per-chunk CIs from PR #98)")
print()
for ctx in (4096, 8192, 16384):
    log = OUT / f"pg19_ctx{ctx}.log"
    if not log.exists():
        print(f"- ctx={ctx}: (run missing)")
        continue
    lines = log.read_text().splitlines()
    tail = [l for l in lines[-40:] if "Loading weights" not in l and "MatMul8bitLt" not in l]
    print(f"- ctx={ctx}:")
    for l in tail[-10:]:
        print(f"    {l}")
print()
print("### RULER aggregated")
print()
log = OUT / "ruler_summary.log"
if log.exists():
    print("\`\`\`")
    print(log.read_text()[-3000:])
    print("\`\`\`")
else:
    print("(ruler_summary.log missing)")
PY

banner "overnight suite done"
echo "Results in $OUT_DIR"
echo "Summary: $OUT_DIR/SUMMARY.md"
