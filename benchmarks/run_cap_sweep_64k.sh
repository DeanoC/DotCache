#!/usr/bin/env bash
# 64K cap sweep: measures cert tok/s for the bounded FP16 key cache
# across capacity values, with per-KV-group enabled (the paper's
# configured setting post-commit 320e6cb9).
#
# Runs the no-timer harness once per cap setting. The loose kernel
# (FAST_ATTEND=0) is skipped — we already have its numbers from the
# pre-PR-#96 baseline. This sweep is specifically to see how the
# new split-K kernel + OrderedDict priority LRU behave at each cap.
#
# Output: benchmarks/results/perf_tests_20260423/cap_<cap>_kvgroup.log
#         and a SUMMARY.md collating the tok/s numbers at the end.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/benchmarks/results/perf_tests_20260423"
mkdir -p "$OUT_DIR"

CAPS=(256 512 1024 2048)   # full-mirror covered by the separate dense/slow/fast run
STEPS=64
WARMUP=8

for cap in "${CAPS[@]}"; do
  out="$OUT_DIR/cap_${cap}_kvgroup.log"
  echo "=== cap=$cap per-KV-group ==="
  DOTCACHE_FP16_CACHE_BLOCKS="$cap" DOTCACHE_PER_KV_GROUP_TOPK=1 \
    "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/benchmarks/bench_decode_64k_no_timer.py" \
    --context-length 65536 \
    --decode-steps "$STEPS" \
    --warmup-steps "$WARMUP" \
    --modes fast \
    2>&1 | tee "$out"
  echo
done

echo "=== collating ==="
OUT_DIR="$OUT_DIR" python3 - <<'PY'
import os, re, pathlib
OUT_DIR = pathlib.Path(os.environ["OUT_DIR"])
lines = ["# 64K capacity sweep (per-KV-group, split-K kernel, OrderedDict LRU)", "",
         "| cap | mean ms/step | p50 ms/step | p95 ms/step | tok/s |",
         "|---|---:|---:|---:|---:|"]
for f in sorted(OUT_DIR.glob("cap_*_kvgroup.log")):
    cap = f.stem.split("_")[1]
    text = f.read_text()
    m = re.search(r"cert FAST_ATTEND=1\s+mean\s+([\d.]+) ms\s+p50\s+([\d.]+) ms\s+p95\s+([\d.]+) ms.*?tok/s\s+([\d.]+)", text)
    if m:
        lines.append(f"| {cap} | {m.group(1)} | {m.group(2)} | {m.group(3)} | {m.group(4)} |")
    else:
        lines.append(f"| {cap} | — | — | — | (no match, see log) |")
(OUT_DIR / "cap_sweep_SUMMARY.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY
