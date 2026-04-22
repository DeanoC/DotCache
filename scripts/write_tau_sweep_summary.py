#!/usr/bin/env python3
"""Aggregator for the tau_cov / per-KV-group sweep at 8K."""
from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    sweep_dir = Path("benchmarks/results/perf_tests_20260422/cache_sweep_tau")
    out_path = sweep_dir / "SUMMARY.md"

    rows = [
        ("cap64_tau0995",          "0.995", "per-Q-head"),
        ("cap64_tau099",           "0.990", "per-Q-head"),
        ("cap64_tau098",           "0.980", "per-Q-head"),
        ("cap64_tau095",           "0.950", "per-Q-head"),
        ("cap64_tau0995_kvgroup",  "0.995", "per-KV-group (8 groups)"),
    ]

    def collect(tag: str) -> dict | None:
        p = sweep_dir / f"{tag}.json"
        if not p.exists():
            return None
        j = json.loads(p.read_text())
        summary = j.get("summary", {}).get("certified", {})
        reps = [r for r in j.get("per_config", {}).get("certified", []) if "skipped" not in r]
        if not reps:
            return None
        hits = sum(r.get("fp16_cache_hits", 0) for r in reps)
        misses = sum(r.get("fp16_cache_misses", 0) for r in reps)
        total = hits + misses
        per_layer_union_est = (total / len(reps)) / 32.0 / 256 if reps else 0.0
        return {
            "tok_per_sec": summary.get("tok_per_sec_mean"),
            "tok_per_sec_std": summary.get("tok_per_sec_std"),
            "p50_ms": summary.get("ms_per_token_p50_median"),
            "hit_rate": (hits / total) if total else 0.0,
            "misses_per_step": misses / len(reps) / 256,
            "per_layer_union_avg": per_layer_union_est * 256,
            "mb_per_step": sum(r.get("fp16_cache_h2d_mb_per_decode_step", 0) for r in reps) / len(reps),
        }

    lines: list[str] = []
    W = lines.append
    W("# τ_cov × selection-mode sweep at 8K (pg19 prefill, cap=64)")
    W("")
    W("**Question.** Does tuning tau_cov or collapsing Q heads into KV-head groups "
      "materially shrink the per-layer FP16 working set, and thus make a small cache useful?")
    W("")
    W("**Setup.** Same as the main perf Test 1: 8K pg19 prefill, 256-token argmax decode "
      "(16 warmup, 240 timed), cap=64, 5 repeats per config.")
    W("")
    W("| τ_cov | Selection mode | tok/s ± std | p50 ms | Hit rate | Misses / step | Per-layer union avg | H2D MB/step |")
    W("|---|---|---|---|---|---|---|---|")
    for tag, tau_label, mode_label in rows:
        r = collect(tag)
        if r is None:
            W(f"| {tau_label} | {mode_label} | — | — | — | — | — | — |")
            continue
        W(f"| {tau_label} | {mode_label} | "
          f"{r['tok_per_sec']:.2f} ± {r['tok_per_sec_std']:.2f} | "
          f"{r['p50_ms']:.1f} | {r['hit_rate']*100:.2f}% | "
          f"{r['misses_per_step']:.0f} | {r['per_layer_union_avg']:.0f} | "
          f"{r['mb_per_step']:.1f} |")
    W("")
    W("## Findings")
    W("")
    W("*(Fill in after the sweep completes — looking for: does lower τ_cov measurably "
      "shrink per-layer union? Does per-KV-group do so more effectively at τ_cov=0.995?)*")
    W("")
    W("## Caveat — this is still 8K")
    W("")
    W("8K is the wrong regime to showcase the tiered architecture. At 8K the K_max/corpus "
      "ratio is 128/512 = 25% per Q head, so the GQA union is fundamentally bounded below "
      "anyway. The real test is at longer contexts where K_max/corpus becomes small "
      "(64K: 3.1%, 128K: 1.6%) and per-layer unions can realistically stay under a small "
      "cache. See `cache_sweep_64k/SUMMARY.md` for the load-bearing throughput story.")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
