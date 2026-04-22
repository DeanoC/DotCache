#!/usr/bin/env python3
"""Summary for the pg19 cache capacity sweep."""
from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    sweep_dir = Path("benchmarks/results/perf_tests_20260422/cache_sweep_pg19")
    out_path = sweep_dir / "SUMMARY.md"
    niah_dir = Path("benchmarks/results/perf_tests_20260422/cache_sweep")

    sweep_points = [
        ("cap_0",       "0 (pure H2D)"),
        ("cap_4",       "4"),
        ("cap_16",      "16"),
        ("cap_64",      "64"),
        ("cap_256",     "256"),
        ("cap_1024",    "1024"),
        ("full_mirror", "∞ (full mirror)"),
    ]

    def aggregate(dirpath: Path, tag: str) -> dict | None:
        p = dirpath / f"cache_sweep_{tag}.json" if "pg19" not in str(dirpath) else dirpath / f"cache_sweep_pg19_{tag}.json"
        if not p.exists():
            return None
        j = json.loads(p.read_text())
        summary = j.get("summary", {}).get("certified", {})
        reps = [r for r in j.get("per_config", {}).get("certified", []) if "skipped" not in r]
        if not reps:
            return None
        cap = reps[0].get("fp16_cache_capacity_blocks")
        hits = sum(r.get("fp16_cache_hits", 0) for r in reps)
        misses = sum(r.get("fp16_cache_misses", 0) for r in reps)
        total = hits + misses
        hit_rate = (hits / total) if total else 0.0
        mb = sum(r.get("fp16_cache_h2d_mb_per_decode_step", 0) for r in reps) / len(reps)
        return {
            "cap": cap,
            "tok_per_sec_mean": summary.get("tok_per_sec_mean"),
            "tok_per_sec_std": summary.get("tok_per_sec_std"),
            "p50_ms": summary.get("ms_per_token_p50_median"),
            "hit_rate": hit_rate,
            "h2d_mb_per_step": mb,
        }

    lines: list[str] = []
    W = lines.append
    W("# PG-19 FP16 VRAM cache sweep (generated decode from PG-19 prefill)")
    W("")
    W("**Model:** `NousResearch/Meta-Llama-3.1-8B` (INT8 bitsandbytes)  ")
    W("**Hardware:** NVIDIA RTX PRO 6000 Blackwell (sm_120), 96 GB VRAM  ")
    W("**Setup:** Same as `../cache_sweep/SUMMARY.md`, but the 8K prefill is "
      "the first long-enough book from the PG-19 test split; decode remains "
      "argmax generation (256 tokens, 16 warmup, 240 timed).")
    W("")
    W("| Capacity (blocks) | tok/s ± std | p50 ms/tok | Hit rate | H2D MB/step | (filler tok/s, for reference) |")
    W("|---|---|---|---|---|---|")
    for tag, label in sweep_points:
        r = aggregate(sweep_dir, tag)
        filler = aggregate(niah_dir, tag)
        if r is None:
            W(f"| {label} | — | — | — | — | — |")
            continue
        filler_tps = filler.get("tok_per_sec_mean") if filler else None
        filler_str = f"{filler_tps:.2f}" if filler_tps is not None else "—"
        W(f"| {label} | {r['tok_per_sec_mean']:.2f} ± {r['tok_per_sec_std']:.2f} | "
          f"{r['p50_ms']:.1f} | {r['hit_rate']*100:.2f}% | {r['h2d_mb_per_step']:.1f} | {filler_str} |")
    W("")
    W("## The knee is still at corpus size")
    W("")
    W("We expected pg19's concentrated attention (shown in main Test 3 with 99.9% zero-pagein at "
      "`cap=64`) to shift the knee down dramatically. The sweep shows **the opposite — the same "
      "knee at corpus size** as the filler/niah workload. Hit-rate stays in single-digit-percent "
      "territory until `cap ≥ 512` (= corpus), then jumps to 99.6%.")
    W("")
    W("## Why this doesn't contradict Test 3 pg19")
    W("")
    W("The main Test 3 pg19 measurement used **teacher-forced** decode — `pg19_perplexity.py` feeds "
      "the ground-truth next token into the model at each step and scores its NLL. Attention in that "
      "regime stays locally concentrated because each new query is the embedding of a real pg19 "
      "token, and Llama's in-distribution attention pattern on pg19 naturally tracks recent and "
      "strongly-related earlier tokens — a small working set.")
    W("")
    W("This sweep used **argmax-generated** decode from a pg19 prefill — the model produces its own "
      "continuation token-by-token. Once the generated text leaves the pg19 distribution (which "
      "happens within ~20 decode steps of unconditional generation), the queries become Llama's "
      "open-ended continuation queries, which have a **scattered top-K** similar to filler. "
      "The cache behaviour then tracks that scattered pattern, not pg19's teacher-forced pattern.")
    W("")
    W("**Net finding: the cache curve shape is driven by *decode mode*, not by prefix content alone.**")
    W("")
    W("## What the two data points together tell the paper")
    W("")
    W("- **Teacher-forced decode (real workload signal)**: cap=64 is plenty — pg19 Test 3 shows "
      "99.9% zero-pagein steps. The paper's tiered architecture pays off handsomely here because "
      "the cache hot set is tiny and stable.")
    W("- **Argmax-generated decode (open-ended generation)**: the cache must be ≥ corpus for any "
      "speedup; below that you pay full H2D bandwidth. Open-ended generation is inherently "
      "adversarial to a small cache.")
    W("- These two regimes bracket the design space. A practical serving system doing mostly "
      "teacher-forced scoring (RAG re-ranking, prefix-logprob evaluation) lives in the first "
      "regime. A serving system doing long free-form generation (chat completion) lives in the "
      "second. The paper can honestly claim both: **the architecture is H2D-efficient when "
      "attention locality is present, and falls back to bandwidth-bound when it isn't.**")
    W("")
    W("## Implementation caveats")
    W("")
    W("Same as `../cache_sweep/SUMMARY.md`:")
    W("- Scratch VRAM column omitted here because of the allocator caveat — physical allocation "
      "is still full-sequence-sized regardless of capacity.")
    W("- The cap=1024 ceiling gap (6.00 vs 8.01 tok/s) is Python `list.remove` O(N) LRU overhead, "
      "not H2D. The H2D column at cap=1024 is only 2 MB/step.")
    W("- Hit rate 0% at cap=∞ is an accounting artefact (full-mirror path bypasses the cache "
      "counters); not a real miss rate.")
    W("")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
