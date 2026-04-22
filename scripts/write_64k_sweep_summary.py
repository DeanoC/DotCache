#!/usr/bin/env python3
"""Aggregator for the 64K throughput sweep. Headline result for the paper's
perf section — the context length where the tiered architecture is load-bearing."""
from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    sweep_dir = Path("benchmarks/results/perf_tests_20260422/cache_sweep_64k")
    out_path = sweep_dir / "SUMMARY.md"

    rows = [
        ("dense",        "dense (baseline)",       "dense",        None),
        ("cert_cap256",  "certified cap=256",      "certified",    256),
        ("cert_cap512",  "certified cap=512",      "certified",    512),
        ("cert_cap1024", "certified cap=1024",     "certified",    1024),
        ("cert_cap4096", "certified cap=4096 (=corpus)", "certified", 4096),
        ("cert_full",    "certified full mirror",  "certified",    None),
    ]

    def collect(tag: str, cfg_key: str) -> dict | None:
        p = sweep_dir / f"64k_{tag}.json"
        if not p.exists():
            return None
        j = json.loads(p.read_text())
        summary = j.get("summary", {}).get(cfg_key, {})
        reps = [r for r in j.get("per_config", {}).get(cfg_key, []) if "skipped" not in r]
        if not reps:
            return None
        hits = sum(r.get("fp16_cache_hits", 0) for r in reps)
        misses = sum(r.get("fp16_cache_misses", 0) for r in reps)
        total = hits + misses
        return {
            "tok_per_sec": summary.get("tok_per_sec_mean"),
            "tok_per_sec_std": summary.get("tok_per_sec_std"),
            "p50_ms": summary.get("ms_per_token_p50_median"),
            "p95_ms": summary.get("ms_per_token_p95_median"),
            "prefill_ms": summary.get("prefill_time_ms_median"),
            "gpu_peak_mb": summary.get("gpu_mem_peak_mb_max"),
            "hit_rate": (hits / total) if total else None,
            "mb_per_step": (
                sum(r.get("fp16_cache_h2d_mb_per_decode_step", 0) for r in reps) / len(reps)
                if reps else 0.0
            ),
            "n_repeats": summary.get("n_repeats"),
        }

    lines: list[str] = []
    W = lines.append
    W("# 64K throughput sweep — the paper's headline perf regime")
    W("")
    W("**Model:** `NousResearch/Meta-Llama-3.1-8B` (INT8 bitsandbytes)  ")
    W("**Hardware:** NVIDIA RTX PRO 6000 Blackwell (sm_120), 96 GB VRAM  ")
    W("**Setup:** 65536-token pg19 prefill + 256-token argmax decode (warmup 16, timed 240), "
      "3 repeats per config.")
    W("**Corpus:** 65536 / 16 = **4096 blocks**. `K_max=128` = 3.1% of corpus — the regime "
      "where the tiered architecture is designed to win.")
    W("")
    W("| Config | tok/s ± std | p50 ms/tok | p95 ms/tok | Hit rate | H2D MB/step | Prefill ms | GPU peak MB |")
    W("|---|---|---|---|---|---|---|---|")
    dense_tps = None
    for tag, label, cfg, cap in rows:
        r = collect(tag, cfg)
        if r is None:
            W(f"| {label} | — | — | — | — | — | — | — |")
            continue
        if tag == "dense":
            dense_tps = r["tok_per_sec"]
        tps_str = f"{r['tok_per_sec']:.2f} ± {r['tok_per_sec_std']:.2f}"
        hit = f"{r['hit_rate']*100:.2f}%" if r["hit_rate"] is not None else "—"
        mb = f"{r['mb_per_step']:.1f}" if r["mb_per_step"] else "0"
        W(f"| {label} | {tps_str} | {r['p50_ms']:.1f} | {r['p95_ms']:.1f} | "
          f"{hit} | {mb} | {r['prefill_ms']:.0f} | {r['gpu_peak_mb']:.0f} |")
    W("")
    W("## Overhead vs dense")
    W("")
    if dense_tps:
        W(f"| Config | tok/s | Overhead vs dense |")
        W("|---|---|---|")
        for tag, label, cfg, cap in rows:
            r = collect(tag, cfg)
            if r is None or tag == "dense":
                continue
            oh = (dense_tps / r["tok_per_sec"] - 1.0) * 100.0 if r["tok_per_sec"] else float("inf")
            W(f"| {label} | {r['tok_per_sec']:.2f} | +{oh:.1f}% |")
        W("")
    W("## Key reads")
    W("")
    W("**(1) The cache knee is still at corpus size, even at 64K.** Hit rate at "
      "cap=256, 512, 1024 all sit at 0.55–0.77% — the theoretical prediction that "
      "4 Q heads × 3.1% ≈ 22% per KV group, so cap=1024 should work, didn't hold. "
      "The per-layer union saturates near corpus size because **there are 8 KV "
      "groups**: even if each group only needs 22% of the corpus, "
      "`4096 · (1 − 0.78⁸) ≈ 3500 blocks` (86% of corpus) across the 8 groups. "
      "Per-layer working set is driven by the 8-way KV diversity × 4-way Q head "
      "fan-out within each group, not by the K_max/corpus ratio of any single head.")
    W("")
    W("**(2) The cache isn't the dominant cost at 64K — the Triton kernel is.** "
      "Even at cap=∞ (full mirror, zero decode-time H2D), certified is 1.93 tok/s "
      "vs dense 19.92 tok/s — a **10.3× slowdown with no page-in cost**. That's the "
      "cost of the Phase-1 INT8 scoring + Phase-2 hybrid attend Triton kernel vs "
      "torch's Flash Attention. At this context length, optimising the Triton "
      "kernel would reclaim ~9× of the ~10× gap; optimising the cache reclaims at "
      "best the ~3× gap between full-mirror (1.93 tok/s) and corpus-cap (0.64 tok/s).")
    W("")
    W("**(3) The sub-corpus cache regime is prohibitively expensive at 64K.** "
      "H2D MB/step at cap<corpus is ~2.6 GB — enough to push p95 latency above 2 "
      "seconds per decoded token. Any practical serving path must either size the "
      "cache at corpus or use a structural fix (per-KV-group selection, see below) "
      "to break the 8-way KV diversity.")
    W("")
    W("**(4) The per-KV-group structural fix deserves a 64K test.** At 8K it didn't "
      "help because K_max/corpus was already 25% — saturating the union anyway. At "
      "64K with K_max=128 (3.1%), collapsing 32 Q heads into 8 groups reduces the "
      "total independent-selections count from 32 to 8, halving the upper bound on "
      "per-layer union. See `cache_sweep_tau/SUMMARY.md` for the 8K negative "
      "result; a 64K re-test is the remaining experiment before the paper has to "
      "commit to 'cache-sized scratch' as the deployment recommendation.")
    W("")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
