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
    W("## Key read — where the cache starts paying off")
    W("")
    W("*(Fill in after data lands. Looking for:*")
    W("*  - At what capacity does hit rate jump? Hypothesis: between 1024 and 4096.*")
    W("*  - Does cap=512 (12.5% corpus) already work? Union arithmetic says 4 Q heads × 3.1% ≈ 12% union,*")
    W("*    so even cap=512 might be enough per-layer group.*")
    W("*  - What's the certified overhead vs dense at the best cache point?)*")
    W("")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
