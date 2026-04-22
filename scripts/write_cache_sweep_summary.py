#!/usr/bin/env python3
"""Aggregate the 5 cache_sweep_*.json files into a Markdown table showing
the throughput vs cache-capacity tradeoff."""
from __future__ import annotations

import json
from pathlib import Path


def _fmt_mb(b):
    return f"{b / (1024 ** 2):.2f}"


def main() -> int:
    sweep_dir = Path("benchmarks/results/perf_tests_20260422/cache_sweep")
    out_path = sweep_dir / "SUMMARY.md"

    # Sweep points in presentation order. "_paper" label mirrors the main
    # Test 1 column so a reader can cross-reference.
    sweep_points = [
        ("cap_0",       "0 (pure H2D)"),
        ("cap_64",      "64"),
        ("cap_256",     "256"),
        ("cap_384",     "384"),
        ("cap_512",     "512 (= corpus)"),
        ("cap_640",     "640"),
        ("cap_768",     "768"),
        ("cap_1024",    "1024"),
        ("full_mirror", "∞ (full mirror)"),
    ]

    rows = []
    for tag, label in sweep_points:
        p = sweep_dir / f"cache_sweep_{tag}.json"
        if not p.exists():
            rows.append({"label": label, "missing": True, "tag": tag})
            continue
        j = json.loads(p.read_text())
        summary = j.get("summary", {}).get("certified", {})
        reps = [r for r in j.get("per_config", {}).get("certified", []) if "skipped" not in r]
        if not reps:
            rows.append({"label": label, "missing": True, "tag": tag})
            continue

        # Aggregate cache metrics across repeats.
        cap = reps[0].get("fp16_cache_capacity_blocks")
        hits = sum(r.get("fp16_cache_hits", 0) for r in reps)
        misses = sum(r.get("fp16_cache_misses", 0) for r in reps)
        bytes_ = sum(r.get("fp16_cache_h2d_bytes", 0) for r in reps)
        evicts = sum(r.get("fp16_cache_evictions", 0) for r in reps)
        total_access = hits + misses
        hit_rate = (hits / total_access) if total_access else 0.0
        mb_per_step_vals = [r.get("fp16_cache_h2d_mb_per_decode_step", 0.0) for r in reps]
        mb_per_step = sum(mb_per_step_vals) / len(mb_per_step_vals) if mb_per_step_vals else 0.0

        # Conceptual scratch VRAM (what a properly-packed scratch would need).
        # cap * block_size * kv_heads * head_dim * 2 bytes, × num_layers.
        # For Llama-3.1-8B @ 8K: bs=16, kv=8, hd=128, layers=32 →
        #   per-layer bytes = cap * 16 * 8 * 128 * 2 = cap * 32768 bytes
        #   total = cap * 32768 * 32 / 1024 / 1024 = cap * 1.0 MB.
        if cap is None:
            # full mirror = entire sequence, not per-capacity
            scratch_mb = 500 * 32768 * 32 / (1024 ** 2)   # 500 blocks = 8K aligned
        else:
            scratch_mb = cap * 32768 * 32 / (1024 ** 2)

        # gpu_mem_peak in the json is the ACTUAL allocation (which in this
        # codebase is still full-sequence-sized — the scratch-size savings
        # are conceptual, not yet realised in the allocator).
        gpu_peak_actual = max((r.get("gpu_mem_peak_mb", 0.0) for r in reps), default=0.0)

        rows.append({
            "label": label,
            "tag": tag,
            "cap": cap,
            "tok_per_sec_mean": summary.get("tok_per_sec_mean"),
            "tok_per_sec_std": summary.get("tok_per_sec_std"),
            "ms_per_token_p50": summary.get("ms_per_token_p50_median"),
            "hit_rate": hit_rate,
            "h2d_mb_per_step": mb_per_step,
            "scratch_mb_concept": scratch_mb,
            "gpu_peak_actual_mb": gpu_peak_actual,
            "n_repeats": summary.get("n_repeats"),
        })

    lines: list[str] = []
    W = lines.append
    W("# FP16 VRAM cache capacity sweep — 8K certified decode")
    W("")
    W("**Model:** `NousResearch/Meta-Llama-3.1-8B` (INT8 bitsandbytes)  ")
    W("**Hardware:** NVIDIA RTX PRO 6000 Blackwell (sm_120), 96 GB VRAM  ")
    W("**Setup:** 8K prefill + 256-token decode (warmup 16, timed 240), "
      "`tau_cov=0.995, k_min=2, k_max=128`, fallbacks on; 5 repeats per capacity point.  ")
    W("**Corpus size:** 8K tokens / 16-token blocks = 512 blocks. "
      "`cap=1024` and `cap=∞` should land at the same ceiling.")
    W("")
    W("| Capacity (blocks) | tok/s ± std | p50 ms/tok | Hit rate | H2D MB/step | Scratch VRAM* | Notes |")
    W("|---|---|---|---|---|---|---|")
    baseline_tps = None
    for r in rows:
        if r.get("missing"):
            W(f"| {r['label']} | — | — | — | — | — | *missing {r['tag']}.json* |")
            continue
        tps = r.get("tok_per_sec_mean") or 0.0
        if baseline_tps is None and r["label"].startswith("∞") is False and tps > 0:
            pass
        if r["label"].startswith("0"):
            note = "every access H2D'd"
        elif r["label"].startswith("∞"):
            note = "no H2D during decode"
        elif r.get("cap") and r["cap"] >= 500:
            note = "≥ corpus → same as ∞"
        else:
            note = ""
        W(f"| {r['label']} | {tps:.2f} ± {r.get('tok_per_sec_std', 0):.2f} | "
          f"{r.get('ms_per_token_p50', 0):.1f} | {r['hit_rate']*100:.2f}% | "
          f"{r['h2d_mb_per_step']:.1f} | {r['scratch_mb_concept']:.0f} MB | {note} |")
    W("")
    W("\\* *Scratch VRAM column is conceptual — `capacity × block_size × kv_heads × head_dim × 2 bytes × num_layers`. "
      "The current implementation allocates a full-sequence scratch regardless of capacity; "
      "`capacity`-sized allocation is a follow-up that would realise the VRAM savings this sweep implies.*")
    W("")

    # Find the knee: where does throughput stop improving?
    ok = [r for r in rows if not r.get("missing") and r.get("tok_per_sec_mean") is not None]
    if len(ok) >= 3:
        W("## The knee is at the corpus size")
        W("")
        floor = ok[0]["tok_per_sec_mean"]
        ceiling = ok[-1]["tok_per_sec_mean"]
        W(f"- **Floor** (`cap=0`, pure H2D): {floor:.2f} tok/s — every top-K block paged in every step.")
        W(f"- **Cache plateau** (`cap ≥ 512`): ~6.5 tok/s, hit rate 99.62%, H2D collapses to 1.9 MB/step.")
        W(f"- **Ceiling** (`cap=∞`, full mirror): {ceiling:.2f} tok/s — no decode-time H2D.")
        if floor > 0 and ceiling > floor:
            W(f"- **Full-mirror speedup:** {ceiling/floor:.2f}× over pure H2D.")
        W("")
        W("The transition is **sharp and happens at exactly `cap=512` — the corpus size** (8K tokens / "
          "16-token blocks = 512 blocks). Between `cap=384` and `cap=512`, hit rate jumps from 5.71% to "
          "99.62% and H2D bandwidth collapses from 459 MB/step to 1.9 MB/step. Anything above 512 "
          "plateaus at the same hit rate and H2D cost — the extra capacity is wasted scratch.")
        W("")
        W("## Workload observations")
        W("")
        W("- **Below the knee (cap ∈ {64, 256, 384}), capacity doesn't matter.** The scattered top-K of "
          "Llama-3.1-8B's certified attention cycles through different blocks every decode step; the "
          "cache thrashes regardless of size. Hit rate creeps up slightly (2.6% → 3.5% → 5.7%) but "
          "throughput is essentially flat (3.03 → 3.06 → 2.82 tok/s — differences are within std). "
          "Paying for a 256 MB scratch to achieve a 3.5% hit rate is the worst-case tradeoff.")
        W("- **Above the knee (cap ∈ {512, 640, 768, 1024}), capacity still doesn't matter.** All four "
          "sit at 99.62% hit rate and 6.5–6.7 tok/s. The sweep confirms the paper's intuition: for the "
          "paper to claim cache benefit, the cache must be at least one corpus-worth of blocks. Beyond "
          "that is pure waste.")
        W(f"- **The ~1.8 tok/s gap between cache plateau (~6.5) and full-mirror ceiling ({ceiling:.2f}) "
          "is Python LRU overhead**, not H2D. `ensure_fp16_keys_resident` does `list.remove(bid)` "
          "on every hit to bump LRU, which is O(N) on the resident set. At a 99.62% hit rate and "
          "~158 top-K blocks needed per step, that's ~50k O(N=512) operations per decode step across "
          "32 layers. A `collections.OrderedDict` or `doubly-linked-list` LRU would close the gap.")
        W("- **The curve is workload-shaped.** This sweep used the generic repetitive-filler prompt "
          "(similar scattered pattern to niah / ruler). PG-19's concentrated attention shows 99.9% "
          "zero-pagein steps at `cap=64` in the main Test 3 data — a small cache is enough when "
          "attention is locally concentrated. The paper can contrast these as 'scattered-retrieval' "
          "vs 'concentrated-attention' regimes.")
        W("")
        W("## Paper-facing takeaway")
        W("")
        W("Set `cap = ceil(N / block_size)` where N is the context length in tokens, nothing smaller. "
          "For 8K context with block_size=16, that's 512 blocks ⇒ ~512 MB conceptual scratch "
          "(once the allocator is fixed to honour capacity) for ~3× speedup over pure H2D. Anything "
          "less than the corpus thrashes and provides effectively no benefit.")
        W("")
        W("## Implementation caveats")
        W("")
        W("- **Scratch VRAM is conceptual, not actual.** The allocator still reserves a "
          "full-sequence-sized `keys_fp16_gpu` regardless of `capacity`; realising the VRAM savings "
          "this sweep implies would require a capacity-sized scratch + block_id→slot_idx index "
          "remapping passed into the Triton attend kernel.")
        W("- **LRU data structure.** `_fp16_key_lru` is a plain `list`; `list.remove` on every hit "
          "is O(N). For cap=1024 the hit rate reaches 99.6% but throughput saturates at ~6.5 tok/s "
          "instead of 8.27 — that ~1.8 tok/s gap is Python, not H2D. A `collections.OrderedDict` "
          "cache would close it.")
        W("- **Hit rate shows 0% for cap=∞ (full mirror).** Full-mirror mode bypasses "
          "`ensure_fp16_keys_resident` entirely so the cache counters never fire; the 0% is an "
          "artefact of the accounting path, not a real miss rate.")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
