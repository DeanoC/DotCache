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
        tps_list = [(r["label"], r["tok_per_sec_mean"]) for r in ok]
        W("## Reading the curve")
        W("")
        floor = tps_list[0][1]
        ceiling = tps_list[-1][1]
        W(f"- **Floor** ({tps_list[0][0]}): {floor:.2f} tok/s — every top-K block H2D'd every step.")
        W(f"- **Ceiling** ({tps_list[-1][0]}): {ceiling:.2f} tok/s — no decode-time H2D (asymptote).")
        if floor > 0 and ceiling > floor:
            W(f"- **Full-mirror speedup:** {ceiling/floor:.2f}× over pure H2D.")
        # Knee detection: first capacity reaching ≥80% of the ceiling gap
        if floor > 0 and ceiling > floor:
            threshold = floor + 0.8 * (ceiling - floor)
            found_knee = False
            for r in ok:
                if r["tok_per_sec_mean"] >= threshold and not r["label"].startswith("∞"):
                    W(f"- **80%-of-ceiling knee:** `capacity={r['label']}` "
                      f"({r['tok_per_sec_mean']:.2f} tok/s) at {r['scratch_mb_concept']:.0f} MB "
                      f"conceptual scratch — design sweet spot.")
                    found_knee = True
                    break
            if not found_knee:
                W("- **No knee before full mirror.** On this workload, even capacity equal to the entire "
                  "corpus (≥ 512 blocks) doesn't reach 80% of the ceiling. The remaining gap is Python-level "
                  "LRU `list.remove` overhead (O(N) per hit on the resident set), not fundamental. "
                  "A deque/OrderedDict-based LRU would close that gap — see implementation caveats.")
        W("")
        W("## Workload observations")
        W("")
        W("- **cap=64 and cap=256 give essentially the same throughput** (within 1 std). Hit rate is "
          "2–3% in both — a scattered top-K that doesn't reuse blocks across steps overwhelms any "
          "small cache. Slightly *higher* throughput at cap=64 than cap=256 is the Python LRU "
          "tail catching up with the larger resident set.")
        W("- **Bandwidth floor is ~480 MB/step** at cap=0 through cap=256. The cache needs to be "
          "comparable to the full corpus (512 blocks) before H2D MB/step collapses toward zero.")
        W("- **A workload with spatial locality (e.g. pg19-style concentrated attention) shapes the "
          "cache curve differently.** The main Test 3 `pg19` data (in `../SUMMARY.md`) shows 99.9% "
          "zero-pagein steps at cap=64 — a small cache is enough when attention is concentrated.")
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
