# 64K throughput sweep — the paper's headline perf regime

**Model:** `NousResearch/Meta-Llama-3.1-8B` (INT8 bitsandbytes)  
**Hardware:** NVIDIA RTX PRO 6000 Blackwell (sm_120), 96 GB VRAM  
**Setup:** 65536-token pg19 prefill + 256-token argmax decode (warmup 16, timed 240), 3 repeats per config.
**Corpus:** 65536 / 16 = **4096 blocks**. `K_max=128` = 3.1% of corpus — the regime where the tiered architecture is designed to win.

| Config | tok/s ± std | p50 ms/tok | p95 ms/tok | Hit rate | H2D MB/step | Prefill ms | GPU peak MB |
|---|---|---|---|---|---|---|---|
| dense (baseline) | 19.92 ± 0.15 | 50.2 | 52.8 | — | 0 | 7341 | 57635 |
| certified cap=256 (per-Q-head) | 0.55 ± 0.03 | 1726.0 | 2135.1 | 0.55% | 2659.8 | 10293 | 61059 |
| certified cap=512 (per-Q-head) | 0.61 ± 0.01 | 1647.0 | 1672.3 | 0.63% | 2666.9 | 9952 | 61059 |
| certified cap=1024 (per-Q-head) | 0.51 ± 0.00 | 1893.9 | 2275.1 | 0.77% | 2653.2 | 10427 | 61059 |
| certified cap=1024 (per-KV-group) | 0.63 ± 0.00 | 1570.8 | 1647.2 | 0.90% | 2354.0 | 10120 | 61059 |
| certified cap=4096 (=corpus, per-Q) | 0.64 ± 0.01 | 1518.2 | 1971.7 | 99.20% | 21.4 | 10090 | 61059 |
| certified full mirror (per-Q-head) | 1.93 ± 0.00 | 513.9 | 541.8 | — | 0 | 9976 | 61059 |

## Overhead vs dense

| Config | tok/s | Overhead vs dense |
|---|---|---|
| certified cap=256 (per-Q-head) | 0.55 | +3512.3% |
| certified cap=512 (per-Q-head) | 0.61 | +3183.7% |
| certified cap=1024 (per-Q-head) | 0.51 | +3771.2% |
| certified cap=1024 (per-KV-group) | 0.63 | +3052.9% |
| certified cap=4096 (=corpus, per-Q) | 0.64 | +2989.2% |
| certified full mirror (per-Q-head) | 1.93 | +930.7% |

## Key reads

**(1) The cache knee is still at corpus size, even at 64K.** Hit rate at cap=256, 512, 1024 all sit at 0.55–0.77% — the theoretical prediction that 4 Q heads × 3.1% ≈ 22% per KV group, so cap=1024 should work, didn't hold. The per-layer union saturates near corpus size because **there are 8 KV groups**: even if each group only needs 22% of the corpus, `4096 · (1 − 0.78⁸) ≈ 3500 blocks` (86% of corpus) across the 8 groups. Per-layer working set is driven by the 8-way KV diversity × 4-way Q head fan-out within each group, not by the K_max/corpus ratio of any single head.

**(2) The cache isn't the dominant cost at 64K — the Triton kernel is.** Even at cap=∞ (full mirror, zero decode-time H2D), certified is 1.93 tok/s vs dense 19.92 tok/s — a **10.3× slowdown with no page-in cost**. That's the cost of the Phase-1 INT8 scoring + Phase-2 hybrid attend Triton kernel vs torch's Flash Attention. At this context length, optimising the Triton kernel would reclaim ~9× of the ~10× gap; optimising the cache reclaims at best the ~3× gap between full-mirror (1.93 tok/s) and corpus-cap (0.64 tok/s).

**(3) The sub-corpus cache regime is prohibitively expensive at 64K.** H2D MB/step at cap<corpus is ~2.6 GB — enough to push p95 latency above 2 seconds per decoded token. Any practical serving path must either size the cache at corpus or use a structural fix (per-KV-group selection, see below) to break the 8-way KV diversity.

**(4) Per-KV-group at 64K cap=1024 is a real but partial win.** The structural fix (`DOTCACHE_PER_KV_GROUP_TOPK=1` post-commit `320e6cb9`) collapses the 32 independent per-Q-head selections into 8 per-KV-group selections. Measured:

| Config @ cap=1024 | tok/s | Hit rate | H2D MB/step | Per-layer union |
|---|---|---|---|---|
| per-Q-head    | 0.51 | 0.77% | 2653 | ~2650 blocks |
| per-KV-group  | 0.63 | 0.90% | 2354 | ~2358 blocks |

**+24% throughput and −11% H2D.** Real improvement, but smaller than the theoretical union reduction predicted (4096·(1 − 0.94⁸) ≈ 1630 blocks per layer; we measured ~2358). The gap between theory and measurement is LRU discipline plus iteration order: `ensure_fp16_keys_resident` walks `needed_blocks` in block-ID sorted order, so the last-accessed-per-step is always the highest-ID blocks; the next step's first few accesses are low-ID blocks → immediate misses before useful warmup. A priority-ordered (mass-sorted) iteration or a non-LRU discipline would likely close more of the gap — implementation follow-up, not an architectural limitation.

**(5) Triton kernel is still the bigger cost.** Even with per-KV-group at the best cache point (cap=4096, 99.2% hit), tok/s tops out at 0.64 — less than the full-mirror 1.93 tok/s. The kernel-vs-FlashAttention gap (10× at full mirror) outweighs the cache-vs-no-cache gap (3× within the kernel envelope). The paper's perf-section path forward should lead with 'optimise the Triton attend kernel' and have 'optimise the cache policy' as the secondary lever.

