# 64K throughput sweep — the paper's headline perf regime

**Model:** `NousResearch/Meta-Llama-3.1-8B` (INT8 bitsandbytes)  
**Hardware:** NVIDIA RTX PRO 6000 Blackwell (sm_120), 96 GB VRAM  
**Setup:** 65536-token pg19 prefill + 256-token argmax decode (warmup 16, timed 240), 3 repeats per config.
**Corpus:** 65536 / 16 = **4096 blocks**. `K_max=128` = 3.1% of corpus — the regime where the tiered architecture is designed to win.

| Config | tok/s ± std | p50 ms/tok | p95 ms/tok | Hit rate | H2D MB/step | Prefill ms | GPU peak MB |
|---|---|---|---|---|---|---|---|
| dense (baseline) | 19.92 ± 0.15 | 50.2 | 52.8 | — | 0 | 7341 | 57635 |
| certified cap=256 | 0.55 ± 0.03 | 1726.0 | 2135.1 | 0.55% | 2659.8 | 10293 | 61059 |
| certified cap=512 | 0.61 ± 0.01 | 1647.0 | 1672.3 | 0.63% | 2666.9 | 9952 | 61059 |
| certified cap=1024 | 0.51 ± 0.00 | 1893.9 | 2275.1 | 0.77% | 2653.2 | 10427 | 61059 |
| certified cap=4096 (=corpus) | 0.64 ± 0.01 | 1518.2 | 1971.7 | 99.20% | 21.4 | 10090 | 61059 |
| certified full mirror | 1.93 ± 0.00 | 513.9 | 541.8 | — | 0 | 9976 | 61059 |

## Overhead vs dense

| Config | tok/s | Overhead vs dense |
|---|---|---|
| certified cap=256 | 0.55 | +3512.3% |
| certified cap=512 | 0.61 | +3183.7% |
| certified cap=1024 | 0.51 | +3771.2% |
| certified cap=4096 (=corpus) | 0.64 | +2989.2% |
| certified full mirror | 1.93 | +930.7% |

## Key reads

**(1) The cache knee is still at corpus size, even at 64K.** Hit rate at cap=256, 512, 1024 all sit at 0.55–0.77% — the theoretical prediction that 4 Q heads × 3.1% ≈ 22% per KV group, so cap=1024 should work, didn't hold. The per-layer union saturates near corpus size because **there are 8 KV groups**: even if each group only needs 22% of the corpus, `4096 · (1 − 0.78⁸) ≈ 3500 blocks` (86% of corpus) across the 8 groups. Per-layer working set is driven by the 8-way KV diversity × 4-way Q head fan-out within each group, not by the K_max/corpus ratio of any single head.

**(2) The cache isn't the dominant cost at 64K — the Triton kernel is.** Even at cap=∞ (full mirror, zero decode-time H2D), certified is 1.93 tok/s vs dense 19.92 tok/s — a **10.3× slowdown with no page-in cost**. That's the cost of the Phase-1 INT8 scoring + Phase-2 hybrid attend Triton kernel vs torch's Flash Attention. At this context length, optimising the Triton kernel would reclaim ~9× of the ~10× gap; optimising the cache reclaims at best the ~3× gap between full-mirror (1.93 tok/s) and corpus-cap (0.64 tok/s).

**(3) The sub-corpus cache regime is prohibitively expensive at 64K.** H2D MB/step at cap<corpus is ~2.6 GB — enough to push p95 latency above 2 seconds per decoded token. Any practical serving path must either size the cache at corpus or use a structural fix (per-KV-group selection, see below) to break the 8-way KV diversity.

**(4) The per-KV-group structural fix deserves a 64K test.** At 8K it didn't help because K_max/corpus was already 25% — saturating the union anyway. At 64K with K_max=128 (3.1%), collapsing 32 Q heads into 8 groups reduces the total independent-selections count from 32 to 8, halving the upper bound on per-layer union. See `cache_sweep_tau/SUMMARY.md` for the 8K negative result; a 64K re-test is the remaining experiment before the paper has to commit to 'cache-sized scratch' as the deployment recommendation.

