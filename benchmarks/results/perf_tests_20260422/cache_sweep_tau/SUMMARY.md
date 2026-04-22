# τ_cov × selection-mode sweep at 8K (pg19 prefill, cap=64)

**Question.** Does tuning tau_cov or collapsing Q heads into KV-head groups materially shrink the per-layer FP16 working set, and thus make a small cache useful?

**Setup.** Same as the main perf Test 1: 8K pg19 prefill, 256-token argmax decode (16 warmup, 240 timed), cap=64, 5 repeats per config.

| τ_cov | Selection mode | tok/s ± std | p50 ms | Hit rate | Misses / step | Per-layer union avg | H2D MB/step |
|---|---|---|---|---|---|---|---|
| 0.995 | per-Q-head | 2.76 ± 0.03 | 362.8 | 2.18% | 16810 | 137476 | 525.3 |
| 0.990 | per-Q-head | 2.86 ± 0.02 | 348.8 | 2.19% | 16807 | 137467 | 525.2 |
| 0.980 | per-Q-head | 2.80 ± 0.02 | 355.0 | 2.24% | 16564 | 135548 | 517.6 |
| 0.950 | per-Q-head | 2.97 ± 0.08 | 343.1 | 2.36% | 15677 | 128448 | 489.9 |
| 0.995 | per-KV-group (8 groups) | 2.60 ± 0.06 | 384.0 | 2.24% | 16769 | 137218 | 524.0 |

## Findings

**Neither tau_cov tuning nor per-KV-group selection materially shrinks the per-layer working set at 8K context.** The measured numbers:

- τ_cov 0.995 → 0.95 lifts hit rate only from 2.18% → 2.36%, tok/s from 2.76 → 2.97. Marginal, not a knee.
- Per-KV-group (8 groups instead of 32 Q heads, τ=0.995) lands at 2.60 tok/s / 2.24% hit rate — slightly *worse* than per-Q-head due to the extra Python-level aggregation it introduces, with no H2D savings.
- Misses per step sit in the 15,700–16,800 range across all five configs (per-layer ≈ 490–525 of 512 corpus blocks — ~96% of corpus per layer).

**Why the levers don't help at 8K.** The 8K corpus is 512 blocks; K_max=128 is 25% of corpus per Q head. The union across 4 Q heads of a KV group (independence model): 512·(1 − 0.75⁴) = 350 blocks (68% of corpus). The union across 8 KV groups at that level saturates near the full corpus. Lowering τ_cov shrinks each head's K* by a couple of percent, and per-KV-group collapses 32→8 independent picks, but neither intervenes against the dominant bound: at this context length, the corpus is just too small for a sub-corpus cache to make sense under any τ_cov/selection regime.

## Caveat — this is still 8K

8K is the wrong regime to showcase the tiered architecture. At 8K the K_max/corpus ratio is 128/512 = 25% per Q head, so the GQA union is fundamentally bounded below anyway. The real test is at longer contexts where K_max/corpus becomes small (64K: 3.1%, 128K: 1.6%) and per-layer unions can realistically stay under a small cache. See `cache_sweep_64k/SUMMARY.md` for the load-bearing throughput story.
