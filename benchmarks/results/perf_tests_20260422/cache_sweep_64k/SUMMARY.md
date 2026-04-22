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

## Attend-kernel optimisation (split-K, 2026-04-22)

**Diagnosis** (from `test2_phase_breakdown_64k.json` at cap=∞): phase2 Triton attend took 560 ms of a 676 ms step — 83% share. The original `selective_attend_multihead_hybrid` launched only `num_q_heads = 32` programs (17% SM occupancy on 188-SM Blackwell) with FP64 softmax state (consumer Blackwell FP64 is ~1/32 FP32 rate).

**Fix:** new `selective_attend_multihead_hybrid_split_k` kernel in `dotcache/kernels/selective_attend_triton.py`:
- Partitions the block axis across `num_splits` programs per Q head (FlashDecoding-style). Grid expands from 32 to `32 × num_splits` (e.g., 512 at num_splits=16).
- Per-split partials (m_i, l_i, acc_i) merged by a small reduction kernel using standard online-softmax recombination: `m* = max m_i`; `out = Σ exp(m_i - m*) · acc_i / Σ exp(m_i - m*) · l_i`.
- FP32 state throughout (FlashAttention-style). Empty splits store `m = -inf`, which reduces to zero contribution via `exp(-inf) = 0`.

Behind `DOTCACHE_FAST_ATTEND=1` (default on). `DOTCACHE_FAST_ATTEND=0` reverts to the original kernel for A/B.

**Micro-bench** (`benchmarks/bench_hybrid_attend_kernel.py`, 64K synthetic inputs, isolated kernel):

| Kernel | ms/launch | grid | vs SDPA dense FP16 |
|---|---:|---:|---:|
| original hybrid            | 13.4 | 32   | 8.0× slower |
| split-K num_splits=1 (FP32 only) | 8.2 | 32 | 4.9× slower |
| split-K num_splits=16       | 0.81 | 512  | 0.5× — i.e. **faster** than SDPA |
| split-K num_splits=64       | 0.70 | 2048 | 0.4× |
| SDPA dense FP16 (reference) | 1.68 | —    | — |

The FP64→FP32 swap alone gave 1.6×; parallelism gave the remaining ~12×. At ns=16 we beat SDPA by avoiding the GQA key repeat-interleave the SDPA reference needs. Default autotune = `num_splits = round_up_pow2(num_blocks / 256)`.

**End-to-end 64K breakdown** (`test2_phase_breakdown_64k_fast.json`):

| Phase | before (ms) | after (ms) | Δ |
|---|---:|---:|---:|
| phase1_int8_scoring | 26.4 | 15.7 | −41% |
| adaptive_selection  |  6.2 |  6.6 |  — |
| ranking_check       |  5.5 |  5.8 |  — |
| phase2_fused_attend | **560.4** | **192.3** | **−66%** |
| overhead_other      | 77.4 | 79.2 |  — |
| **Total step (timer-on)** | **675.9** | **299.4** | **−56%** |
| tok/s (timer-on)    | 1.48 | 3.34 | **2.26×** |

**Correctness:** `benchmarks/check_split_k_equivalence.py` at 8K/32-steps → 33/33 tokens identical between the two kernels. Max absolute error on synthetic inputs: 0.000000; max relative error: 0.001 (reduction-order FP32 rounding, sub-argmax-threshold in practice).

**Remaining gap:** the attend phase is still 192 ms vs the ~22 ms pure-kernel floor from the micro-bench (0.7 ms × 32 layers). The extra ~170 ms is Python wrapping — `torch.zeros(no_skip)` per layer, `adaptive_topk_mask.to(int32).contiguous()` per layer, and duplicate `.contiguous()` guards on cache tensors. Next lever; not blocking.

**Paper-relevant number:** the paper's instrumented-off full-mirror tok/s was 1.93. Applying the timer-overhead ratio (1.93 / 1.48 = 1.30×) to the new instrumented reading gives an estimated **~4.4 tok/s** at 64K full-mirror — close to the gap to dense (19.9) being 4.5× rather than 10×.

## Timer-off measurement (2026-04-22, updated)

The phase-timer's `torch.cuda.synchronize()` on every `_PhaseTimer.__exit__`
adds ~180 ms/step at 64K (76% relative inflation), serialising what should
pipeline across layers. `benchmarks/bench_decode_64k_no_timer.py` removes
all phase timing and reports the production tok/s the paper would measure.

| Config | mean ms/step | p50 | p95 | tok/s | vs dense |
|---|---:|---:|---:|---:|---:|
| dense (baseline) | 50.84 | 50.45 | 54.12 | **19.67** | 1.00× |
| cert FAST_ATTEND=0 (original hybrid) | 482.20 | 480.69 | 488.73 | 2.07 | 9.48× |
| **cert FAST_ATTEND=1 (split-K)** | **125.63** | **123.24** | **132.46** | **7.96** | **2.47×** |

End-to-end throughput improvement from the kernel swap alone: **3.84×**
(cert-slow → cert-fast). Gap to dense collapsed from **9.5×** to **2.47×**.

## Priority-ordered LRU + O(1) OrderedDict (2026-04-22)

Two changes to `TieredKeyCacheLayer._fp16_key_resident` and the
bounded-cache call site:

1. Replaced the list-based LRU (O(n) `.remove()` + `.insert(0)` on every
   hit) with an `OrderedDict` (O(1) `move_to_end` + `popitem(last=False)`).
   At cap=1024 with ~2400 hits per step, the list version was doing
   ~2.5M Python ops/step of LRU bookkeeping.
2. Sort `needed_blocks` ASCENDING by max m_b across heads before
   iteration in `ensure_fp16_keys_resident`. Since the cache is
   insert-MRU-last, high-score blocks now land at the MRU-tail and
   survive longer; low-score blocks land near the LRU-front and are
   evicted first.

### Measured at 64K cap=1024 per-KV-group (no phase timer)

| Config | mean ms/step | tok/s | vs original |
|---|---:|---:|---:|
| Original (old kernel + list-LRU, block-ID order) | ~1570 | **0.63** | 1.00× |
| Old kernel + priority OrderedDict-LRU | 1520 | **0.66** | 1.05× |
| Split-K kernel + priority OrderedDict-LRU | **1165** | **0.86** | **1.36×** |

Priority LRU alone contributed +5%; split-K kernel +30% on top. The
priority-LRU win is smaller than the theoretical union-reduction
ceiling (~30%) because at cap=1024 the H2D cost is dominated by the
per-step working-set size (union of top-K across 8 KV groups), not by
which specific block is the LRU victim.

### Full-mirror regression check (cap=∞)

After the changes: 123.55 ms/step → **8.09 tok/s** (2.47× dense).
Previous 125.63 ms/step → 7.96 tok/s. Slight improvement attributed to
the O(1) init and MRU-bump replacing the O(n) list ops. No regression.

## Drop call-site .contiguous() copies (2026-04-22, final)

`torch.profiler` on a cert decode step at 64K showed **`aten::copy_` as
the #1 self-CUDA consumer at ~25 ms/step (31.85% of total)**, dominantly
the four per-layer `.contiguous()` calls (`keys_int8`, `keys_scale`,
`keys_fp16`, `values_fp16`). The stride-aware split-K kernel already
handles non-contig slices via per-KV-head stride args.

Dropping the copies:

| Config at 64K cap=∞ | mean ms/step | tok/s | vs dense |
|---|---:|---:|---:|
| dense baseline | 50.90 | 19.65 | 1.00× |
| cert FAST (with .contiguous()) | 123.55 | 8.09 | 2.47× |
| cert FAST (no .contiguous()) | **105.09** | **9.52** | **2.06×** |
| cert SLOW (original kernel) | 477.96 | 2.09 | 9.39× |

**+18% tok/s end-to-end** from eliminating the copies. The previous
"load-bearing" measurement of 264 ms was an artifact of an intermediate
state; with the OrderedDict LRU, priority ordering, and stride-aware
kernel all in place, the non-contig path is the clear win.

## Full summary — starting from 9.48× dense → landed at 2.06× dense

| Change | Step ms | tok/s | vs dense |
|---|---:|---:|---:|
| Baseline (original kernel) | 477.96 | 2.09 | 9.39× |
| + split-K kernel (FP32 state, FlashDecoding partition) | 125.63 | 7.96 | 2.47× |
| + OrderedDict LRU + priority-ordered iteration | 123.55 | 8.09 | 2.47× |
| + drop call-site .contiguous() copies | **105.09** | **9.52** | **2.06×** |

End-to-end throughput improvement from the PR: **4.55× faster decode**
at 64K cap=∞ full-mirror (2.09 → 9.52 tok/s). Gap to dense collapsed
from 9.39× to 2.06× — within a few percent of the user's 2× dense target.

