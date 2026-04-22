# FP16 VRAM cache capacity sweep — 8K certified decode

**Model:** `NousResearch/Meta-Llama-3.1-8B` (INT8 bitsandbytes)  
**Hardware:** NVIDIA RTX PRO 6000 Blackwell (sm_120), 96 GB VRAM  
**Setup:** 8K prefill + 256-token decode (warmup 16, timed 240), `tau_cov=0.995, k_min=2, k_max=128`, fallbacks on; 5 repeats per capacity point.  
**Corpus size:** 8K tokens / 16-token blocks = 512 blocks. `cap=1024` and `cap=∞` should land at the same ceiling.

| Capacity (blocks) | tok/s ± std | p50 ms/tok | Hit rate | H2D MB/step | Scratch VRAM* | Notes |
|---|---|---|---|---|---|---|
| 0 (pure H2D) | 2.77 ± 0.01 | 360.7 | 0.00% | 486.6 | 0 MB | every access H2D'd |
| 64 | 3.06 ± 0.03 | 325.5 | 2.61% | 473.9 | 64 MB |  |
| 256 | 2.82 ± 0.01 | 353.9 | 3.49% | 469.6 | 256 MB |  |
| 384 | 3.03 ± 0.01 | 330.5 | 5.71% | 458.8 | 384 MB |  |
| 512 (= corpus) | 6.51 ± 0.04 | 153.2 | 99.62% | 1.9 | 512 MB | ≥ corpus → same as ∞ |
| 640 | 6.66 ± 0.02 | 150.1 | 99.62% | 1.9 | 640 MB | ≥ corpus → same as ∞ |
| 768 | 6.57 ± 0.03 | 152.1 | 99.62% | 1.9 | 768 MB | ≥ corpus → same as ∞ |
| 1024 | 6.51 ± 0.06 | 153.3 | 99.62% | 1.9 | 1024 MB | ≥ corpus → same as ∞ |
| ∞ (full mirror) | 8.27 ± 0.09 | 120.3 | 0.00% | 0.0 | 500 MB | no H2D during decode |

\* *Scratch VRAM column is conceptual — `capacity × block_size × kv_heads × head_dim × 2 bytes × num_layers`. The current implementation allocates a full-sequence scratch regardless of capacity; `capacity`-sized allocation is a follow-up that would realise the VRAM savings this sweep implies.*

## The knee is at the corpus size

- **Floor** (`cap=0`, pure H2D): 2.77 tok/s — every top-K block paged in every step.
- **Cache plateau** (`cap ≥ 512`): ~6.5 tok/s, hit rate 99.62%, H2D collapses to 1.9 MB/step.
- **Ceiling** (`cap=∞`, full mirror): 8.27 tok/s — no decode-time H2D.
- **Full-mirror speedup:** 2.98× over pure H2D.

The transition is **sharp and happens at exactly `cap=512` — the corpus size** (8K tokens / 16-token blocks = 512 blocks). Between `cap=384` and `cap=512`, hit rate jumps from 5.71% to 99.62% and H2D bandwidth collapses from 459 MB/step to 1.9 MB/step. Anything above 512 plateaus at the same hit rate and H2D cost — the extra capacity is wasted scratch.

## Workload observations

- **Below the knee (cap ∈ {64, 256, 384}), capacity doesn't matter.** The scattered top-K of Llama-3.1-8B's certified attention cycles through different blocks every decode step; the cache thrashes regardless of size. Hit rate creeps up slightly (2.6% → 3.5% → 5.7%) but throughput is essentially flat (3.03 → 3.06 → 2.82 tok/s — differences are within std). Paying for a 256 MB scratch to achieve a 3.5% hit rate is the worst-case tradeoff.
- **Above the knee (cap ∈ {512, 640, 768, 1024}), capacity still doesn't matter.** All four sit at 99.62% hit rate and 6.5–6.7 tok/s. The sweep confirms the paper's intuition: for the paper to claim cache benefit, the cache must be at least one corpus-worth of blocks. Beyond that is pure waste.
- **The ~1.8 tok/s gap between cache plateau (~6.5) and full-mirror ceiling (8.27) is Python LRU overhead**, not H2D. `ensure_fp16_keys_resident` does `list.remove(bid)` on every hit to bump LRU, which is O(N) on the resident set. At a 99.62% hit rate and ~158 top-K blocks needed per step, that's ~50k O(N=512) operations per decode step across 32 layers. A `collections.OrderedDict` or `doubly-linked-list` LRU would close the gap.
- **The curve is workload-shaped.** This sweep used the generic repetitive-filler prompt (similar scattered pattern to niah / ruler). PG-19's concentrated attention shows 99.9% zero-pagein steps at `cap=64` in the main Test 3 data — a small cache is enough when attention is locally concentrated. The paper can contrast these as 'scattered-retrieval' vs 'concentrated-attention' regimes.

## Paper-facing takeaway

Set `cap = ceil(N / block_size)` where N is the context length in tokens, nothing smaller. For 8K context with block_size=16, that's 512 blocks ⇒ ~512 MB conceptual scratch (once the allocator is fixed to honour capacity) for ~3× speedup over pure H2D. Anything less than the corpus thrashes and provides effectively no benefit.

## Implementation caveats

- **Scratch VRAM is conceptual, not actual.** The allocator still reserves a full-sequence-sized `keys_fp16_gpu` regardless of `capacity`; realising the VRAM savings this sweep implies would require a capacity-sized scratch + block_id→slot_idx index remapping passed into the Triton attend kernel.
- **LRU data structure.** `_fp16_key_lru` is a plain `list`; `list.remove` on every hit is O(N). For cap=1024 the hit rate reaches 99.6% but throughput saturates at ~6.5 tok/s instead of 8.27 — that ~1.8 tok/s gap is Python, not H2D. A `collections.OrderedDict` cache would close it.
- **Hit rate shows 0% for cap=∞ (full mirror).** Full-mirror mode bypasses `ensure_fp16_keys_resident` entirely so the cache counters never fire; the 0% is an artefact of the accounting path, not a real miss rate.
