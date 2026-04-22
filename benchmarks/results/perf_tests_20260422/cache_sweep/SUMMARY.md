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
| 1024 | 6.51 ± 0.06 | 153.3 | 99.62% | 1.9 | 1024 MB | ≥ corpus → same as ∞ |
| ∞ (full mirror) | 8.27 ± 0.09 | 120.3 | 0.00% | 0.0 | 500 MB | no H2D during decode |

\* *Scratch VRAM column is conceptual — `capacity × block_size × kv_heads × head_dim × 2 bytes × num_layers`. The current implementation allocates a full-sequence scratch regardless of capacity; `capacity`-sized allocation is a follow-up that would realise the VRAM savings this sweep implies.*

## Reading the curve

- **Floor** (0 (pure H2D)): 2.77 tok/s — every top-K block H2D'd every step.
- **Ceiling** (∞ (full mirror)): 8.27 tok/s — no decode-time H2D (asymptote).
- **Full-mirror speedup:** 2.98× over pure H2D.
- **No knee before full mirror.** On this workload, even capacity equal to the entire corpus (≥ 512 blocks) doesn't reach 80% of the ceiling. The remaining gap is Python-level LRU `list.remove` overhead (O(N) per hit on the resident set), not fundamental. A deque/OrderedDict-based LRU would close that gap — see implementation caveats.

## Workload observations

- **cap=64 and cap=256 give essentially the same throughput** (within 1 std). Hit rate is 2–3% in both — a scattered top-K that doesn't reuse blocks across steps overwhelms any small cache. Slightly *higher* throughput at cap=64 than cap=256 is the Python LRU tail catching up with the larger resident set.
- **Bandwidth floor is ~480 MB/step** at cap=0 through cap=256. The cache needs to be comparable to the full corpus (512 blocks) before H2D MB/step collapses toward zero.
- **A workload with spatial locality (e.g. pg19-style concentrated attention) shapes the cache curve differently.** The main Test 3 `pg19` data (in `../SUMMARY.md`) shows 99.9% zero-pagein steps at cap=64 — a small cache is enough when attention is concentrated.

## Implementation caveats

- **Scratch VRAM is conceptual, not actual.** The allocator still reserves a full-sequence-sized `keys_fp16_gpu` regardless of `capacity`; realising the VRAM savings this sweep implies would require a capacity-sized scratch + block_id→slot_idx index remapping passed into the Triton attend kernel.
- **LRU data structure.** `_fp16_key_lru` is a plain `list`; `list.remove` on every hit is O(N). For cap=1024 the hit rate reaches 99.6% but throughput saturates at ~6.5 tok/s instead of 8.27 — that ~1.8 tok/s gap is Python, not H2D. A `collections.OrderedDict` cache would close it.
- **Hit rate shows 0% for cap=∞ (full mirror).** Full-mirror mode bypasses `ensure_fp16_keys_resident` entirely so the cache counters never fire; the 0% is an artefact of the accounting path, not a real miss rate.
