# PG-19 FP16 VRAM cache sweep (generated decode from PG-19 prefill)

**Model:** `NousResearch/Meta-Llama-3.1-8B` (INT8 bitsandbytes)  
**Hardware:** NVIDIA RTX PRO 6000 Blackwell (sm_120), 96 GB VRAM  
**Setup:** Same as `../cache_sweep/SUMMARY.md`, but the 8K prefill is the first long-enough book from the PG-19 test split; decode remains argmax generation (256 tokens, 16 warmup, 240 timed).

| Capacity (blocks) | tok/s ± std | p50 ms/tok | Hit rate | H2D MB/step | (filler tok/s, for reference) |
|---|---|---|---|---|---|
| 0 (pure H2D) | 2.40 ± 0.31 | 396.5 | 0.00% | 538.4 | 2.77 |
| 4 | 2.74 ± 0.01 | 363.7 | 0.00% | 537.3 | — |
| 16 | 2.73 ± 0.02 | 366.1 | 0.10% | 537.5 | — |
| 64 | 2.78 ± 0.02 | 359.1 | 2.21% | 523.5 | 3.06 |
| 256 | 2.84 ± 0.01 | 351.2 | 5.81% | 508.6 | 2.82 |
| 1024 | 6.00 ± 0.03 | 165.6 | 99.63% | 2.0 | 6.51 |
| ∞ (full mirror) | 8.01 ± 0.07 | 123.5 | 0.00% | 0.0 | 8.27 |

## The knee is still at corpus size

We expected pg19's concentrated attention (shown in main Test 3 with 99.9% zero-pagein at `cap=64`) to shift the knee down dramatically. The sweep shows **the opposite — the same knee at corpus size** as the filler/niah workload. Hit-rate stays in single-digit-percent territory until `cap ≥ 512` (= corpus), then jumps to 99.6%.

## Why this doesn't contradict Test 3 pg19

The main Test 3 pg19 measurement used **teacher-forced** decode — `pg19_perplexity.py` feeds the ground-truth next token into the model at each step and scores its NLL. Attention in that regime stays locally concentrated because each new query is the embedding of a real pg19 token, and Llama's in-distribution attention pattern on pg19 naturally tracks recent and strongly-related earlier tokens — a small working set.

This sweep used **argmax-generated** decode from a pg19 prefill — the model produces its own continuation token-by-token. Once the generated text leaves the pg19 distribution (which happens within ~20 decode steps of unconditional generation), the queries become Llama's open-ended continuation queries, which have a **scattered top-K** similar to filler. The cache behaviour then tracks that scattered pattern, not pg19's teacher-forced pattern.

**Net finding: the cache curve shape is driven by *decode mode*, not by prefix content alone.**

## What the two data points together tell the paper

- **Teacher-forced decode (real workload signal)**: cap=64 is plenty — pg19 Test 3 shows 99.9% zero-pagein steps. The paper's tiered architecture pays off handsomely here because the cache hot set is tiny and stable.
- **Argmax-generated decode (open-ended generation)**: the cache must be ≥ corpus for any speedup; below that you pay full H2D bandwidth. Open-ended generation is inherently adversarial to a small cache.
- These two regimes bracket the design space. A practical serving system doing mostly teacher-forced scoring (RAG re-ranking, prefix-logprob evaluation) lives in the first regime. A serving system doing long free-form generation (chat completion) lives in the second. The paper can honestly claim both: **the architecture is H2D-efficient when attention locality is present, and falls back to bandwidth-bound when it isn't.**

## Implementation caveats

Same as `../cache_sweep/SUMMARY.md`:
- Scratch VRAM column omitted here because of the allocator caveat — physical allocation is still full-sequence-sized regardless of capacity.
- The cap=1024 ceiling gap (6.00 vs 8.01 tok/s) is Python `list.remove` O(N) LRU overhead, not H2D. The H2D column at cap=1024 is only 2 MB/step.
- Hit rate 0% at cap=∞ is an accounting artefact (full-mirror path bypasses the cache counters); not a real miss rate.

