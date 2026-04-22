# Performance benchmarks — SUMMARY (2026-04-22)

**Model:** `NousResearch/Meta-Llama-3.1-8B` (INT8 bitsandbytes)  
**Hardware:** NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120), 96 GB VRAM  
**Context length:** 8192 tokens  
**Certified config:** `tau_cov=0.995, k_min=2, k_max=128`, ranking_fallback on, score_consistency_check on (eps_guard=0.01), exploration_rate=0.02, v_tolerance=0.05, Rungs 1–4 all wired (post-`79d1a0da` Δ-bound fix).

## Test 1 — Decode throughput

| Config | tok/s mean | ± std | p50 ms | p95 ms | p99 ms | Overhead vs dense |
|---|---|---|---|---|---|---|
| `dense` | 27.74 | 0.18 | 35.99 | 38.49 | 39.64 | — |
| `certified` | 8.25 | 0.06 | 120.50 | 125.57 | 132.95 | +236.2% |
| `certified-no-fallback` | 9.53 | 0.05 | 104.41 | 108.55 | 109.77 | +190.9% |
| `quantised-only` | 15.68 | 0.12 | 63.70 | 67.63 | 69.74 | +77.0% |

**Net user-visible overhead (certified vs dense):** 236.2%. dense = 27.74 tok/s, certified = 8.25 tok/s.
**Fallback-monitor cost:** 15.6% additional throughput loss going from certified-no-fallback (9.53 tok/s) to full certified (8.25 tok/s).

## Test 2 — Per-step latency breakdown (certified)

| Phase | Mean μs | p50 μs | p95 μs | Share of step |
|---|---|---|---|---|
| `phase1_int8_scoring` | 4206.4 | 4134.4 | 4953.1 | 3.1% |
| `adaptive_selection` | 5892.5 | 5855.2 | 6254.1 | 4.4% |
| `ranking_check` | 5285.2 | 5244.9 | 5670.3 | 3.9% |
| `h2d_pagein` | 0.0 | 0.0 | 0.0 | 0.0% |
| `value_check` | 0.0 | 0.0 | 0.0 | 0.0% |
| `phase2_fused_attend` | 48789.3 | 48794.7 | 50058.0 | 36.2% |
| `overhead_other` | 70624.7 | 70386.7 | 74985.7 | 52.4% |

**Total step:** mean 134.80 ms, p50 134.46 ms, p95 139.86 ms, p99 141.22 ms  
*(Measured with `phase_timings` active — ~5 extra GPU syncs/layer/step, so total step time here overstates Test 1's tok/s. Phase ratios are the meaningful output.)*

## Test 3 — H2D page-in and VRAM-resident cache telemetry

| Benchmark | n steps | MB/tok mean | p50 | p95 | max | % zero-pagein | VRAM key cache | VRAM value cache |
|---|---|---|---|---|---|---|---|---|
| pg19 | 1638 | 0.000 | 0.000 | 0.000 | 0.000 | 100.0% | 409.62 MB | 409.62 MB |
| niah | 1121 | 0.000 | 0.000 | 0.000 | 0.000 | 100.0% | 500.19 MB | 500.19 MB |
| ruler | 4058 | 0.000 | 0.000 | 0.000 | 0.000 | 100.0% | 501.56 MB | 501.56 MB |

| Benchmark | Rung-1 rate | Rung-2 rate | Rung-3 rate | Rung-4 rate | K* mean | K* max | RSS peak | /proc/meminfo Cached Δ |
|---|---|---|---|---|---|---|---|---|
| pg19 | 0.06% | 0.00% | 0.06% | 0.00% | 180.9 | 256 | 4293 MB | 1233 MB |
| niah | 100.00% | 0.00% | 66.28% | 0.00% | 159.7 | 256 | 3277 MB | 1770 MB |
| ruler | 100.00% | 0.00% | 71.86% | 0.00% | 140.6 | 256 | 3986 MB | 2101 MB |

### Quality cross-check (Test 3 piggyback)

| Benchmark | Dense | Certified | Δ |
|---|---|---|---|
| pg19 ppl | 6.4118 | 6.4025 | -0.0093 |
| niah acc | 0.9333 | 0.9000 | -0.0333 |
| ruler acc | 0.9193 | 0.9236 | +0.0043 |

## Key findings

- **Rung 4 never fires** on any of the three benchmarks. After the `79d1a0da` Δ-bound fix the score-consistency canary is both calibrated and zero-firing — Theorem 2 holds empirically with ample headroom.
- **H2D transfer during decode is essentially zero** under the default tiered configuration. The VRAM-resident FP16 mirror (`keys_fp16_gpu`, `values_fp16`) covers 100% of hot accesses; only Rung-2 value escalations would incur a page-in, and they don't fire on this workload.

## Caveats

- Test 1 `triton-fp16` config is not implemented in this codebase — Phase 1 bypass would require a new adapter path. The 4-config matrix (dense / certified / certified-no-fallback / quantised-only) still decomposes net overhead into monitoring vs. kernel cost.
- Test 2 total-step time is inflated by the phase timers' GPU syncs; the per-phase ratios are reliable but the absolute latency should be taken from Test 1.
- Test 3 sample counts are scaled down from the arXiv v1 sweep — enough for telemetry convergence (~1-2k decode steps per bench) without repeating full quality runs at 8K.
