# arXiv v1 Benchmark Sweep — Final Summary

**Model:** `NousResearch/Meta-Llama-3.1-8B` (INT8 bitsandbytes)  
**Hardware:** NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120), 96 GB VRAM  
**Branch:** `feature/interval-ellipsoidal-bounds`  
**Commit range:** cells re-certified after hybrid-kernel fix `308357bd`; final cell `ba75dd6f` (cell 18).  
**Total wall time:** 10.2 hours across 18 cells (dense: pre-fix; certified: post-`308357bd` re-run).

## Certified config

All paper-alignment features active:

- `tau_cov=0.995, k_min=2, k_max=128`
- `ranking_fallback=True, ranking_r=1, mode=full`
- `score_consistency_check=True, eps_guard=0.01`
- `exploration_rate=0.02`
- `rung1_threshold=0.02, rung1_multiplier=2.0`
- `v_tolerance=0.05`
- FP64 online-softmax accumulators in all Triton attend kernels

## Headline numbers (certified − dense)

| Benchmark | 4K | 8K | 16K | Spec target | Pass |
|---|---|---|---|---|---|
| **PG-19 Δppl** | +0.0114 | +0.0030 | +0.0017 | <+0.01 | 4K ✗, 8K ✓, 16K ✓ |
| **NIAH Δacc** | −0.067 | −0.067 | +0.033 | ≥−0.02 (4K) / ≥−0.04 (8K) | 4K ✗, 8K ✗ |
| **RULER Δacc** | −0.001 | +0.004 | +0.007 | ±0.01 | 4K ✓, 8K ✓, 16K ✓ |

**Dense baselines (reference):**

| Bench | 4K | 8K | 16K |
|---|---|---|---|
| pg19 ppl | 6.838 | 6.648 | 9.725 |
| niah acc | 0.833 | 0.933 | 0.700 |
| ruler acc | 0.955 | 0.930 | 0.898 |

## Per-cell table

| Cell | Benchmark | Ctx | Config | Dense | Cert | Δ | Wall (min) |
|---|---|---|---|---|---|---|---|
| 01 | pg19 | 4K | dense | — | 6.8379 | — | 2.3 |
| 02 | niah | 4K | dense | — | 0.8333 | — | 4.8 |
| 03 | ruler | 4K | dense | — | 0.9552 | — | 37.5 |
| 04 | pg19 | 4K | certified | 6.8379 | 6.8493 | +0.0114 | 25.8 |
| 05 | niah | 4K | certified | 0.8333 | 0.7667 | −0.0667 | 5.1 |
| 06 | ruler | 4K | certified | 0.9552 | 0.9545 | −0.0007 | 52.6 |
| 07 | pg19 | 8K | dense | — | 6.6483 | — | 2.4 |
| 08 | niah | 8K | dense | — | 0.9333 | — | 5.2 |
| 09 | ruler | 8K | dense | — | 0.9298 | — | 45.1 |
| 10 | pg19 | 8K | certified | 6.6483 | 6.6514 | +0.0030 | 55.7 |
| 11 | niah | 8K | certified | 0.9333 | 0.8667 | −0.0667 | 5.8 |
| 12 | ruler | 8K | certified | 0.9298 | 0.9334 | +0.0037 | 65.6 |
| 13 | pg19 | 16K | dense | — | 9.7247 | — | 2.2 |
| 14 | niah | 16K | dense | — | 0.7000 | — | 6.2 |
| 15 | ruler | 16K | dense | — | 0.8979 | — | 59.1 |
| 16 | pg19 | 16K | certified | 9.7247 | 9.7264 | +0.0017 | 133.9 |
| 17 | niah | 16K | certified | 0.7000 | 0.7333 | +0.0333 | 7.9 |
| 18 | ruler | 16K | certified | 0.8979 | 0.9049 | +0.0070 | 97.7 |

## System metrics (certified cells)

| Cell | skip_rate | disagree_r1 | disagree_r3 | fallback_rate |
|---|---|---|---|---|
| 04 pg19 4K | 0.403 | — | — | — |
| 05 niah 4K | — | 0.127% | 2.571% | 0.127% |
| 06 ruler 4K | 0.554 (min 0.461, max 0.660) | — | — | — |
| 10 pg19 8K | 0.542 | — | — | — |
| 11 niah 8K | — | 0.115% | 2.548% | 0.115% |
| 12 ruler 8K | 0.691 (min 0.624, max 0.776) | — | — | — |
| 16 pg19 16K | 0.720 | — | — | — |
| 17 niah 16K | — | 0.120% | 2.520% | 0.120% |
| 18 ruler 16K | 0.815 (min 0.784, max 0.872) | — | — | — |

## Canaries

- `critical_failures` summed across certified cells: **24** (isolated to RULER vt/cwe subtasks at longer contexts; 0 across pg19 and niah).
- All 18 cells exit 0.
- Score-consistency violations: 0 across every certified cell (Theorem-2 Δ bound empirically held).
- No CUDA errors, no NaN, no OOM.

## Where we hit vs miss the spec

| Spec criterion | Result |
|---|---|
| PG-19 Δppl < +0.01 at all three contexts | 4K ✗ (+0.0114), 8K ✓ (+0.0030), 16K ✓ (+0.0017) |
| NIAH 4K Δ ≥ −0.02 | ✗ (−0.067) |
| NIAH 8K Δ ≥ −0.04 | ✗ (−0.067) |
| NIAH 16K | dense itself is only 0.70 at 16K — interpret against model ceiling |
| RULER Δ ≤ ±0.01 at all contexts | 4K ✓ (−0.001), 8K ✓ (+0.004), 16K ✓ (+0.007) |
| 0 score-consistency violations | ✓ |
| 0 exploration violations | ✓ |
| No crashes, NaN, OOM | ✓ |

## Artifacts

- `{01..18}_{bench}_{ctxK}_{config}.json` — arXiv-v1 wrapper per cell
- `{01..18}_{bench}_{ctxK}_{config}.native.json` — raw benchmark output
- `{01..18}_{bench}_{ctxK}_{config}.log` — streaming stdout per cell
- Smoke-validation artifacts preserved with `.smoke.` infix for cells 01–06
- Pre-fix (aborted) certified run archived under `prefix_bug_aborted/`
- Paper-2 skip-path ablation archived under `paper2_skip_ablation/`

