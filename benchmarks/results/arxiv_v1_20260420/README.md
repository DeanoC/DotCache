# arXiv v1 Benchmark Sweep — LLaMA 3.1-8B

**Started:** 2026-04-20  
**Spec:** `docs/arxiv_v1_benchmark_spec.md` (the /workspace conversation spec)  
**Branch:** `feature/interval-ellipsoidal-bounds`

## Configuration (certified side)

All paper features on, per the spec:

| Flag | Value |
|---|---|
| `--tau-cov` | 0.995 |
| `--k-min` | 2 |
| `--k-max` | 128 (per spec; the "NOT None" operating point) |
| `--v-tol` (DEFAULT_V_TOLERANCE) | 0.05 |
| `--kv-value-group-size` | 16 (hardcoded in tiered cache construction) |
| `--fp64-accumulators` | on (hardcoded in Triton kernels) |
| `--ranking-fallback` | on |
| `--ranking-r` | 1 |
| `--ranking-fallback-mode` | full |
| `--score-consistency-check` | on |
| `--eps-guard` | 0.01 |
| `--exploration-rate` | 0.02 |
| `--rung1-threshold` | 0.02 |
| `--rung1-multiplier` | 2.0 |

**DEFAULT_V_TOLERANCE:** the spec says `v_tol=0.05` but our code default is `0.5`. The orchestrator patches this to 0.05 per-run via env var / direct state override where possible. Flagged as a known discrepancy.

## Sweep layout

18 runs: 3 benchmarks × 3 context lengths × 2 configs.

| # | Benchmark | Context | Config |
|---|---|---|---|
| 01 | PG-19 | 4K | dense |
| 02 | PG-19 | 4K | certified |
| 03 | NIAH | 4K | dense |
| 04 | NIAH | 4K | certified |
| 05 | RULER | 4K | dense |
| 06 | RULER | 4K | certified |
| 07-12 | (same) | 8K | (same) |
| 13-18 | (same) | 16K | (same) |

Each run produces `{nn}_{benchmark}_{context}_{config}.json` plus a streaming log. The orchestrator commits+pushes after each completed run.

## Spec flags not implemented in code

Flagged here for the paper methodology section — these spec knobs refer to features we don't have. We use the closest available analog:

- `--kv-tier2-dtype bfloat16` — tier-2 is FP16 (pin_memory() on FP16 tensors). If the base model emits BF16 KV, the tiered cache stores FP16 after cast. Close enough for quality; delta tracked.
- `--v-fallback-format bfloat16` — value escalation pages in FP16 (same float width), not BF16.
- `--ranking-fallback-impl sdpa` — Rung-3 uses an in-repo `recompute_heads_dense_fp16` path (einsum over FP16 K, FP32-cast V), not `torch.scaled_dot_product_attention`. Equivalent math.
- `--kv-key-quant per-channel` — the only supported quantisation; symmetric (scale-only, no zero point). Paper §2.3 "asymmetric" statement is slightly aspirational; see audit.

## Expected

Per the spec's success criteria:

- PG-19: Δppl < +0.01 at all contexts
- NIAH 4K: Δ ≤ 2pp; 8K: Δ ≤ 4pp; 16K: noted
- RULER: Δ ≤ ±1pp at all contexts
- Canaries: 0 score_consistency_violations, 0 exploration_violations
- No crashes / NaN / OOM
