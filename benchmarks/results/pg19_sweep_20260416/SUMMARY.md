# PG-19 perplexity sweep — 2026-04-16

**Model:** NousResearch/Meta-Llama-3.1-8B (INT8 via bitsandbytes)
**Profile:** `configs/NousResearch_Meta-Llama-3.1-8B_calibrated.npz`
**Setup:** 10 chunks × 4K context, eval_start_frac=0.5 (last 2048 tokens evaluated with certified attention)

| Threshold | Cert ppl | Dense ppl | Δppl (abs) | Δppl (%) | Skip rate |
|----------:|---------:|----------:|-----------:|---------:|----------:|
| 0.00      | 6.9255   | 6.9153    | +0.0102    | +0.147%  | 56.65% |
| 0.01      | 6.9255   | 6.9153    | +0.0102    | +0.147%  | 56.64% |
| 0.02      | 6.9219   | 6.9153    | +0.0066    | +0.096%  | 56.60% |

## Findings

1. **Certified attention matches dense perplexity on PG-19 to within 0.15%** across all thresholds — well inside the margin where INT8 quantisation noise dominates.
2. **Skip rate ~56.6% is indistinguishable across thresholds.** Entropy gating at 1% and 2% produces essentially the same behaviour as gate-off.
3. **PG-19 attention is more concentrated than expected.** The prior hypothesis (language modelling → diffuse attention → gate triggers often) is wrong; per-head max-block-mass is almost always ≥ 2%.
4. **Threshold=0.02 is safe for PG-19** — no skip-rate regression, marginal ppl improvement.

## Decode speedup

This sweep only became feasible after fixing the cert decode hot path (session 20260416c):
- Pre-fix: 730 ms/cert step → ~10 h per PG-19 config (infeasible)
- Post-fix: 88 ms/cert step → ~30 min per config (this sweep: ~90 min total)

Root cause: `sdpa_attend_with_skip` did a CPU→GPU copy of the growing key cache every layer every step. Fixed by mirroring `keys_fp16_cpu` to a GPU-resident `keys_fp16_gpu` buffer. Also vectorised the top-K mask-clear loop and removed a per-layer `.any()` sync.
