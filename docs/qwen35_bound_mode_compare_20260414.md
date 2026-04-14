# Bound Mode Comparison (2026-04-14)

Benchmarks the three certified upper-bound modes introduced by the paper
"Certified Adaptive Attention with Native Compressed Execution" against the
canonical real-mixed Stage 9 serving config on Mac Mini MPS.

## Setup

- Model: Qwen/Qwen3.5-0.8B, float16, MPS
- Corpus: 8-case broader public validation manifest (no overlap with rounds 1–3)
- Serving config: real-mixed Stage 9 (M0/M3 mixed, certified streaming, early-exit
  `check_interval=16`, `priority_value_hybrid` order mode, `direct_m0` strategy)
- Decode steps: 8 per case

## Lanes

| Lane | `enable_interval_bound` | `enable_ellipsoidal_bound` |
|---|---|---|
| `spherical_only` | False | False |
| `interval` | True | False |
| `interval_ellip` | True | True |

The `spherical_only` lane is the pre-paper baseline (Cauchy-Schwarz centroid+radius).
`interval` is the paper's primary bound (per-dimension sign-conditional multiply, O(d) cost).
`interval_ellip` adds the first-principal-component anisotropic bound on top.

## Results

| Lane | avg ms/step | exact_match_vs_dense | score_ms/case | cert_stop_blocks/case |
|---|---|---|---|---|
| `spherical_only` | 538.36 | 0.875 | 203.69 | 384.0 |
| `interval` | **507.82** | **0.875** | 254.80 | 384.0 |
| `interval_ellip` | 830.22 | 0.875 | 419.87 | 384.0 |

### Speed relative to spherical_only

- `interval`: **+5.7% faster** (507.82 vs 538.36 ms/step)
- `interval_ellip`: **-54.2% slower** (830.22 vs 538.36 ms/step)

### Per-case detail

| Case | spherical | interval | interval_ellip | interval delta |
|---|---|---|---|---|
| aae_stage_summary | 727.66 | 660.09 | 887.54 | −9.3% |
| bench_decode_code | 393.33 | 322.85 | 644.20 | −17.9% |
| benchmark_report | 638.79 | 696.24 | 961.53 | +9.0% |
| compressed_page_rfc | 557.90 | 575.59 | 827.09 | +3.2% |
| hip_call_flow | 530.71 | 521.19 | 835.18 | −1.8% |
| local_layer_profiles | 517.64 | 507.90 | 786.85 | −1.9% |
| test_attention_vs_dense | 389.68 | 257.69 | 771.05 | **−33.9%** |
| turboquant_comparison_plan | 551.16 | 521.02 | 928.33 | −5.5% |

Exact-match failure (`turboquant_comparison_plan`) is a pre-existing issue with the
serving config on that case; it fails identically in all three lanes.

## Interpretation

### Interval bound: production-ready, enabled by default

The interval bound delivers a **5.7% average speedup** vs spherical while preserving
exact-match parity. The `cert_stop_blocks` and `checkpoint_count` are identical between
the two lanes — the interval bound does not cause earlier certified exit at the current
`full_attention_mass_eps=1e-3` / `value_eps=1e-3` settings. The latency benefit
comes from the bound computation itself — the O(d) sign-conditional multiply is a
different compute pattern than the spherical Cauchy-Schwarz, leading to different
downstream tensor pipeline behaviour on MPS.

The standout case is `test_attention_vs_dense` at −33.9% (257 vs 390 ms/step), which
suggests the interval bound is significantly tighter for that file's key distribution,
influencing block prioritisation even without crossing the certified exit threshold.

`score_ms/case` is ~25% higher for interval vs spherical (254 vs 203 ms) — the extra
`torch.where` + `sum(dim=-1)` per head adds overhead. Despite this, wall-clock decode is
faster, confirming the benefit exceeds the cost.

**Conclusion**: `enable_interval_bound=True` (the new default) is the correct
production setting for MPS.

### Ellipsoidal bound: too expensive on MPS, keep disabled

The ellipsoidal bound (`enable_ellipsoidal_bound=True`) is **54.2% slower** than
spherical. This is expected: for every decode step, the bound evaluation requires loading
`block_k_pc1[num_blocks, head_dim]` and running several extra tensor ops
(`matmul`, `**2`, `clamp_min`, `sqrt`, `mul`, `add`) per KV head. On MPS, each
kernel launch has a fixed overhead, and these small per-head ops accumulate across
all 28 attention layers × 8 decode steps × up to 100 blocks per call.

The ellipsoidal bound remains gated off by default (`enable_ellipsoidal_bound=False`)
and is not recommended for MPS production use. It may be worth revisiting if
the bound computation is vectorised across all heads in a single kernel (eliminating
the Python-level head loop), or on CUDA where kernel launch overhead is lower.

## Artefacts

- Benchmark script: `benchmarks/bench_qwen35_bound_mode_compare.py`
- JSON results: `benchmarks/results/qwen35_bound_mode_compare_20260414_mps/qwen35_bound_mode_compare.json`
- MD results: `benchmarks/results/qwen35_bound_mode_compare_20260414_mps/qwen35_bound_mode_compare.md`
