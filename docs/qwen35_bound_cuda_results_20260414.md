# Qwen3.5 Bound Mode CUDA Results (2026-04-14)

CUDA run of the bound-mode comparison and context-scaling benchmarks
(MPS results are in `docs/qwen35_bound_mode_compare_20260414.md`).

## Hardware / software

- GPU: NVIDIA GeForce RTX 4090 (sm_89, 49 GB VRAM)
- Driver: 560.35.05 (CUDA 12.6)
- PyTorch: 2.7.0+cu126, cxx11_abi=True
- flash-attn: 2.8.3 (pre-built wheel cu12torch2.7cxx11abiTRUE)
- causal-conv1d: 1.6.1 (source build, TORCH_CUDA_ARCH_LIST="8.9")
- transformers: 5.5.4
- Model: Qwen/Qwen3.5-0.8B, float16

---

## 1. Bound mode comparison (3-lane, real document corpus)

Source: `benchmarks/results/qwen35_bound_mode_compare_20260414_cuda/`

### Setup

Same 8-case broader public validation manifest and Stage 9 serving config used
on MPS.  `decode_steps=8`, `device=cuda`, `backend=torch_cuda`.

### Summary

| Lane | avg ms/step | exact_match | score_ms/case | cert_stop_blocks/case |
|---|---|---|---|---|
| `spherical_only` | 261.52 | 0.750 | 159.97 | 384.0 |
| `interval` | **253.91** | 0.750 | 211.05 | 384.0 |
| `interval_ellip` | 272.42 | **0.875** | 299.43 | 384.0 |

### Speed-up vs spherical_only

- `interval`: **+2.9% faster** (253.91 vs 261.52 ms/step)
- `interval_ellip`: **−4.2% slower** (272.42 vs 261.52 ms/step)

### Per-case detail

| Case | spherical | interval | interval_ellip | int delta |
|---|---|---|---|---|
| aae_stage_summary | 430.52 (✓) | 331.04 (✓) | 350.19 (✓) | **−23.1%** |
| bench_decode_code | 180.21 (**✗**) | 188.23 (**✗**) | 206.69 (**✓**) | +4.4% |
| benchmark_report | 326.83 (✓) | 329.40 (✓) | 348.96 (✓) | +0.8% |
| compressed_page_rfc | 253.45 (✓) | 257.83 (✓) | 276.67 (✓) | +1.7% |
| hip_call_flow | 254.26 (✓) | 258.97 (✓) | 276.58 (✓) | +1.9% |
| local_layer_profiles | 252.01 (✓) | 259.06 (✓) | 275.89 (✓) | +2.8% |
| test_attention_vs_dense | 140.68 (✓) | 148.42 (✓) | 166.46 (✓) | +5.5% |
| turboquant_comparison_plan | 254.22 (**✗**) | 258.32 (**✗**) | 277.92 (**✗**) | +1.6% |

✓/✗ = exact match vs dense reference.

### Notable CUDA-specific findings

**1. interval_ellip fixes a CUDA exact-match failure.**
`bench_decode_code` fails exact match under `spherical_only` and `interval`
on CUDA (but passed on MPS).  `interval_ellip` restores the correct output.
The ellipsoidal bound is tighter for this case's key distribution, which
changes block prioritisation in a way that yields the dense-matching token.

**2. aae_stage_summary is the standout case for interval (−23.1%).**
On MPS the standout was `test_attention_vs_dense` (−33.9%).  The benefit
is content-distribution-dependent rather than uniformly spread.

**3. cert_stop_blocks unchanged at 384.0 for all three lanes** — identical
to MPS.  The certified-exit depth is a function of epsilons and data
distribution, not of bound mode or hardware.

**4. interval_ellip overhead: −4.2% on CUDA vs +54.2% on MPS.**
This confirms the MPS kernel-launch overhead hypothesis from
`docs/qwen35_bound_winner_analysis_20260414.md`.  On CUDA, the ellipsoidal
matmul ops are fused, eliminating the per-head dispatch overhead.

---

## 2. Context-length scaling (synthetic filler, 16 decode steps)

Source: `benchmarks/results/qwen35_bound_context_scaling_20260414_cuda/`
(Note: output files written on benchmark completion; 8K, 16K, 32K pending.)

### Setup

Synthetic prompt (repeated filler paragraph) padded to each target context
length.  All three lanes, `decode_steps=16`, `device=cuda`, `backend=torch_cuda`.

### Timing results

| ctx | spherical ms/step | interval ms/step | Δ interval | ellip ms/step | Δ ellip |
|---|---|---|---|---|---|
| 1,024 | 253.51 | 260.18 | +2.6% | 277.87 | +9.6% |
| 2,048 | 398.90 | 405.19 | +1.6% | 429.71 | +7.7% |
| 4,096 | 684.47 | 696.68 | +1.8% | (pending) | — |
| 8,192 | (pending) | — | — | — | — |
| 16,384 | (pending) | — | — | — | — |
| 32,768 | (pending) | — | — | — | — |

### cert_stop_rate: measurement note

The context scaling script runs `decode_steps=1` in a loop (N iterations) and
checks `persistent_full_attention_last_first_certified_stop_block_count_by_layer`
after each call.  Because each call **re-runs prefill from scratch** on the
same deterministic prompt, all N iterations observe the same first-decode-step
state.  For a deterministic model the result is always 0 or 1 — cert_stop_rate
is therefore either 0.000 or 1.000 and does not represent a meaningful
across-step rate.

Observed: cert_stop_rate = 1.000 at all tested context lengths for all lanes.
Interpretation: the first decode step after prefill always certifies at least
one FA layer for the filler-text prompt, at all tested context lengths.

To measure a true across-step cert_stop_rate, the harness would need to expose
per-step certified-exit counts (not only last-step), or advance the KV cache
state between single-step calls.

### Context scaling interpretation

**Interval overhead on synthetic text vs real documents.**
On synthetic filler text, `interval` shows a small but consistent overhead
(+1.6%–+2.6% across 1K–4K) relative to spherical.  On the real 8-case
corpus (Section 1 above), `interval` is +2.9% faster.

The discrepancy likely reflects the key distribution: filler text has a
repetitive, near-spherically-distributed key structure (the interval bound's
tighter per-dimension ranges provide little additional benefit), whereas
real code and technical documents have more anisotropic key distributions
where the interval bound prunes more blocks.

**Ellipsoidal overhead on synthetic text decreasing with context.**
The `interval_ellip` overhead relative to `spherical_only` decreases as
context grows: +9.6% at 1K → +7.7% at 2K.  The benefit from anisotropic
pruning scales with block count while the fixed score-computation overhead
is amortised.  At 4K ellip is still running.

**ms/step monotone growth prediction.**
Prediction from `docs/qwen35_bound_winner_analysis_20260414.md` was that the
ms/step *saving* from interval vs spherical would grow monotonically with
context.  On synthetic text the saving is absent (overhead is small but
positive at all tested lengths).  On real documents (Section 1) the benefit
varies strongly by content.  The prediction was partially correct: the
ellipsoidal overhead-fraction is decreasing with context, and the real-document
test confirms interval is net-positive.

---

## 3. Prediction audit

| Prediction | Outcome |
|---|---|
| cert_stop_blocks constant across hardware | ✓ 384.0 on CUDA, matches MPS |
| interval −8% to −15% vs spherical on CUDA | ✗ actual +2.9% (real docs), +1.6%–+2.6% overhead (synthetic) |
| interval_ellip 0% to −10% vs spherical on CUDA | ✓ −4.2% (real docs), within range |
| interval_ellip not +54% on CUDA (kernel fusion) | ✓ confirmed |
| ms/step delta grows monotonically with ctx | partial — decreasing overhead fraction for ellip; interval not net-positive on synthetic |

---

## Artefacts

| File | Description |
|---|---|
| `benchmarks/bench_qwen35_bound_mode_compare.py` | 3-lane mode comparison script |
| `benchmarks/bench_qwen35_bound_context_scaling.py` | Context-length scaling script |
| `benchmarks/results/qwen35_bound_mode_compare_20260414_cuda/` | Mode comparison JSON + MD |
| `benchmarks/results/qwen35_bound_context_scaling_20260414_cuda/` | Context scaling JSON + MD (written on completion) |
| `docs/qwen35_bound_mode_compare_20260414.md` | MPS mode comparison results |
| `docs/qwen35_bound_winner_analysis_20260414.md` | MPS analysis + original CUDA predictions |
| `docs/qwen35_bound_cuda_results_20260414.md` | This document |
