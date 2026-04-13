# Qwen3.5 Stage 9 `state_cache_roadmap` First-Divergence Localization on CUDA (2026-04-13)

This note records the CUDA-side single-case repro for the strongest current public DotCache-family vs dense boundary.

Single-case repro manifest:

- [benchmarks/manifests/qwen35_stage9_state_cache_roadmap_first_divergence_20260413.json](../benchmarks/manifests/qwen35_stage9_state_cache_roadmap_first_divergence_20260413.json)

CUDA result bundles:

- real mixed:
  [benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_state_cache_roadmap_first_divergence_cuda_real_mixed_densecheck/qwen35_persistent_serving_policy_compare.md](../benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_state_cache_roadmap_first_divergence_cuda_real_mixed_densecheck/qwen35_persistent_serving_policy_compare.md)
- non-M0 Stage 9:
  [benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_state_cache_roadmap_first_divergence_cuda_stage9_non_m0_densecheck/qwen35_persistent_serving_policy_compare.md](../benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_state_cache_roadmap_first_divergence_cuda_stage9_non_m0_densecheck/qwen35_persistent_serving_policy_compare.md)
- conservative certified:
  [benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_state_cache_roadmap_first_divergence_cuda_conservative_priority_value_hybrid_ci16_densecheck/qwen35_persistent_serving_policy_compare.md](../benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_state_cache_roadmap_first_divergence_cuda_conservative_priority_value_hybrid_ci16_densecheck/qwen35_persistent_serving_policy_compare.md)

Related MPS note:

- [qwen35_stage9_state_cache_roadmap_first_divergence_20260413.md](./qwen35_stage9_state_cache_roadmap_first_divergence_20260413.md)

## Main result

CUDA reproduces the same core first-divergence pattern as MPS on `state_cache_roadmap`:

- dense generated IDs:
  - `[12, 264, 11782, 314, 279, 1118, 220, 16]`
- real mixed generated IDs:
  - `[12, 1118, 78361, 321, 1118, 7652, 29642, 364]`
- non-M0 Stage 9 generated IDs:
  - `[12, 1118, 78361, 321, 1118, 7652, 29642, 364]`
- conservative certified generated IDs:
  - `[12, 1118, 78361, 321, 1118, 7652, 29642, 364]`

So on CUDA too:

- all serving-family lanes match each other exactly
- all serving-family lanes diverge from dense
- the shared prefix length vs dense is `1`
- the first divergent generated token index is `2`

That keeps `state_cache_roadmap` as the cleanest single public repro for earliest-difference debugging.

## CUDA execution shape

### Real mixed

- processed blocks: `85`
- checkpoints: `0`
- executed `M0` blocks: `4032`
- executed all-`M3` blocks: `48`
- executed exact-key `M3` blocks: `56`

### Non-M0 Stage 9

- processed blocks: `85`
- checkpoints: `0`
- executed `M0` blocks: `0`
- executed all-`M3` blocks: `0`
- executed exact-key `M3` blocks: `0`

### Conservative certified

- processed blocks: `85`
- checkpoints: `0`
- executed `M0` blocks: `0`
- executed all-`M3` blocks: `0`
- executed exact-key `M3` blocks: `0`

## CUDA vs MPS read

The main first-divergence behavior matches across backends:

- same dense output
- same serving-family output
- same shared prefix length `1`
- same first divergent token index `2`
- same real-mixed processed-block count `85`
- same real-mixed mixed frontier shape:
  - `M0=4032`
  - `all-M3=48`
  - `exact-key-M3=56`

The main backend difference in the current-tree runtime is checkpointing:

- MPS targeted real mixed recorded `6` checkpoints
- CUDA targeted real mixed recorded `0` checkpoints

That means the repro is backend-consistent at the output and execution-shape level, but the current CUDA runtime is still not recreating the older checkpoint-heavy behavior seen in the MPS note.

## Current interpretation

Best current read after the single-case CUDA repro:

- this remains a shared DotCache-family vs dense boundary
- it is not CUDA-only
- it is not a Stage 9 mixed-only regression
- the earliest externally visible divergence is still generated token `2`

So this manifest is the right canonical CUDA target for any deeper instrumentation that tries to answer:

- what internal quantity first differs from dense before token `2` is emitted?
