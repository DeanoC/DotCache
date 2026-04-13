# Qwen3.5 Stage 9 Backend Comparison (2026-04-12)

This note records the current backend-dependent Stage 9 picture after:

- the refreshed MPS portable real-mixed reruns
- the latest CUDA same-tree real-mixed cleanup and controls

The goal is to keep one stable read of what currently wins on each backend.

Canonical generated matrix artifact:

- [benchmarks/results/qwen35_stage9_backend_matrix_20260412/qwen35_stage9_backend_matrix.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_stage9_backend_matrix_20260412/qwen35_stage9_backend_matrix.md)

## Main conclusion

The core algorithmic thesis now appears to transfer across backends:

- certified streaming works
- key-side mixed `M0` execution works
- exact-match behavior on the tested portable corpora is preserved

But the best serving policy is currently backend-dependent:

- on MPS, real mixed Stage 9 is the serving winner
- on CUDA, real mixed Stage 9 now wins on `large`, `broad`, and `external` in same-tree comparisons

That is an even cleaner result. The method transfers, and the remaining backend-dependent gap is no longer about which high-level policy wins. It is now mainly about how much headroom the current mixed realization still has on each backend.

## Current MPS portable reference

Refreshed MPS real-mixed bundles on the current runtime:

- large:
  - [benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_repo_promptfiles_large_mps_refreshed/qwen35_persistent_real_mixed_probe.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_repo_promptfiles_large_mps_refreshed/qwen35_persistent_real_mixed_probe.md)
  - bias `1407.44 ms/step`
- broad:
  - [benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_repo_promptfiles_broad_mps_refreshed/qwen35_persistent_real_mixed_probe.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_repo_promptfiles_broad_mps_refreshed/qwen35_persistent_real_mixed_probe.md)
  - bias `1627.72 ms/step`
- external:
  - [benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_repo_promptfiles_external_mps_refreshed/qwen35_persistent_real_mixed_probe.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_repo_promptfiles_external_mps_refreshed/qwen35_persistent_real_mixed_probe.md)
  - bias `843.77 ms/step`

Across these refreshed MPS portable runs:

- `bias` beats `hand` on every corpus
- exact-match stays `1.0`
- the runtime is genuinely doing heavy key-side `M0`
- a small exact-key fallback frontier still exists:
  - `executed_exact_key_m3_blocks_per_case = 8.0`

Refreshed MPS comparison baselines on the same portable corpus:

- non-`M0` Stage 9:
  - large `3784.86`
  - broad `4234.40`
  - external `1700.51`
- conservative certified:
  - large `2054.42`
  - broad `2828.78`
  - external `1340.05`

That makes the current MPS read very clean:

- real mixed is the serving winner on all three portable corpora
- refreshed MPS non-`M0` is no longer competitive on this current runtime
- refreshed MPS conservative remains useful as the safe exact lane, but not the speed winner

## Current CUDA portable reference

Current CUDA real-mixed bundles:

- large:
  - [benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_large_cuda_frontier_batchedresidual_v18_clean/qwen35_persistent_real_mixed_probe.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_large_cuda_frontier_batchedresidual_v18_clean/qwen35_persistent_real_mixed_probe.md)
  - bias `374.06 ms/step`
  - exact-match `1.0`
- broad:
  - [benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_broad_cuda_frontier_batchedresidual_v18_clean/qwen35_persistent_real_mixed_probe.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_broad_cuda_frontier_batchedresidual_v18_clean/qwen35_persistent_real_mixed_probe.md)
  - bias `443.30 ms/step`
  - exact-match `1.0`
- external:
  - [benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_external_cuda_frontier_batchedresidual_v17_clean/qwen35_persistent_real_mixed_probe.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_external_cuda_frontier_batchedresidual_v17_clean/qwen35_persistent_real_mixed_probe.md)
  - bias `240.84 ms/step`
  - exact-match `1.0`

### CUDA public-validation reference

- real-mixed:
  - [benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_public_validation/qwen35_persistent_real_mixed_probe.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_public_validation/qwen35_persistent_real_mixed_probe.md)
  - bias `327.70 ms/step`
  - exact-match `1.0`
- non-`M0`:
  - [benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_public_validation_stage9_non_m0/qwen35_persistent_serving_policy_compare.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_public_validation_stage9_non_m0/qwen35_persistent_serving_policy_compare.md)
  - bias `339.66 ms/step`
- conservative certified:
  - [benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_public_validation_conservative_priority_value_hybrid_ci16/qwen35_persistent_serving_policy_compare.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_public_validation_conservative_priority_value_hybrid_ci16/qwen35_persistent_serving_policy_compare.md)
  - bias `333.35 ms/step`

CUDA public-validation main read:

- real mixed `327.70 ms/step`
- conservative `333.35 ms/step`
- non-`M0` `339.66 ms/step`
- real mixed still wins (`1.69%` and `3.52%` margins vs conservative and non-`M0`)
- exact-match stays `1.0`

Important CUDA observation:

- the updated real-mixed CUDA path is no longer just a research lane on every corpus
- it now beats the same-tree non-`M0` baseline on all three portable corpora
- the remaining CUDA work is about increasing headroom, not proving mixed viability
- the remaining `performance_journal` tail diff is now localized to tiny upstream mixed-path numeric drift before argmax
- forced CUDA capture shows direct-`M0` / `final_mix` matches float32 recompute very closely, so `final_mix` is not the correctness bug
- that leaves CUDA focused on performance headroom, especially `final_mix` and `direct_m0_score`, rather than correctness closure

## CUDA baselines on the same portable corpus

Non-`M0` Stage 9 CUDA bundles:

- large:
  - [benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_large_stage9_non_m0_currenttree_v2/qwen35_persistent_serving_policy_compare.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_large_stage9_non_m0_currenttree_v2/qwen35_persistent_serving_policy_compare.md)
  - bias `388.78 ms/step`
- broad:
  - [benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_broad_stage9_non_m0_currenttree_v2/qwen35_persistent_serving_policy_compare.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_broad_stage9_non_m0_currenttree_v2/qwen35_persistent_serving_policy_compare.md)
  - bias `465.40 ms/step`
- external:
  - [benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_external_stage9_non_m0_currenttree/qwen35_persistent_serving_policy_compare.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_external_stage9_non_m0_currenttree/qwen35_persistent_serving_policy_compare.md)
  - bias `244.02 ms/step`

Conservative certified CUDA bundles:

- large:
  - [benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_large_conservative_priority_value_hybrid_ci16/qwen35_persistent_serving_policy_compare.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_large_conservative_priority_value_hybrid_ci16/qwen35_persistent_serving_policy_compare.md)
  - bias `652.91 ms/step`
- broad:
  - [benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_broad_conservative_priority_value_hybrid_ci16/qwen35_persistent_serving_policy_compare.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_broad_conservative_priority_value_hybrid_ci16/qwen35_persistent_serving_policy_compare.md)
  - bias `792.60 ms/step`
- external:
  - [benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_external_conservative_priority_value_hybrid_ci16/qwen35_persistent_serving_policy_compare.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_external_conservative_priority_value_hybrid_ci16/qwen35_persistent_serving_policy_compare.md)
  - bias `380.85 ms/step`

## Winner table

Portable-corpus bias winners at the current checkpoints:

| Corpus | MPS winner | CUDA winner |
| --- | --- | --- |
| large | real mixed `1407.44` | real mixed `374.06` |
| broad | real mixed `1627.72` | real mixed `443.30` |
| external | real mixed `843.77` | real mixed `240.84` |
| public_validation | N/A | real mixed `327.70` |

This is the main backend-dependent result.

## Compact matrix

Portable-corpus `bias` comparison at the current checked-in checkpoints:

| Corpus | MPS real mixed | MPS non-`M0` | MPS conservative | CUDA real mixed | CUDA non-`M0` Stage 9 | CUDA conservative certified |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| large | `1407.44` | `3784.86` | `2054.42` | `374.06` | `388.78` | `652.91` |
| broad | `1627.72` | `4234.40` | `2828.78` | `443.30` | `465.40` | `792.60` |
| external | `843.77` | `1700.51` | `1340.05` | `240.84` | `244.02` | `380.85` |
| public_validation | N/A | N/A | N/A | `327.70` | `339.66` | `333.35` |

Across this matrix:

- exact-match stays `1.0` for the checked-in MPS real-mixed and CUDA real-mixed runs
- refreshed MPS real mixed is the current serving winner on all three portable corpora
- CUDA real mixed is now the current serving winner on all three portable corpora in same-tree comparisons
- CUDA conservative certified remains a useful safe lane, but not the latency winner

## Larger-model note

One useful machine-specific limitation is worth recording:

- on this Apple MPS host, `Qwen/Qwen3.5-4B` is loadable and can execute the Stage 9 real-mixed path
- but it causes enough paging that it is not a practical benchmark lane on this machine

So for backend comparison, the meaningful current cross-device reference remains the `0.8B` portable corpus set.

## Interpretation

### What looks stable already

- MPS supports the stronger systems claim:
  - real mixed Stage 9 is the serving winner on the main local measurements
- CUDA supports the algorithmic claim:
  - certified streaming and mixed `M0` both work
  - outputs stay exact on the tested portable corpora

### What is not stable yet

- the preferred Stage 9 execution policy is not yet fully backend-invariant
- the current `direct_m0` realization now looks strong enough on CUDA across all three portable corpora
- the remaining divergence is more about backend-specific headroom and hotspots than about which policy wins

### Practical current policy

If we had to choose today:

- MPS:
  - prefer real mixed Stage 9 `bias`
- CUDA:
  - prefer real mixed Stage 9 `bias`
  - keep the CUDA mixed path focused on reducing `final_mix`

## Immediate next work

The current CUDA optimization target should not be phrased as "make mixed work at all." It already works and now wins on all three portable corpora in same-tree comparisons.

The real CUDA question is:

- can the remaining `final_mix` cost be reduced enough to widen the current mixed headroom, especially on `external`

That makes the next split very clean:

- CUDA:
  - optimize the remaining mixed `final_mix` realization, especially on `external`
- local/MPS:
  - keep documenting the backend comparison honestly
  - keep the portable MPS reference bundles current

This is a strong place to be scientifically:

- the method is portable enough to reproduce
- the best systems realization is still backend-specific
- and the next optimization target on CUDA is now well localized
