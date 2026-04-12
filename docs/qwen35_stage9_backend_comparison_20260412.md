# Qwen3.5 Stage 9 Backend Comparison (2026-04-12)

This note records the current backend-dependent Stage 9 picture after:

- the refreshed MPS portable real-mixed reruns
- the first honest CUDA portable reproduction with working fast-path libraries

The goal is to keep one stable read of what currently wins on each backend.

## Main conclusion

The core algorithmic thesis now appears to transfer across backends:

- certified streaming works
- key-side mixed `M0` execution works
- exact-match behavior on the tested portable corpora is preserved

But the best serving policy is currently backend-dependent:

- on MPS, real mixed Stage 9 is the serving winner
- on CUDA, real mixed Stage 9 now wins on `large` and `broad`
- `external` still prefers the older non-`M0` Stage 9 path

That is still a useful result, not a contradiction. It means the method transfers, and the remaining backend-dependent gap is now much narrower and more localized.

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

## Current CUDA portable reference

Current CUDA real-mixed bundles:

- large:
  - [benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_large_cuda_frontier_batchedresidual_v6/qwen35_persistent_real_mixed_probe.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_large_cuda_frontier_batchedresidual_v6/qwen35_persistent_real_mixed_probe.md)
  - bias `399.74 ms/step`
  - exact-match `1.0`
- broad:
  - [benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_broad_cuda_frontier_batchedresidual_v6/qwen35_persistent_real_mixed_probe.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_broad_cuda_frontier_batchedresidual_v6/qwen35_persistent_real_mixed_probe.md)
  - bias `460.70 ms/step`
  - exact-match `1.0`
- external:
  - [benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_external_cuda_frontier_batchedresidual_v6/qwen35_persistent_real_mixed_probe.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_external_cuda_frontier_batchedresidual_v6/qwen35_persistent_real_mixed_probe.md)
  - bias `252.42 ms/step`
  - exact-match `1.0`

Important CUDA observation:

- the updated real-mixed CUDA path is no longer just a research lane on every corpus
- it now beats the older non-`M0` baseline on `large` and `broad`
- `external` remains the main mixed-path holdout

## CUDA baselines on the same portable corpus

Non-`M0` Stage 9 CUDA bundles:

- large:
  - [benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_large_stage9_non_m0/qwen35_persistent_serving_policy_compare.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_large_stage9_non_m0/qwen35_persistent_serving_policy_compare.md)
  - bias `468.69 ms/step`
- broad:
  - [benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_broad_stage9_non_m0/qwen35_persistent_serving_policy_compare.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_broad_stage9_non_m0/qwen35_persistent_serving_policy_compare.md)
  - bias `632.12 ms/step`
- external:
  - [benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_external_stage9_non_m0/qwen35_persistent_serving_policy_compare.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_external_stage9_non_m0/qwen35_persistent_serving_policy_compare.md)
  - bias `192.54 ms/step`

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
| large | real mixed `1407.44` | real mixed `399.74` |
| broad | real mixed `1627.72` | real mixed `460.70` |
| external | real mixed `843.77` | non-`M0` Stage 9 `192.54` |

This is the main backend-dependent result.

## Compact matrix

Portable-corpus `bias` comparison at the current checked-in checkpoints:

| Corpus | MPS real mixed | CUDA real mixed | CUDA non-`M0` Stage 9 | CUDA conservative certified |
| --- | ---: | ---: | ---: | ---: |
| large | `1407.44` | `399.74` | `468.69` | `652.91` |
| broad | `1627.72` | `460.70` | `632.12` | `792.60` |
| external | `843.77` | `252.42` | `192.54` | `380.85` |

Across this matrix:

- exact-match stays `1.0` for the checked-in MPS real-mixed and CUDA real-mixed runs
- CUDA real mixed is now the current serving winner on `large` and `broad`
- CUDA non-`M0` Stage 9 still wins on `external`
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
- the current `direct_m0` realization now looks strong enough on CUDA for `large` and `broad`
- the remaining divergence is concentrated in the `external` corpus and its `final_mix` cost

### Practical current policy

If we had to choose today:

- MPS:
  - prefer real mixed Stage 9 `bias`
- CUDA:
  - prefer real mixed Stage 9 `bias` for `large` and `broad`
  - keep non-`M0` Stage 9 `bias` as the better current external-corpus lane
  - keep the CUDA mixed path focused on reducing `final_mix`

## Immediate next work

The current CUDA optimization target should not be phrased as "make mixed work at all." It already works and already wins on two of the three portable corpora.

The real CUDA question is:

- can the remaining `external` `final_mix` cost be reduced enough for real mixed to beat the current non-`M0` Stage 9 baseline there too

That makes the next split very clean:

- CUDA:
  - optimize the remaining mixed `final_mix` realization on `external`
- local/MPS:
  - keep documenting the backend comparison honestly
  - keep the portable MPS reference bundles current

This is a strong place to be scientifically:

- the method is portable enough to reproduce
- the best systems realization is still backend-specific
- and the next optimization target on CUDA is now well localized
