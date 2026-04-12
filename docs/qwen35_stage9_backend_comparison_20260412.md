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
- on CUDA, non-`M0` Stage 9 is currently the serving winner on the portable corpus

That is a useful result, not a contradiction. It means the method transfers, while the best systems realization still differs by backend.

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

Real-mixed CUDA bundles:

- large:
  - [benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_large/qwen35_persistent_real_mixed_probe.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_large/qwen35_persistent_real_mixed_probe.md)
  - hand `617.78 ms/step`
  - bias `611.34 ms/step`
  - exact-match `1.0`
- broad:
  - [benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_broad/qwen35_persistent_real_mixed_probe.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_broad/qwen35_persistent_real_mixed_probe.md)
  - hand `752.07 ms/step`
  - bias `750.07 ms/step`
  - exact-match `1.0`
- external:
  - [benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_external/qwen35_persistent_real_mixed_probe.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_external/qwen35_persistent_real_mixed_probe.md)
  - hand `378.04 ms/step`
  - bias `376.70 ms/step`
  - exact-match `1.0`

Important CUDA observation:

- the real-mixed CUDA path is not losing because it silently fell back to an exact path
- it is executing `M0` only on these portable runs
- so the current CUDA loss is the remaining cost of the mixed `direct_m0` score/mix path itself

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
| large | real mixed `1407.44` | non-`M0` Stage 9 `468.69` |
| broad | real mixed `1627.72` | non-`M0` Stage 9 `632.12` |
| external | real mixed `843.77` | non-`M0` Stage 9 `192.54` |

This is the main backend-dependent result.

## Compact matrix

Portable-corpus `bias` comparison at the current checked-in checkpoints:

| Corpus | MPS real mixed | CUDA real mixed | CUDA non-`M0` Stage 9 | CUDA conservative certified |
| --- | ---: | ---: | ---: | ---: |
| large | `1407.44` | `611.34` | `468.69` | `652.91` |
| broad | `1627.72` | `750.07` | `632.12` | `792.60` |
| external | `843.77` | `376.70` | `192.54` | `380.85` |

Across this matrix:

- exact-match stays `1.0` for the checked-in MPS real-mixed and CUDA real-mixed runs
- CUDA non-`M0` Stage 9 is the current serving winner on all three portable corpora
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

- the preferred Stage 9 execution policy is not yet backend-invariant
- the current `direct_m0` realization is MPS-friendly enough to win there
- the same path is still too expensive on CUDA relative to the non-`M0` Stage 9 baseline

### Practical current policy

If we had to choose today:

- MPS:
  - prefer real mixed Stage 9 `bias`
- CUDA:
  - prefer non-`M0` Stage 9 `bias`
  - keep real mixed `direct_m0` as the optimization/research lane

## Immediate next work

The current CUDA optimization target should not be phrased as "make mixed work at all." It already works.

The real CUDA question is:

- can a more CUDA-native packed score/mix path beat the current non-`M0` Stage 9 baseline

That makes the next split very clean:

- CUDA:
  - optimize the packed score/mix realization
- local/MPS:
  - keep documenting the backend comparison honestly
  - keep the portable MPS reference bundles current

This is a strong place to be scientifically:

- the method is portable enough to reproduce
- the best systems realization is still backend-specific
- and the next optimization target on CUDA is now well localized
