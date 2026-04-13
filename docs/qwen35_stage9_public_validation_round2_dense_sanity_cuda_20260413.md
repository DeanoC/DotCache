# Qwen3.5 Stage 9 Round 2 Dense Sanity Check on CUDA (2026-04-13)

This note records the CUDA-side equivalent of the round-2 dense sanity pass on the original two suspect cases.

Primary suspect manifest:

- [benchmarks/manifests/qwen35_stage9_repo_public_validation_round2_suspects_20260413.json](../benchmarks/manifests/qwen35_stage9_repo_public_validation_round2_suspects_20260413.json)

CUDA result bundles:

- real mixed:
  [benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_repo_promptfiles_public_validation_round2_suspects_cuda_real_mixed_densecheck/qwen35_persistent_serving_policy_compare.md](../benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_repo_promptfiles_public_validation_round2_suspects_cuda_real_mixed_densecheck/qwen35_persistent_serving_policy_compare.md)
- non-M0 Stage 9:
  [benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_repo_promptfiles_public_validation_round2_suspects_cuda_stage9_non_m0_densecheck/qwen35_persistent_serving_policy_compare.md](../benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_repo_promptfiles_public_validation_round2_suspects_cuda_stage9_non_m0_densecheck/qwen35_persistent_serving_policy_compare.md)
- conservative certified:
  [benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_repo_promptfiles_public_validation_round2_suspects_cuda_conservative_priority_value_hybrid_ci16_densecheck/qwen35_persistent_serving_policy_compare.md](../benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_repo_promptfiles_public_validation_round2_suspects_cuda_conservative_priority_value_hybrid_ci16_densecheck/qwen35_persistent_serving_policy_compare.md)

Related MPS notes:

- [qwen35_stage9_public_validation_round2_dense_sanity_20260413.md](./qwen35_stage9_public_validation_round2_dense_sanity_20260413.md)
- [qwen35_stage9_round2_cousin_dense_check_20260413.md](./qwen35_stage9_round2_cousin_dense_check_20260413.md)

## Main result

CUDA shows the same core pattern as MPS on the original two suspect cases:

- all three serving-family lanes match each other exactly
- all three serving-family lanes diverge from dense
- real mixed still matches hand exactly
- non-M0 still matches hand exactly
- conservative still matches hand exactly

So on CUDA too, this currently looks like a shared DotCache-family vs dense boundary, not a Stage 9 mixed-only regression.

## Cases

### `performance_journal`

- dense generated IDs:
  - `[198, 220, 471, 1510, 77518, 28, 15, 7561]`
- real mixed generated IDs:
  - `[198, 220, 471, 1510, 8412, 1551, 7408, 63]`
- non-M0 Stage 9 generated IDs:
  - `[198, 220, 471, 1510, 8412, 1551, 7408, 63]`
- conservative certified generated IDs:
  - `[198, 220, 471, 1510, 8412, 1551, 7408, 63]`
- serving-family lanes matching each other:
  - yes
- serving-family lanes matching dense:
  - no
- prefix match length vs dense:
  - real mixed: `4`
  - non-M0 Stage 9: `4`
  - conservative certified: `4`
- first divergent generated token vs dense:
  - real mixed: token `5`
  - non-M0 Stage 9: token `5`
  - conservative certified: token `5`
- processed block count at final step:
  - real mixed: `129`
  - non-M0 Stage 9: `129`
  - conservative certified: `129`
- checkpoint count at final step:
  - real mixed: `0`
  - non-M0 Stage 9: `0`
  - conservative certified: `0`
- execution shape:
  - real mixed: `M0=6192`, `all-M3=0`, `exact-key-M3=8`
  - non-M0 Stage 9: `M0=0`, `all-M3=0`, `exact-key-M3=0`
  - conservative certified: `M0=0`, `all-M3=0`, `exact-key-M3=0`

### `state_cache_roadmap`

- dense generated IDs:
  - `[12, 264, 11782, 314, 279, 1118, 220, 16]`
- real mixed generated IDs:
  - `[12, 1118, 78361, 321, 1118, 7652, 29642, 364]`
- non-M0 Stage 9 generated IDs:
  - `[12, 1118, 78361, 321, 1118, 7652, 29642, 364]`
- conservative certified generated IDs:
  - `[12, 1118, 78361, 321, 1118, 7652, 29642, 364]`
- serving-family lanes matching each other:
  - yes
- serving-family lanes matching dense:
  - no
- prefix match length vs dense:
  - real mixed: `1`
  - non-M0 Stage 9: `1`
  - conservative certified: `1`
- first divergent generated token vs dense:
  - real mixed: token `2`
  - non-M0 Stage 9: token `2`
  - conservative certified: token `2`
- processed block count at final step:
  - real mixed: `85`
  - non-M0 Stage 9: `85`
  - conservative certified: `85`
- checkpoint count at final step:
  - real mixed: `0`
  - non-M0 Stage 9: `0`
  - conservative certified: `0`
- execution shape:
  - real mixed: `M0=4032`, `all-M3=48`, `exact-key-M3=56`
  - non-M0 Stage 9: `M0=0`, `all-M3=0`, `exact-key-M3=0`
  - conservative certified: `M0=0`, `all-M3=0`, `exact-key-M3=0`

## CUDA vs MPS nuance

The main pattern matches MPS, but one dense-only backend difference is worth recording:

- `performance_journal` CUDA dense generated IDs are
  - `[198, 220, 471, 1510, 77518, 28, 15, 7561]`
- `performance_journal` MPS dense generated IDs were
  - `[198, 220, 471, 1510, 77518, 28, 16, 7561]`

That dense-only tail change does not alter the serving-family read:

- the serving-family lanes still align exactly with each other on both backends
- the serving-family divergence from dense still begins earlier, after the same shared prefix length of `4`

## Interpretation

Best current read after the CUDA suspect pass:

- shared DotCache-family boundary remains the best explanation
- this is not CUDA-only behavior
- this is not evidence of a Stage 9 mixed-only bug
- `state_cache_roadmap` remains the stronger structural repro because it diverges at token `2` and is the only original suspect that exercises a visible exact-key / all-M3 frontier in real mixed

With the new cousin dense check now on branch, the broader divergence family is:

- `performance_journal`
- `state_cache_roadmap`
- `submission_execution_plan`

while nearby cousin cases such as `statecache_showcase`, `gemma4_apple_compatibility`, and `model_roadmap` remain clean against dense on MPS.
