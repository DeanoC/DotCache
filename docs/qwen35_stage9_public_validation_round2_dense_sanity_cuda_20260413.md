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

After the fixed-tree handoff fix, the earlier structural divergence is gone on CUDA:

- `performance_journal` now matches dense across real mixed, non-M0 Stage 9, and conservative
- `state_cache_roadmap` now matches dense across all three serving-family lanes

So on fixed-tree CUDA, this is no longer a shared-stage-9-only divergence. The remaining `performance_journal` tail difference is now a tie-boundary case where dense/serving pick `15` from equal `logit[15] = logit[16]`.

## Cases

### `performance_journal`

- dense generated IDs:
  - `[198, 220, 471, 1510, 77518, 28, 15, 7561]`
- real mixed / non-M0 / conservative generated IDs:
  - `[198, 220, 471, 1510, 77518, 28, 15, 7561]`
- serving-family lanes matching each other:
  - yes
- serving-family lanes matching dense:
  - yes
- prefix match length vs dense:
  - real mixed: `8`
  - non-M0 Stage 9: `8`
  - conservative certified: `8`
- first divergent generated token vs dense:
  - real mixed: none
  - non-M0 Stage 9: none
  - conservative certified: none
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
- real mixed / non-M0 Stage 9 / conservative generated IDs:
  - `[12, 264, 11782, 314, 279, 1118, 220, 16]`
- serving-family lanes matching each other:
  - yes
- serving-family lanes matching dense:
  - yes
- prefix match length vs dense:
  - real mixed: `8`
  - non-M0 Stage 9: `8`
  - conservative certified: `8`
- first divergent generated token vs dense:
  - real mixed: none
  - non-M0 Stage 9: none
  - conservative certified: none
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

The remaining divergence between CUDA and MPS on `performance_journal` is now a backend-tail tie behavior:

- CUDA dense and serving both use `[... 28, 15, 7561]`
- MPS dense and serving use `[... 28, 16, 7561]`
- all MPS/CUDA divergences are now interpreted as a late tie-boundary effect, not a Stage 9 mixed regression

## Interpretation

Best current read after the CUDA suspect pass:

- shared DotCache-family boundary remains the best explanation
- this is not evidence of a Stage 9 mixed-only bug
- the structural `state_cache_roadmap` repro is now fixed
- `submission_execution_plan` is no longer included in the original two-case suspects, and is now treated as a separate cousin family that was addressed on MPS post-fix reruns

For historical context from this suspect pass (before the handoff fix), the broader divergence family included:

- `performance_journal`
- `state_cache_roadmap`
- `submission_execution_plan`

while nearby cousin cases such as `statecache_showcase`, `gemma4_apple_compatibility`, and `model_roadmap` remain clean against dense on MPS.
