# Qwen3.5 Stage 9 Round 2 Fixed-Tree CUDA Confidence Pass (2026-04-13)

This note records the fixed-tree CUDA confidence sweep after the logical-sequence-length
handoff fix from [qwen35_stage9_state_cache_roadmap_root_cause_20260413.md](qwen35_stage9_state_cache_roadmap_root_cause_20260413.md).

It updates the earlier pre-fix CUDA sanity read from
[qwen35_stage9_public_validation_round2_dense_sanity_cuda_20260413.md](qwen35_stage9_public_validation_round2_dense_sanity_cuda_20260413.md),
and it should be read alongside the post-fix MPS note
[qwen35_stage9_round2_postfix_mps_20260413.md](qwen35_stage9_round2_postfix_mps_20260413.md)
and the CUDA tie-boundary note
[qwen35_stage9_performance_journal_residual_cuda_20260413.md](qwen35_stage9_performance_journal_residual_cuda_20260413.md).

## CUDA fixed-tree round-2 suspects

Artifacts:

- [real mixed suspect bundle](../benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_repo_promptfiles_public_validation_round2_suspects_cuda_real_mixed_densecheck/qwen35_persistent_serving_policy_compare.md)
- [non-M0 suspect bundle](../benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_repo_promptfiles_public_validation_round2_suspects_cuda_stage9_non_m0_densecheck/qwen35_persistent_serving_policy_compare.md)
- [conservative suspect bundle](../benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_repo_promptfiles_public_validation_round2_suspects_cuda_conservative_priority_value_hybrid_ci16_densecheck/qwen35_persistent_serving_policy_compare.md)

### `state_cache_roadmap`

This case no longer shows the earlier structural divergence on CUDA.

- dense:
  - `[12, 264, 11782, 314, 279, 1118, 220, 16]`
- real mixed:
  - `[12, 264, 11782, 314, 279, 1118, 220, 16]`
- non-M0:
  - `[12, 264, 11782, 314, 279, 1118, 220, 16]`
- conservative:
  - `[12, 264, 11782, 314, 279, 1118, 220, 16]`

Exact-match flags:

- real mixed vs dense: `true`
- non-M0 vs dense: `true`
- conservative vs dense: `true`
- all three lanes vs hand: `true`

Serving telemetry:

- real mixed processed blocks: `85`
- real mixed checkpoints: `0`
- real mixed executed M0: `4032`
- real mixed executed all-M3: `48`
- real mixed executed exact-key M3: `56`

So the canonical early-divergence repro from the pre-fix CUDA note is gone on fixed tree.

### `performance_journal`

This is the only remaining suspect-case outlier, and it is late rather than structural.

- dense:
  - `[198, 220, 471, 1510, 77518, 28, 15, 7561]`
- real mixed:
  - `[198, 220, 471, 1510, 77518, 28, 16, 7561]`
- non-M0:
  - `[198, 220, 471, 1510, 77518, 28, 15, 7561]`
- conservative:
  - `[198, 220, 471, 1510, 77518, 28, 15, 7561]`

Exact-match flags:

- real mixed vs dense: `false`
- non-M0 vs dense: `true`
- conservative vs dense: `true`
- all three lanes vs hand: `true`

Prefix behavior:

- real mixed prefix match vs dense: `6`
- non-M0 prefix match vs dense: `8`
- conservative prefix match vs dense: `8`

The only remaining mismatch is the late `15`/`16` tail step. No suspect case now diverges before the final two generated positions, and `performance_journal` still reconverges on the final token `7561`.

## CUDA fixed-tree round-2 cousins

Artifacts:

- [real mixed cousin bundle](../benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_repo_promptfiles_public_validation_round2_cousins_cuda_real_mixed_densecheck/qwen35_persistent_serving_policy_compare.md)
- [non-M0 cousin bundle](../benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_repo_promptfiles_public_validation_round2_cousins_cuda_stage9_non_m0_densecheck/qwen35_persistent_serving_policy_compare.md)
- [conservative cousin bundle](../benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_repo_promptfiles_public_validation_round2_cousins_cuda_conservative_priority_value_hybrid_ci16_densecheck/qwen35_persistent_serving_policy_compare.md)

`submission_execution_plan` now matches dense on CUDA in all three lanes:

- dense:
  - `[12, 42466, 22449, 17587, 271, 26044, 261, 15705]`
- real mixed:
  - `[12, 42466, 22449, 17587, 271, 26044, 261, 15705]`
- non-M0:
  - `[12, 42466, 22449, 17587, 271, 26044, 261, 15705]`
- conservative:
  - `[12, 42466, 22449, 17587, 271, 26044, 261, 15705]`

Exact-match flags:

- real mixed vs dense: `true`
- non-M0 vs dense: `true`
- conservative vs dense: `true`
- all three lanes vs hand: `true`

The broader cousin sweep is clean on fixed-tree CUDA, which matches the post-fix MPS read.

## CUDA residual `performance_journal` hotspot

Artifacts:

- [fixed-tree detailed CUDA hotspot bundle](../benchmarks/results/qwen35_persistent_real_mixed_probe_20260413_repo_promptfiles_public_validation_round2_suspects_cuda_detailed_postfix/qwen35_persistent_real_mixed_probe.md)
- [earlier CUDA tie-boundary localization](qwen35_stage9_performance_journal_residual_cuda_20260413.md)

The fixed-tree detailed CUDA probe still shows the same real-mixed execution shape on `performance_journal`:

- executed M0 blocks: `6192`
- executed all-M3 blocks: `0`
- executed exact-key M3 blocks: `8`

Bias-lane phase totals for `performance_journal`:

- direct-M0 query-prep ms/case: `60.5492`
- direct-M0 gather ms/case: `81.5419`
- direct-M0 score ms/case: `149.9468`
- exact-key M3 score ms/case: `1.0221`
- aux exact-M3 score ms/case: `0.0000`
- final-mix ms/case: `238.8064`
- final-mix logits ms/case: `94.2229`
- final-mix softmax ms/case: `87.6131`
- final-mix value ms/case: `55.8494`

So the residual hotspot is still `final_mix`, followed by `direct_m0_score`, then `direct_m0_gather`, then `query_prep`.

Per-token hotspot window:

- the disputed position is generated step `6` (the seventh generated token)
- the earlier direct CUDA localization showed:
  - `logit[15] = 20.625`
  - `logit[16] = 20.625`
- that is an exact tie at stored CUDA logit precision, so `15`/`16` is a backend-sensitive tie-boundary rather than a structural serving-path divergence

Compared with the pre-fix CUDA note, the structural `state_cache_roadmap` family is gone. Compared with the post-fix MPS note, CUDA now tells the same main story: `state_cache_roadmap` and `submission_execution_plan` are clean, and the only residual public behavior is the late `performance_journal` tie boundary.

## Conclusion

shared DotCache-family boundary vs dense is structural; no Stage 9 mixed-only correctness blocker remains.
