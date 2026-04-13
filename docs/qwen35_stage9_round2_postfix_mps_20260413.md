# Qwen3.5 Stage 9 Round 2 Post-Fix MPS Read (2026-04-13)

This note records the MPS reruns after the logical-sequence-length handoff fix from:

- [qwen35_stage9_state_cache_roadmap_root_cause_20260413.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/docs/qwen35_stage9_state_cache_roadmap_root_cause_20260413.md)

## Refreshed suspect bundle

- [benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_repo_promptfiles_public_validation_round2_suspects_mps_real_mixed_densecheck_postfix/qwen35_persistent_serving_policy_compare.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_repo_promptfiles_public_validation_round2_suspects_mps_real_mixed_densecheck_postfix/qwen35_persistent_serving_policy_compare.md)

## Refreshed cousin bundle

- [benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_repo_promptfiles_public_validation_round2_cousins_mps_real_mixed_densecheck_postfix/qwen35_persistent_serving_policy_compare.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_repo_promptfiles_public_validation_round2_cousins_mps_real_mixed_densecheck_postfix/qwen35_persistent_serving_policy_compare.md)

## Main result

The handoff fix eliminates the previously important public divergence family on MPS.

Now:

- `state_cache_roadmap` matches dense exactly again
- `submission_execution_plan` matches dense exactly again
- all four cousin cases match dense exactly

The remaining outlier in these MPS reruns is:

- `performance_journal`

## `state_cache_roadmap`

After the fix:

- dense:
  - `[12, 264, 11782, 314, 279, 1118, 220, 16]`
- real mixed / hand:
  - `[12, 264, 11782, 314, 279, 1118, 220, 16]`
- dense prefix match length:
  - `8`

So the canonical early-divergence repro is resolved by the logical-sequence-length fix.

## `submission_execution_plan`

After the fix:

- dense:
  - `[12, 42466, 22449, 17587, 271, 26044, 261, 15705]`
- real mixed / hand:
  - `[12, 42466, 22449, 17587, 271, 26044, 261, 15705]`
- dense prefix match length:
  - `8`

So the third member of the earlier divergence family is also resolved by the same fix.

## Remaining outlier

`performance_journal` still differs on this MPS rerun, but it is now much narrower:

- dense:
  - `[198, 220, 471, 1510, 77518, 28, 16, 7561]`
- real mixed / hand:
  - `[198, 220, 471, 1510, 77518, 28, 15, 7561]`
- dense prefix match length:
  - `6`

This is qualitatively different from the original family:

- the earlier structural repros are now fixed
- the remaining difference is late and narrow
- it also matches the current CUDA dense tail better than the earlier MPS dense tail
- this residual is now best interpreted as a tie-boundary class when viewed from fixed-tree CUDA where dense/serving agree on `[198, 220, 471, 1510, 77518, 28, 15, 7561]`

## Interpretation

Best current read:

- the logical-sequence-length handoff bug was the main public round-2 correctness problem
- fixing it collapses the original divergence family
- `performance_journal` remains as a smaller residual backend-sensitive or dense-alignment issue, not evidence against Stage 9 mixed execution itself
