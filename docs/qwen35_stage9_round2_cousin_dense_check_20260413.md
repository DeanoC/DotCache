# Qwen3.5 Stage 9 Round 2 Cousin Dense Check (2026-04-13)

This note records a small follow-up dense check on four files adjacent to the original round-2 suspect cases.

Cousin manifest:

- [benchmarks/manifests/qwen35_stage9_repo_public_validation_round2_cousins_20260413.json](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/manifests/qwen35_stage9_repo_public_validation_round2_cousins_20260413.json)

Targeted real-mixed dense-check bundle:

- [benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_repo_promptfiles_public_validation_round2_cousins_mps_real_mixed_densecheck/qwen35_persistent_serving_policy_compare.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_repo_promptfiles_public_validation_round2_cousins_mps_real_mixed_densecheck/qwen35_persistent_serving_policy_compare.md)

## Cases checked

- `statecache_showcase`
- `gemma4_apple_compatibility`
- `model_roadmap`
- `submission_execution_plan`

## Result

Three of the four cousin cases stay clean against dense:

- `statecache_showcase`
  - `bias_matches_dense_exact = true`
- `gemma4_apple_compatibility`
  - `bias_matches_dense_exact = true`
- `model_roadmap`
  - `bias_matches_dense_exact = true`

One cousin joins the dense-divergence side:

- `submission_execution_plan`
  - `bias_matches_dense_exact = false`
  - `bias_matches_hand_tuned_exact = true`
  - dense prefix match length `5`

Generated IDs on `submission_execution_plan`:

- dense:
  - `[12, 42466, 22449, 17587, 271, 26044, 261, 15705]`
- real mixed / hand:
  - `[12, 42466, 22449, 17587, 271, 50215, 9375, 15705]`

## Interpretation

This helps narrow the boundary:

- the divergence is not a broad "all roadmap/state-cache style docs fail" pattern
- it is also not isolated to only the original two files
- the current boundary looks like a small structured-text family that now includes:
  - `performance_journal`
  - `state_cache_roadmap`
  - `submission_execution_plan`

That makes `submission_execution_plan` a useful third repro:

- it diverges later than `state_cache_roadmap`
- but earlier enough to be informative
- and it stays in the same "matches hand, diverges from dense" family
