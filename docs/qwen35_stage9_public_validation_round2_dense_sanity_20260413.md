# Qwen3.5 Stage 9 Round 2 Dense Sanity Check (2026-04-13)

This note records a targeted dense-comparison sanity pass on the two round-2 public-validation cases that diverged from dense in the non-`M0` and conservative lanes.

Suspect-case manifest:

- [benchmarks/manifests/qwen35_stage9_repo_public_validation_round2_suspects_20260413.json](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/manifests/qwen35_stage9_repo_public_validation_round2_suspects_20260413.json)

Targeted real-mixed dense-check bundle:

- [benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_repo_promptfiles_public_validation_round2_suspects_mps_real_mixed_densecheck/qwen35_persistent_serving_policy_compare.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_repo_promptfiles_public_validation_round2_suspects_mps_real_mixed_densecheck/qwen35_persistent_serving_policy_compare.md)

## Cases checked

- `performance_journal`
- `state_cache_roadmap`

## Main result

This was the pre-fix baseline for the round-2 suspects on MPS:

- real-mixed Stage 9 diverged from dense on both cases
- hand still matched real mixed

Post-fix status is in [qwen35_stage9_round2_postfix_mps_20260413.md](./qwen35_stage9_round2_postfix_mps_20260413.md), where the handoff fix removed the structural divergence and narrowed `performance_journal` to a late tie-boundary effect.

### `performance_journal`

- dense generated IDs:
  - `[198, 220, 471, 1510, 77518, 28, 16, 7561]`
- real mixed `bias` generated IDs:
  - `[198, 220, 471, 1510, 8412, 1551, 7408, 63]`
- real mixed:
  - `bias_matches_dense_exact = false`
  - `bias_matches_hand_tuned_exact = true`
- hand:
  - `hand_tuned_matches_dense_exact = false`

### `state_cache_roadmap`

- dense generated IDs:
  - `[12, 264, 11782, 314, 279, 1118, 220, 16]`
- real mixed `bias` generated IDs:
  - `[12, 1118, 78361, 321, 1118, 7652, 29642, 364]`
- real mixed:
  - `bias_matches_dense_exact = false`
  - `bias_matches_hand_tuned_exact = true`
- hand:
  - `hand_tuned_matches_dense_exact = false`

## Post-fix read

On the MPS post-fix suite, both:

- `performance_journal`
- `state_cache_roadmap`

now match dense on the serving-family lanes.

For `performance_journal`, the remaining difference is the late tail tie class already documented as CUDA-aligned:

- CUDA/serving: `[198, 220, 471, 1510, 77518, 28, 15, 7561]`
- MPS dense/serving: `[198, 220, 471, 1510, 77518, 28, 16, 7561]`

## Interpretation

This is the useful outcome of the sanity pass:

- the original divergence on these two cases is now understood as a fixed-tree handoff sequence-length issue plus tie-tail alignment noise
- the shared serving-family behavior remains a DotCache-family boundary against dense rather than a Stage 9-only regression

That does not prove the mainline path is correct. It means the next debugging step should be framed as:

- "why does the broader DotCache serving family diverge from dense on these two cases?"

rather than:

- "why does Stage 9 real mixed diverge while the rest stays clean?"
