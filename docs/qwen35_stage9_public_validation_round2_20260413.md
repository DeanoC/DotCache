# Qwen3.5 Stage 9 Public Validation Round 2 (2026-04-13)

This note records a second, broader repo-local public validation pass on the current MPS runtime.

Validation manifest:

- [benchmarks/manifests/qwen35_stage9_repo_public_validation_round2_20260413.json](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/manifests/qwen35_stage9_repo_public_validation_round2_20260413.json)

This corpus is deliberately different from the first public-validation pack. It mixes:

- evaluation and performance notes
- compatibility and showcase docs
- roadmap material
- benchmark source files
- test code

## Checked-in bundles

### Real mixed

- [benchmarks/results/qwen35_persistent_real_mixed_probe_20260413_repo_promptfiles_public_validation_round2_mps/qwen35_persistent_real_mixed_probe.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_20260413_repo_promptfiles_public_validation_round2_mps/qwen35_persistent_real_mixed_probe.md)
  - bias `571.68 ms/step`
  - hand `682.63`
  - exact-match vs hand `1.0`

### Non-`M0` Stage 9

- [benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_repo_promptfiles_public_validation_round2_mps_stage9_non_m0/qwen35_persistent_serving_policy_compare.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_repo_promptfiles_public_validation_round2_mps_stage9_non_m0/qwen35_persistent_serving_policy_compare.md)
  - bias `1984.42 ms/step`
  - hand `2108.42`
  - bias vs dense exact-match `0.75`
  - bias vs hand exact-match `1.0`

### Conservative certified

- [benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_repo_promptfiles_public_validation_round2_mps_conservative_priority_value_hybrid_ci16/qwen35_persistent_serving_policy_compare.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_serving_policy_compare_20260413_repo_promptfiles_public_validation_round2_mps_conservative_priority_value_hybrid_ci16/qwen35_persistent_serving_policy_compare.md)
  - bias `1387.69 ms/step`
  - hand `1524.73`
  - bias vs dense exact-match `0.75`
  - bias vs hand exact-match `1.0`

## Main read

The winner ordering still holds on this broader second corpus:

- real mixed `571.68`
- conservative certified `1387.69`
- non-`M0` Stage 9 `1984.42`

That means real mixed is:

- about `58.8%` faster than conservative certified
- about `71.2%` faster than non-`M0` Stage 9

## What is new here

This is not just another validation win. It also exposed a more interesting boundary:

- real mixed still matched hand exactly across the whole corpus
- non-`M0` and conservative both matched hand exactly
- but both of those lanes only matched dense on `75%` of cases

So round 2 is useful in two ways:

- it strengthens the serving-win claim for real mixed on a broader public corpus
- it also gives us a sharper stress set for studying where the broader Stage 9 family starts to diverge from dense

## Interpretation

This second public-validation corpus is a better confidence test than the first one because it is less self-referential to the earlier Stage 9 writeups and includes more varied repo-local material.

The current MPS takeaway is:

- real mixed remains the strongest serving lane
- conservative remains the middle lane
- non-`M0` remains the slowest of the three
- this corpus should be carried over to CUDA, because it is now a more informative boundary test than round 1 alone
