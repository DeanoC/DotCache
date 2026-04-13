# Qwen3.5 Stage 9 Public Validation (2026-04-12)

This note records a broader repo-local public validation pass on the current MPS runtime.

Validation manifest:

- [benchmarks/manifests/qwen35_stage9_repo_public_validation_20260412.json](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/manifests/qwen35_stage9_repo_public_validation_20260412.json)

The manifest deliberately mixes:

- roadmap / planning docs
- backend comparison notes
- CUDA paper-table style notes
- code-heavy benchmark source files

## Checked-in bundles

### Real mixed

- [benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_repo_promptfiles_public_validation_mps/qwen35_persistent_real_mixed_probe.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_repo_promptfiles_public_validation_mps/qwen35_persistent_real_mixed_probe.md)
  - bias `552.24 ms/step`
  - hand `611.64`
  - exact-match `1.0`

### Non-`M0` Stage 9

- [benchmarks/results/qwen35_persistent_serving_policy_compare_20260412_repo_promptfiles_public_validation_mps_stage9_non_m0/qwen35_persistent_serving_policy_compare.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_serving_policy_compare_20260412_repo_promptfiles_public_validation_mps_stage9_non_m0/qwen35_persistent_serving_policy_compare.md)
  - bias `2273.01 ms/step`
  - hand `2330.83`
  - exact-match vs hand `1.0`

### Conservative certified

- [benchmarks/results/qwen35_persistent_serving_policy_compare_20260412_repo_promptfiles_public_validation_mps_conservative_priority_value_hybrid_ci16/qwen35_persistent_serving_policy_compare.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_serving_policy_compare_20260412_repo_promptfiles_public_validation_mps_conservative_priority_value_hybrid_ci16/qwen35_persistent_serving_policy_compare.md)
  - bias `1434.78 ms/step`
  - hand `1471.18`
  - exact-match vs hand `1.0`

## Main read

The current Stage 9 winner ordering survives this broader public-validation mix on MPS:

- real mixed `552.24`
- conservative certified `1434.78`
- non-`M0` Stage 9 `2273.01`

That means real mixed is:

- about `61.5%` faster than conservative certified
- about `75.7%` faster than non-`M0` Stage 9

### CUDA same-tree public-validation

- [benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_public_validation/qwen35_persistent_real_mixed_probe.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_public_validation/qwen35_persistent_real_mixed_probe.md)
  - bias `327.70 ms/step`
- [benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_public_validation_stage9_non_m0/qwen35_persistent_serving_policy_compare.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_public_validation_stage9_non_m0/qwen35_persistent_serving_policy_compare.md)
  - bias `339.66 ms/step`
- [benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_public_validation_conservative_priority_value_hybrid_ci16/qwen35_persistent_serving_policy_compare.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_public_validation_conservative_priority_value_hybrid_ci16/qwen35_persistent_serving_policy_compare.md)
  - bias `333.35 ms/step`

On CUDA same-tree public-validation:

- real mixed: `327.70 ms/step`
- conservative certified: `333.35 ms/step`
- non-`M0` Stage 9: `339.66 ms/step`
- real mixed still wins, but margins are tighter than on large/broad/external
- exact-match vs hand remains `1.0`

Margin to call out:

- real mixed is `1.69%` faster than conservative certified
- real mixed is `3.52%` faster than non-`M0` Stage 9

Observed caveat:

- same-tree CUDA conservative and non-`M0` runs on this branch/runtime were checkpoint-light (`optional_selection=0`, `diverse_selection=0`) and do not recreate the older checkpoint-heavy behavior of earlier CUDA conservative checks.

## Per-case read

Real mixed `bias` beat `hand` on all `6/6` cases and preserved exact-match throughout:

- `cuda_shortlist_paper_table`: `657.80` vs `766.69`
- `model_roadmap`: `531.68` vs `584.70`
- `real_mixed_probe_code`: `530.03` vs `552.64`
- `serving_policy_compare_code`: `654.18` vs `680.47`
- `stage9_backend_comparison`: `404.14` vs `513.36`
- `submission_execution_plan`: `535.60` vs `571.96`

## Interpretation

This is useful because it is not just another long-doc pack. The validation set mixes different public repo-local prompt styles, and the same high-level conclusion still holds:

- the current real-mixed Stage 9 path remains the clear MPS serving winner
- the result is not limited to the earlier large / broad / external repo-local manifests
- the confidence case for the merged Stage 9 checkpoint is stronger than it was before this run

## Repeatability

Two additional real-mixed reruns on the same public-validation manifest came out at:

- `591.98 ms/step`
- `590.81`

Together with the checked-in run:

- `552.24`
- `591.98`
- `590.81`

That gives:

- mean `578.34 ms/step`
- population stdev `18.46`

So the broader public-validation result is not just a one-off win. The current real-mixed path still holds a large margin over the checked-in non-`M0` and conservative baselines across the observed repeat spread.
