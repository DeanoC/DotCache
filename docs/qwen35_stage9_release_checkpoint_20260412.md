# Qwen3.5 Stage 9 Release Checkpoint (2026-04-12)

This note names the current canonical Stage 9 checkpoint for the `codex/qwen35-certified-streaming` branch.

## Release read

The core Stage 9 thesis is now supported on both Apple MPS and CUDA for the `Qwen/Qwen3.5-0.8B` portable corpus set:

- certified streaming is active in the real serving loop
- key-side mixed `M0` execution is active
- exact-match stays `1.0` on the checked-in real-mixed bundles
- real mixed `bias` is the serving winner on the current checked-in portable corpora for both backends

The generated comparison matrix is the canonical summary artifact:

- [benchmarks/results/qwen35_stage9_backend_matrix_20260412/qwen35_stage9_backend_matrix.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_stage9_backend_matrix_20260412/qwen35_stage9_backend_matrix.md)

## Canonical backend policy

- MPS:
  - prefer real mixed Stage 9 `bias`
- CUDA:
  - prefer real mixed Stage 9 `bias`

This recommendation is for the current `0.8B` portable corpus set. It does not imply that every larger model or future corpus will keep the same winner ordering without rerunning the matrix.

## Canonical benchmark bundles

### MPS real mixed

- large:
  - [benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_repo_promptfiles_large_mps_refreshed/qwen35_persistent_real_mixed_probe.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_repo_promptfiles_large_mps_refreshed/qwen35_persistent_real_mixed_probe.md)
  - bias `1407.44 ms/step`
- broad:
  - [benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_repo_promptfiles_broad_mps_refreshed/qwen35_persistent_real_mixed_probe.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_repo_promptfiles_broad_mps_refreshed/qwen35_persistent_real_mixed_probe.md)
  - bias `1627.72 ms/step`
- external:
  - [benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_repo_promptfiles_external_mps_refreshed/qwen35_persistent_real_mixed_probe.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_repo_promptfiles_external_mps_refreshed/qwen35_persistent_real_mixed_probe.md)
  - bias `843.77 ms/step`

### CUDA real mixed

- large:
  - [benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_large_cuda_frontier_batchedresidual_v18_clean/qwen35_persistent_real_mixed_probe.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_large_cuda_frontier_batchedresidual_v18_clean/qwen35_persistent_real_mixed_probe.md)
  - bias `374.06 ms/step`
- broad:
  - [benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_broad_cuda_frontier_batchedresidual_v18_clean/qwen35_persistent_real_mixed_probe.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_broad_cuda_frontier_batchedresidual_v18_clean/qwen35_persistent_real_mixed_probe.md)
  - bias `443.30 ms/step`
- external:
  - [benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_external_cuda_frontier_batchedresidual_v17_clean/qwen35_persistent_real_mixed_probe.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_external_cuda_frontier_batchedresidual_v17_clean/qwen35_persistent_real_mixed_probe.md)
  - bias `240.84 ms/step`

## Canonical comparison baselines

### MPS

- non-`M0` Stage 9:
  - large `3784.86`
  - broad `4234.40`
  - external `1700.51`
- conservative certified:
  - large `2054.42`
  - broad `2828.78`
  - external `1340.05`

### CUDA

- non-`M0` Stage 9:
  - large `388.78`
  - broad `465.40`
  - external `244.02`
- conservative certified:
  - large `652.91`
  - broad `792.60`
  - external `380.85`

## Scope boundaries

- value-side `M0` is still off
- learned ordering/scoring is still out of scope
- `Qwen/Qwen3.5-4B` on this MPS host is feasibility-only because paging makes it an impractical benchmark lane

## Next research directions

- broaden public-corpus validation
- refine the exact-key fallback policy
- investigate value-side `M0` only after the current cross-backend checkpoint is stable
