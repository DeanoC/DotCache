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

Update:

- broader public-corpus validation has now started with:
  - [docs/qwen35_stage9_public_validation_20260412.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/docs/qwen35_stage9_public_validation_20260412.md)
- cheap live exact-key fallback heuristics have now been tested too:
  - [benchmarks/results/qwen35_persistent_exact_key_live_policy_compare_20260412/qwen35_persistent_exact_key_live_policy_compare.md](/Users/deanocalver/.codex/worktrees/9f76/DotCache/benchmarks/results/qwen35_persistent_exact_key_live_policy_compare_20260412/qwen35_persistent_exact_key_live_policy_compare.md)
  - current baseline still wins in the live runtime, so no layer-15 heuristic is being promoted from that study

Additional same-tree CUDA read:

- public-validation corpus bundles now report:
  - real mixed `327.70 ms/step`
  - conservative certified `333.35 ms/step`
  - non-`M0` Stage 9 `339.66 ms/step`
- real mixed remains the same-tree winner on this broader public set

Residual tie-boundary classification:

- the fixed handoff path closes the substantive `performance_journal` and `state_cache_roadmap` structural divergence narrative
- on fixed-tree CUDA, `state_cache_roadmap` and `submission_execution_plan` now match dense in all three serving lanes
- the only remaining public residual is `performance_journal`, where dense / non-`M0` / conservative produce `[198, 220, 471, 1510, 77518, 28, 15, 7561]` and real mixed produces `[198, 220, 471, 1510, 77518, 28, 16, 7561]`
- the tied-step diagnostic shows this is not a `final_mix` bug:
  - dense and non-`M0` remain exactly tied at `15` vs `16`
  - real mixed nudges token `16` slightly higher before argmax
  - the first drift appears upstream at full-attention layer `3`
- this is now interpreted as tiny upstream mixed-path numeric drift before argmax, not a Stage 9 mixed-only correctness blocker

Current fixed-tree CUDA baseline:

- native CUDA `final_mix` is now default-on for supported mixed-mode calls
- fused query-first combined-cache `direct_m0_score` is now default-on when the combined cache is available
- native CUDA generic mixed stream-stats `final_mix` is now default-on when shape limits fit
- Triton scorer / fused paths remain opt-in only

Measured kept gains on the fixed-tree CUDA performance probes:

- `performance_journal`
  - native `final_mix`: `405.18 -> 397.16 ms/step`
  - fused query-first scorer: `397.16 -> 395.51 ms/step`
  - native stream-stats `final_mix`: `395.51 -> 384.93 ms/step`
- round-2 repo-local public-validation subset
  - native `final_mix`: `311.07 -> 308.78 ms/step`
  - fused query-first scorer: `308.78 -> 306.57 ms/step`
  - native stream-stats `final_mix`: `306.57 -> 304.20 ms/step`

The newest kept CUDA win is the most important one:

- it cuts generic mixed `final_mix` cost by about `48%` on both checked probes
- end-to-end latency improves by about `2.67%` on `performance_journal`
- end-to-end latency improves by about `0.77%` on the round-2 repo-local public-validation subset
- exact-match vs hand stays `1.0`

So the current CUDA checkpoint is no longer just "correct and viable." It now has a meaningful fixed-tree default baseline improvement, with the strongest recent gain coming from the generic mixed stream-stats `final_mix` kernel.
