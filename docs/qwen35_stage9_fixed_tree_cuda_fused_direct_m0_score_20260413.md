# Qwen3.5 Stage 9 Fixed-Tree CUDA Fused `direct_m0_score` Pass (2026-04-13)

This pass stays on the fixed-tree CUDA baseline with native `final_mix` default-on and targets the next hot bucket: `direct_m0_score`.

Kept change:
- when the CUDA mixed-mode score cache already has the combined `[fused_scaled | bias]` tensor, score it with one query-first GEMM instead of separate score and bias GEMMs
- emit logits directly in `[query_count, token_count]` layout so the downstream `final_mix` path does not pay an extra transpose/layout tax
- keep the old split-cache scorer as the fallback path when the combined score cache is unavailable

Code paths touched:
- `dotcache/backends/torch_mps.py`
- `dotcache/backends/metal/persistent_runtime.py`

Measurement artifacts:
- [performance_journal CUDA fused-score probe](/workspace/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_20260413_performance_journal_cuda_fused_score_queryfirst/qwen35_persistent_real_mixed_probe.md)
- [round-2 repo-local CUDA fused-score probe](/workspace/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_20260413_repo_promptfiles_public_validation_round2_cuda_fused_score_queryfirst/qwen35_persistent_real_mixed_probe.md)
- [prior native-final-mix baseline note](qwen35_stage9_fixed_tree_cuda_native_final_mix_20260413.md)

## `performance_journal`

Before (current fixed-tree native-`final_mix` baseline):
- avg ms/step: `397.1591`
- direct_m0_gather ms/case: `68.4685`
- direct_m0_score ms/case: `118.1439`
- final_mix ms/case: `194.6770`
- exact-match vs hand: `1.0`

After (fused query-first combined-cache scorer):
- avg ms/step: `395.5052`
- direct_m0_gather ms/case: `68.0543`
- direct_m0_score ms/case: `82.8551`
- final_mix ms/case: `198.9716`
- exact-match vs hand: `1.0`

Delta:
- avg ms/step: `-1.6539` (`-0.42%`)
- direct_m0_gather ms/case: `-0.4142` (`-0.60%`)
- direct_m0_score ms/case: `-35.2887` (`-29.87%`)
- final_mix ms/case: `+4.2947` (`+2.21%`)

Read:
- the scorer win is real on the pure-`M0` public probe too
- `final_mix` rises slightly, but not enough to erase the score-side gain

## Round-2 Public Validation

Repo-local round-2 subset, same tree/runtime, detailed timing on.

Before (current fixed-tree native-`final_mix` baseline):
- avg ms/step: `308.7841`
- direct_m0_gather ms/case: `50.1982`
- direct_m0_score ms/case: `86.6854`
- final_mix ms/case: `143.2065`
- exact-match vs hand: `1.0`

After (fused query-first combined-cache scorer):
- avg ms/step: `306.5711`
- direct_m0_gather ms/case: `50.1207`
- direct_m0_score ms/case: `60.3289`
- final_mix ms/case: `144.9095`
- exact-match vs hand: `1.0`

Delta:
- avg ms/step: `-2.2130` (`-0.72%`)
- direct_m0_gather ms/case: `-0.0775` (`-0.15%`)
- direct_m0_score ms/case: `-26.3565` (`-30.40%`)
- final_mix ms/case: `+1.7030` (`+1.19%`)

Read:
- the round-2 public-validation bundle also improves end-to-end
- gather stays effectively flat
- exact-match behavior does not change

## Conclusion

The kept gain is not from a new kernel. It comes from using the already-materialized combined CUDA score cache more efficiently:
- old path: `score = Q * K_fused + group_sums * bias`
- new path: `score = [Q | group_sums] * [K_fused | bias]^T`

Doing that in query-first layout cuts `direct_m0_score` by about `30%` on both probes while keeping gather flat. `final_mix` is slightly more expensive than the native-final-mix baseline, but the score reduction still improves overall fixed-tree CUDA latency and keeps exact-match `1.0` vs hand on the public-validation probes.
