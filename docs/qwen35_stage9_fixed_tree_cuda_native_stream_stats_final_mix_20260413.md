# Qwen3.5 Stage 9 Fixed-Tree CUDA Native Stream-Stats `final_mix` (2026-04-13)

This pass retargets the dominant remaining CUDA mixed-runtime bucket after the fused-score baseline:

- keep native CUDA `final_mix` default-on
- keep fused query-first combined-cache `direct_m0_score` default-on
- add a native CUDA fallback-gated kernel for the generic mixed `collect_stream_stats` path:
  - inputs: `logits`, `token_block_ids`, `values`
  - outputs: `h`, `m`, `l`, per-block max logits, per-block mass numerators
- leave the old PyTorch path intact as the fallback when shapes do not fit the native kernel limits

Code paths touched:

- `dotcache/backends/native_direct_m0.py`
- `dotcache/backends/cuda_kernels/native_direct_m0.cpp`
- `dotcache/backends/cuda_kernels/native_direct_m0_kernel.cu`
- `dotcache/backends/metal/persistent_runtime.py`
- `tests/test_triton_direct_m0.py`

Measurement artifacts:

- [performance_journal CUDA native-stream-stats probe](/workspace/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_20260413_performance_journal_cuda_native_stream_stats_final_mix/qwen35_persistent_real_mixed_probe.md)
- [round-2 repo-local CUDA native-stream-stats probe](/workspace/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_20260413_repo_promptfiles_public_validation_round2_cuda_native_stream_stats_final_mix/qwen35_persistent_real_mixed_probe.md)
- [prior fused-score baseline note](qwen35_stage9_fixed_tree_cuda_fused_direct_m0_score_20260413.md)

## `performance_journal`

Before (fixed-tree fused-score baseline, bias lane):

- avg ms/step: `395.5052`
- direct_m0_gather ms/case: `68.0543`
- direct_m0_score ms/case: `82.8551`
- final_mix ms/case: `198.9716`
- exact-match vs hand: `1.0`

After (native stream-stats `final_mix`, bias lane):

- avg ms/step: `384.9285`
- direct_m0_gather ms/case: `68.9621`
- direct_m0_score ms/case: `83.9364`
- final_mix ms/case: `103.5070`
- exact-match vs hand: `1.0`

Delta:

- avg ms/step: `-10.5767` (`-2.67%`)
- direct_m0_gather ms/case: `+0.9078` (`+1.33%`)
- direct_m0_score ms/case: `+1.0813` (`+1.30%`)
- final_mix ms/case: `-95.4646` (`-47.98%`)

Read:

- the remaining hotspot really was the generic mixed `final_mix` bookkeeping path
- moving the rowwise softmax plus block-stats accumulation into one CUDA kernel cuts the mixed `final_mix` cost almost in half on this case
- small gather/score regressions remain, but they are much smaller than the `final_mix` win

## Round-2 Public Validation

Before (fixed-tree fused-score baseline, bias lane):

- avg ms/step: `306.5711`
- direct_m0_gather ms/case: `50.1207`
- direct_m0_score ms/case: `60.3289`
- final_mix ms/case: `144.9095`
- exact-match vs hand: `1.0`

After (native stream-stats `final_mix`, bias lane):

- avg ms/step: `304.2026`
- direct_m0_gather ms/case: `52.0347`
- direct_m0_score ms/case: `62.1430`
- final_mix ms/case: `76.5842`
- exact-match vs hand: `1.0`

Delta:

- avg ms/step: `-2.3685` (`-0.77%`)
- direct_m0_gather ms/case: `+1.9140` (`+3.82%`)
- direct_m0_score ms/case: `+1.8141` (`+3.01%`)
- final_mix ms/case: `-68.3253` (`-47.15%`)

Read:

- the repo-local round-2 public-validation mix also improves end-to-end
- exact-match behavior stays unchanged
- the win generalizes beyond the single `performance_journal` probe even though the non-`final_mix` buckets rise slightly

## Verification

- `.venv/bin/python -m py_compile dotcache/backends/native_direct_m0.py dotcache/backends/metal/persistent_runtime.py tests/test_triton_direct_m0.py`
- `.venv/bin/python -m pytest tests/test_triton_direct_m0.py -k "softmax_value_context or softmax_value_stream_stats"`
- `benchmarks/bench_qwen35_persistent_real_mixed_cached_probe.py` on the `performance_journal` 1344-token mixed prefix
- `benchmarks/bench_qwen35_persistent_real_mixed_probe.py` on:
  - single-case `performance_journal`
  - `benchmarks/manifests/qwen35_stage9_repo_public_validation_round2_20260413.json`

## Conclusion

The kept gain comes from replacing the generic mixed `final_mix` PyTorch bookkeeping path with a fallback-gated native CUDA kernel that computes context and stream stats together. On fixed-tree CUDA, that lowers real-mixed end-to-end latency by about `2.67%` on `performance_journal` and about `0.77%` on the round-2 repo-local public-validation pack, while keeping exact-match vs hand at `1.0`.
