# Qwen3.5 Stage 9 Fixed-Tree CUDA Native `final_mix` Pass (2026-04-13)

This pass returns CUDA work to performance after the fixed-tree handoff correctness fix.

Kept change:
- default-enable the existing native CUDA `softmax_value_context_cuda` path for mixed-mode `final_mix`
- explicit opt-out still works with `DOTCACHE_ENABLE_NATIVE_DIRECT_M0_FINAL_MIX=0`
- fallback behavior is unchanged when the native extension is unavailable or the score dtype is unsupported

Rejected change:
- default-enabling Triton direct-M0 scorer / fused scorer was tested and not kept
- on this tree it increased gather-side cost enough to wipe out the `final_mix` win

Code paths touched:
- `dotcache/backends/native_direct_m0.py`
- `dotcache/backends/metal/persistent_runtime.py`
- `dotcache/backends/triton_direct_m0.py`

Measurement artifacts:
- [performance_journal CUDA native-final-mix probe](/workspace/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_20260413_performance_journal_cuda_native_final_mix/qwen35_persistent_real_mixed_probe.md)
- [round-2 repo-local CUDA native-final-mix probe](/workspace/DotCache/benchmarks/results/qwen35_persistent_real_mixed_probe_20260413_repo_promptfiles_public_validation_round2_cuda_native_final_mix/qwen35_persistent_real_mixed_probe.md)

Note on the round-2 sweep:
- the branch manifest includes a macmini-only absolute path, so the CUDA sweep used the repo-local subset materialized from the same manifest records
- case count remained `8`

## `performance_journal`

Canonical `bias` readout, same tree/runtime, detailed timing on:

Before (`DOTCACHE_ENABLE_NATIVE_DIRECT_M0_FINAL_MIX=0`):
- avg ms/step: `405.1796`
- final_mix ms/case: `201.5492`
- direct_m0_score ms/case: `121.9503`
- exact-match vs hand: `1.0`

After (native CUDA `final_mix` default-on):
- avg ms/step: `397.1591`
- final_mix ms/case: `194.6770`
- direct_m0_score ms/case: `118.1439`
- exact-match vs hand: `1.0`

Delta:
- avg ms/step: `-8.0205` (`-1.98%`)
- final_mix ms/case: `-6.8722` (`-3.41%`)
- direct_m0_score ms/case: `-3.8064` (`-3.12%`)

## Round-2 Public Validation

Repo-local round-2 subset, same tree/runtime, detailed timing on.

Canonical `bias` readout:

Before (`DOTCACHE_ENABLE_NATIVE_DIRECT_M0_FINAL_MIX=0`):
- avg ms/step: `311.0682`
- final_mix ms/case: `144.5604`
- direct_m0_score ms/case: `87.2179`
- exact-match vs hand: `1.0`

After (native CUDA `final_mix` default-on):
- avg ms/step: `308.7841`
- final_mix ms/case: `143.2065`
- direct_m0_score ms/case: `86.6854`
- exact-match vs hand: `1.0`

Delta:
- avg ms/step: `-2.2841` (`-0.73%`)
- final_mix ms/case: `-1.3539` (`-0.94%`)
- direct_m0_score ms/case: `-0.5326` (`-0.61%`)

## Read

The kept gain comes from promoting the already-existing native CUDA fused softmax/value reduction for `final_mix` into the default supported path on fixed-tree CUDA. That reduces the `final_mix` bucket directly and also trims a smaller amount of end-to-end time from `direct_m0_score` on these probes, likely by reducing downstream pressure rather than by changing the scorer kernel itself.

Exact-match behavior did not change on either probe:
- `performance_journal`: `1.0` vs hand
- round-2 repo-local public-validation subset: `1.0` vs hand

Conclusion:
- correctness stayed unchanged
- `final_mix` improved cleanly
- no separate scorer-side kernel promotion was kept because the Triton scorer attempt regressed this tree
