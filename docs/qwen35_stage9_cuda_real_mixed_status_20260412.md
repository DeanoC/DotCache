## Qwen3.5 Stage 9 CUDA Real-Mixed Status (2026-04-12)

Branch: `codex/stage9-public-validation`

Portable-manifest CUDA real-mixed result bundles:

- `benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_large_cuda_frontier_batchedresidual_v18_clean/`
- `benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_broad_cuda_frontier_batchedresidual_v18_clean/`
- `benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_external_cuda_frontier_batchedresidual_v17_clean/`

Current best clean CUDA real-mixed `bias` results on the repo-local portable corpus:

- `large`: `374.06 ms/step`
- `broad`: `443.30 ms/step`
- `external`: `240.84 ms/step`
- exact-match rate: `1.0` on all three

Current-tree CUDA Stage 9 non-`M0` comparison bundles:

- `benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_large_stage9_non_m0_currenttree_v2/`
- `benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_broad_stage9_non_m0_currenttree_v2/`
- `benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_external_stage9_non_m0_currenttree/`

Current-tree CUDA non-`M0` `bias` results:

- `large`: `388.78 ms/step`
- `broad`: `465.40 ms/step`
- `external`: `244.02 ms/step`

CUDA same-tree public-validation bundles:

- `benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_public_validation/`
- `benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_public_validation_stage9_non_m0/`
- `benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_public_validation_conservative_priority_value_hybrid_ci16/`

Current-tree CUDA public-validation `bias` results:

- real-mixed: `327.70 ms/step`
- non-`M0`: `339.66 ms/step`
- conservative certified: `333.35 ms/step`

Observed result:

- real mixed is the same-tree winner for this broader public-validation set
- real mixed margins are narrower, but still ahead of non-`M0` and conservative
- exact-match rate remained `1.0` in the checked-in run
- this run is checkpoint-light for conservative/non-`M0` vs the older checkpoint-heavy CUDA check baseline and should be interpreted as same-tree policy-level comparison

Canonical config preserved:

- `bits_k=8`
- `bits_v=4`
- `tokens_per_page=16`
- `group_size=32`
- `mixed_execution_strategy=direct_m0`
- `allow_value_m0=False`
- `enable_early_exit=True`
- `full_attention_check_interval=16`
- `full_attention_streaming_order_mode=priority_value_hybrid`
- `full_attention_streaming_priority_value_upper_weight=0.25`
- `full_attention_key_centroid_count_by_layer={19: 8, 23: 16}`

What changed on CUDA:

- batched residual/certificate tracker setup and updates
- CUDA block-mass serving path kept on the fast path
- stream-stats path no longer materializes normalized token weights when attention weights are not requested
- tranche merge of `m/l/h` and block-mass accumulators is vectorized
- block-mass numerator accumulation is vectorized across heads

Current policy read:

- CUDA real-mixed `direct_m0` now wins against current-tree Stage 9 non-`M0` on all three portable corpora.
- Same-tree margins:
  - `large`: `374.06` vs `388.78`
  - `broad`: `443.30` vs `465.40`
  - `external`: `240.84` vs `244.02`
- The older bring-up non-`M0` numbers are no longer the right ship comparison for this branch.

Remaining hotspot:

- `final_mix` is still the largest mixed bucket on CUDA.
- External detailed timing reference:
  - `benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_external_cuda_frontier_batchedresidual_v5_detailed/`
  - bias `final_mix`: `109.83 ms/case`
  - logits: `42.60`
  - softmax: `40.88`
  - value: `25.89`

Recommended next step:

- keep the current CUDA larger-path changes
- target the remaining `final_mix` reduction path on the `external` corpus first if more headroom is needed
- continue to keep MPS behavior unchanged
