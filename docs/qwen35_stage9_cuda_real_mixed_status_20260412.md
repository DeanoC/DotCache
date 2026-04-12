## Qwen3.5 Stage 9 CUDA Real-Mixed Status (2026-04-12)

Branch: `codex/qwen35-certified-streaming`

Portable-manifest CUDA result bundles:

- `benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_large_cuda_frontier_batchedresidual_v6/`
- `benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_broad_cuda_frontier_batchedresidual_v6/`
- `benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_external_cuda_frontier_batchedresidual_v6/`

Current best CUDA real-mixed `bias` results on the repo-local portable corpus:

- `large`: `399.74 ms/step`
- `broad`: `460.70 ms/step`
- `external`: `252.42 ms/step`
- exact-match rate: `1.0` on all three

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

- CUDA real-mixed `direct_m0` is now the preferred Stage 9 policy for `large` and `broad` on the portable corpus.
- `external` improved substantially but remains the hardest case.
- On the older CUDA comparison numbers used during bring-up:
  - `large` real-mixed now beats the old non-`M0` reference (`468.69`)
  - `broad` real-mixed now beats the old non-`M0` reference (`632.12`)
  - `external` still trails the old non-`M0` reference (`192.54`)

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
- target the remaining `final_mix` reduction path on the `external` corpus first
- continue to keep MPS behavior unchanged
