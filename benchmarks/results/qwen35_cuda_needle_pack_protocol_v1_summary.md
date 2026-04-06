# Qwen3.5 CUDA Needle Pack Summary

| Case | Context | n prompts | Retrieval accuracy | Exact-match rate | Mean decode ms/step | Min | Max | Stddev | 95% CI | Mean selected pages | Decode path |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exact | 32768 | 4 | 1.00 | 0.75 | 2496.10 | 2206.20 | 2875.22 | 323.30 | 316.83 | 0.00 | {"grouped_batched": 0, "per_kv_fallback": 72} |
| exact | 49152 | 4 | 1.00 | 0.75 | 3966.83 | 3561.54 | 4462.69 | 443.41 | 434.54 | 0.00 | {"grouped_batched": 0, "per_kv_fallback": 72} |
| shortlist_base | 32768 | 4 | 1.00 | 1.00 | 561.51 | 462.66 | 781.12 | 147.68 | 144.72 | 12240.00 | {"grouped_batched": 0, "per_kv_fallback": 72} |
| shortlist_base | 49152 | 4 | 1.00 | 0.75 | 759.82 | 628.87 | 1025.77 | 180.16 | 176.56 | 12240.00 | {"grouped_batched": 0, "per_kv_fallback": 72} |
| shortlist_l23_ctx | 32768 | 4 | 1.00 | 1.00 | 509.53 | 477.29 | 536.90 | 26.58 | 26.05 | 12336.00 | {"grouped_batched": 0, "per_kv_fallback": 72} |
| shortlist_l23_ctx | 49152 | 4 | 1.00 | 0.75 | 641.14 | 623.21 | 679.30 | 25.71 | 25.20 | 12336.00 | {"grouped_batched": 0, "per_kv_fallback": 72} |
