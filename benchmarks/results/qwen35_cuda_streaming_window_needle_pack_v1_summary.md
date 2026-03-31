# Qwen3.5 CUDA Needle Pack Summary

| Case | Context | n prompts | Retrieval accuracy | Exact-match rate | Mean decode ms/step | Min | Max | Stddev | 95% CI | Mean selected pages | Decode path |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exact | 32768 | 4 | 1.00 | 0.75 | 2521.60 | 2209.45 | 2983.81 | 377.87 | 370.31 | 0.00 | {"grouped_batched": 0, "per_kv_fallback": 72} |
| exact | 49152 | 4 | 1.00 | 0.75 | 3909.29 | 3515.58 | 4351.92 | 449.30 | 440.31 | 0.00 | {"grouped_batched": 0, "per_kv_fallback": 72} |
| shortlist_base | 32768 | 4 | 1.00 | 1.00 | 474.23 | 451.71 | 500.95 | 24.24 | 23.75 | 12240.00 | {"grouped_batched": 0, "per_kv_fallback": 72} |
| shortlist_base | 49152 | 4 | 1.00 | 0.75 | 629.46 | 602.98 | 660.41 | 28.76 | 28.19 | 12240.00 | {"grouped_batched": 0, "per_kv_fallback": 72} |
| shortlist_l23_ctx | 32768 | 4 | 1.00 | 1.00 | 492.86 | 469.68 | 531.77 | 27.60 | 27.05 | 12336.00 | {"grouped_batched": 0, "per_kv_fallback": 72} |
| shortlist_l23_ctx | 49152 | 4 | 1.00 | 0.75 | 636.96 | 609.14 | 668.82 | 24.63 | 24.14 | 12336.00 | {"grouped_batched": 0, "per_kv_fallback": 72} |
| streaming_sink_recent | 32768 | 4 | 0.00 | 0.00 | 156.65 | 142.34 | 183.47 | 18.69 | 18.31 | 11664.00 | {"grouped_batched": 0, "per_kv_fallback": 72} |
| streaming_sink_recent | 49152 | 4 | 0.00 | 0.00 | 188.55 | 185.75 | 192.83 | 3.27 | 3.21 | 11664.00 | {"grouped_batched": 0, "per_kv_fallback": 72} |
