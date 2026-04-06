# Qwen3.5 CUDA Passkey Pack Summary

| Case | Context | n prompts | Retrieval accuracy | Exact-match rate | Mean decode ms/step | Min | Max | Stddev | 95% CI | Mean selected pages | Decode path |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| exact | 32768 | 4 | 1.00 | 0.25 | 2390.28 | 2246.44 | 2693.28 | 204.49 | 200.40 | 0.00 | {"grouped_batched": 0, "per_kv_fallback": 72} |
| exact | 49152 | 4 | 1.00 | 0.25 | 3823.87 | 3517.13 | 4396.27 | 394.54 | 386.65 | 0.00 | {"grouped_batched": 0, "per_kv_fallback": 72} |
| shortlist_base | 32768 | 4 | 1.00 | 0.25 | 500.58 | 462.95 | 549.55 | 36.29 | 35.57 | 12240.00 | {"grouped_batched": 0, "per_kv_fallback": 72} |
| shortlist_base | 49152 | 4 | 1.00 | 0.25 | 662.52 | 599.50 | 748.76 | 63.73 | 62.46 | 12240.00 | {"grouped_batched": 0, "per_kv_fallback": 72} |
| shortlist_l23_ctx | 32768 | 4 | 1.00 | 0.25 | 511.36 | 485.41 | 573.03 | 41.52 | 40.69 | 12336.00 | {"grouped_batched": 0, "per_kv_fallback": 72} |
| shortlist_l23_ctx | 49152 | 4 | 1.00 | 0.25 | 657.81 | 596.86 | 684.06 | 41.38 | 40.55 | 12336.00 | {"grouped_batched": 0, "per_kv_fallback": 72} |
