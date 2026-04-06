# Qwen3.5 CUDA Shortlist Repro Serving Summary

| Mode | Case | Context | n prompts | Mean decode ms/step | Min | Max | Stddev | 95% CI | Selected pages | Decode path |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| default | shortlist_base | 32768 | 3 | 623.88 | 610.76 | 634.03 | 11.92 | 13.48 | 4080 | {"grouped_batched": 0, "per_kv_fallback": 24} |
| default | shortlist_base | 49152 | 3 | 741.45 | 722.64 | 768.11 | 23.73 | 26.85 | 4080 | {"grouped_batched": 0, "per_kv_fallback": 24} |
| default | shortlist_l23_ctx | 32768 | 3 | 626.15 | 623.92 | 628.93 | 2.55 | 2.88 | 4112 | {"grouped_batched": 0, "per_kv_fallback": 24} |
| default | shortlist_l23_ctx | 49152 | 3 | 792.68 | 760.28 | 809.87 | 28.07 | 31.76 | 4112 | {"grouped_batched": 0, "per_kv_fallback": 24} |
| forced_grouped | shortlist_base | 32768 | 3 | 669.76 | 650.74 | 688.73 | 18.99 | 21.49 | 4232 | {"grouped_batched": 24, "per_kv_fallback": 0} |
| forced_grouped | shortlist_base | 49152 | 3 | 775.01 | 751.77 | 807.00 | 28.64 | 32.41 | 4222 | {"grouped_batched": 24, "per_kv_fallback": 0} |
| forced_grouped | shortlist_l23_ctx | 32768 | 3 | 672.73 | 668.41 | 678.05 | 4.90 | 5.54 | 4282 | {"grouped_batched": 24, "per_kv_fallback": 0} |
| forced_grouped | shortlist_l23_ctx | 49152 | 3 | 788.97 | 775.86 | 800.22 | 12.29 | 13.91 | 4274 | {"grouped_batched": 24, "per_kv_fallback": 0} |
