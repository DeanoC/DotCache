# Qwen3.5 Backend Truth Report

## Decode And Memory

| Context | Variant | Decode ms/step | Decode p95 | Prefill ms | Resident MiB | KV MiB | M0 frac | M3 frac | Shortlist | Selector ms | Selector us/inv |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 89.00 | 95.73 | 15.27 | 11.03 | 7.48 | 50.00 | 16.67 | - | 0.00 | - |
| 1024 | shortlist_base | 93.68 | 93.99 | 15.71 | 11.03 | 7.48 | 50.00 | 16.67 | 3120/3120 | 0.00 | - |
| 1024 | learned_selector | 30.89 | 32.57 | 15.50 | 17.78 | 12.11 | 0.98 | 99.02 | - | 158.56 | 25.8 |
| 1024 | learned_selector_k_only | 59.08 | 60.39 | 15.61 | 13.59 | 9.43 | 34.05 | 65.69 | - | 87.35 | 28.4 |
| 1024 | learned_selector_v_only | 62.17 | 64.19 | 15.50 | 14.22 | 10.17 | 16.93 | 50.00 | - | 87.32 | 28.4 |
| 2048 | exact | 142.98 | 148.13 | 18.86 | 21.84 | 14.79 | 50.00 | 16.67 | - | 0.00 | - |
| 2048 | shortlist_base | 111.36 | 112.56 | 19.09 | 21.60 | 14.79 | 50.00 | 16.67 | 4080/6192 | 0.00 | - |
| 2048 | learned_selector | 42.17 | 42.32 | 18.66 | 35.29 | 24.10 | 0.55 | 99.45 | - | 554.63 | 25.8 |
| 2048 | learned_selector_k_only | 88.08 | 88.18 | 18.73 | 26.91 | 18.73 | 33.76 | 66.11 | - | 306.25 | 28.5 |
| 2048 | learned_selector_v_only | 98.14 | 100.99 | 18.71 | 28.22 | 20.17 | 16.80 | 50.00 | - | 305.08 | 28.4 |

## Backend Breakdown

| Context | Variant | Prepare | Score | Mix | Softmax | Unpack | Chunk | Payload MiB/step | Metadata KiB/step |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 0.00 | 40.06 | 33.76 | 0.15 | 0.00 | 0.19 | 3.35 | 4344.60 |
| 1024 | shortlist_base | 0.00 | 40.38 | 34.48 | 0.15 | 0.00 | 0.19 | 3.35 | 4344.60 |
| 1024 | learned_selector | 0.00 | 9.12 | 6.93 | 0.16 | 0.00 | 0.16 | 11.94 | 658.86 |
| 1024 | learned_selector_k_only | 0.00 | 9.25 | 34.56 | 0.16 | 0.00 | 0.19 | 8.76 | 917.88 |
| 1024 | learned_selector_v_only | 0.00 | 39.88 | 7.06 | 0.15 | 0.00 | 0.17 | 6.53 | 4085.58 |
| 2048 | exact | 0.00 | 69.02 | 57.16 | 0.14 | 0.00 | 0.20 | 6.66 | 8705.85 |
| 2048 | shortlist_base | 0.00 | 47.77 | 41.62 | 0.16 | 0.00 | 0.20 | 4.38 | 5732.53 |
| 2048 | learned_selector | 0.00 | 13.98 | 12.35 | 0.15 | 0.00 | 0.17 | 23.93 | 1302.86 |
| 2048 | learned_selector_k_only | 0.00 | 14.10 | 57.54 | 0.15 | 0.00 | 0.19 | 17.56 | 1821.88 |
| 2048 | learned_selector_v_only | 0.00 | 69.14 | 12.46 | 0.14 | 0.00 | 0.17 | 13.03 | 8186.83 |

## Speedups

| Context | Variant | Vs Exact | Vs Shortlist |
| ---: | --- | ---: | ---: |
| 1024 | learned_selector | 2.88 | 3.03 |
| 1024 | learned_selector_k_only | 1.51 | 1.59 |
| 1024 | learned_selector_v_only | 1.43 | 1.51 |
| 2048 | learned_selector | 3.39 | 2.64 |
| 2048 | learned_selector_k_only | 1.62 | 1.26 |
| 2048 | learned_selector_v_only | 1.46 | 1.13 |
