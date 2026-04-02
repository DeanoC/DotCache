# Qwen3.5 Backend Truth Report

## Decode And Memory

| Context | Variant | Decode ms/step | Decode p95 | Prefill ms | Resident MiB | KV MiB | M0 frac | M3 frac | Shortlist | Selector ms | Selector us/inv |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 89.42 | 96.62 | 16.81 | 11.03 | 7.48 | 50.00 | 16.67 | - | 0.00 | - |
| 1024 | shortlist_base | 91.82 | 92.98 | 15.43 | 11.03 | 7.48 | 50.00 | 16.67 | 3120/3120 | 0.00 | - |
| 1024 | learned_selector | 60.26 | 60.29 | 15.25 | 13.66 | 9.25 | 39.19 | 60.81 | - | 165.16 | 26.9 |
| 1024 | learned_selector_k_only | 74.80 | 76.47 | 15.41 | 11.41 | 7.72 | 56.84 | 42.90 | - | 88.01 | 28.6 |
| 1024 | learned_selector_v_only | 73.96 | 74.22 | 15.29 | 12.60 | 9.01 | 32.36 | 34.57 | - | 88.11 | 28.7 |
| 2048 | exact | 142.22 | 142.69 | 18.80 | 21.84 | 14.79 | 50.00 | 16.67 | - | 0.00 | - |
| 2048 | shortlist_base | 110.87 | 110.97 | 18.92 | 21.60 | 14.79 | 50.00 | 16.67 | 4080/6192 | 0.00 | - |
| 2048 | learned_selector | 103.99 | 105.10 | 18.78 | 25.66 | 17.66 | 43.49 | 56.51 | - | 576.17 | 26.8 |
| 2048 | learned_selector_k_only | 128.65 | 129.31 | 18.76 | 21.08 | 14.71 | 60.58 | 39.29 | - | 305.65 | 28.4 |
| 2048 | learned_selector_v_only | 117.15 | 117.90 | 18.80 | 25.05 | 17.75 | 32.91 | 33.89 | - | 309.18 | 28.8 |

## Backend Breakdown

| Context | Variant | Prepare | Score | Mix | Softmax | Unpack | Chunk | Payload MiB/step | Metadata KiB/step |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 0.00 | 39.75 | 34.13 | 0.16 | 0.00 | 0.19 | 3.35 | 4344.60 |
| 1024 | shortlist_base | 0.00 | 39.67 | 33.96 | 0.15 | 0.00 | 0.19 | 3.35 | 4344.60 |
| 1024 | learned_selector | 0.00 | 25.92 | 18.80 | 0.15 | 0.00 | 0.21 | 8.50 | 952.36 |
| 1024 | learned_selector_k_only | 0.00 | 25.94 | 33.84 | 0.15 | 0.00 | 0.19 | 6.71 | 1092.88 |
| 1024 | learned_selector_v_only | 0.00 | 39.95 | 18.82 | 0.15 | 0.00 | 0.21 | 5.14 | 4204.08 |
| 2048 | exact | 0.00 | 68.78 | 56.88 | 0.14 | 0.00 | 0.20 | 6.66 | 8705.85 |
| 2048 | shortlist_base | 0.00 | 47.93 | 41.49 | 0.16 | 0.00 | 0.19 | 4.38 | 5732.53 |
| 2048 | learned_selector | 0.00 | 55.73 | 32.27 | 0.14 | 0.00 | 0.20 | 16.20 | 1962.36 |
| 2048 | learned_selector_k_only | 0.00 | 55.50 | 56.83 | 0.14 | 0.00 | 0.19 | 12.73 | 2233.88 |
| 2048 | learned_selector_v_only | 0.00 | 68.61 | 32.17 | 0.14 | 0.00 | 0.20 | 10.13 | 8434.33 |

## Speedups

| Context | Variant | Vs Exact | Vs Shortlist |
| ---: | --- | ---: | ---: |
| 1024 | learned_selector | 1.48 | 1.52 |
| 1024 | learned_selector_k_only | 1.20 | 1.23 |
| 1024 | learned_selector_v_only | 1.21 | 1.24 |
| 2048 | learned_selector | 1.37 | 1.07 |
| 2048 | learned_selector_k_only | 1.11 | 0.86 |
| 2048 | learned_selector_v_only | 1.21 | 0.95 |
