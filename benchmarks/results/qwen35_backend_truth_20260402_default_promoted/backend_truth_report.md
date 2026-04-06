# Qwen3.5 Backend Truth Report

## Decode And Memory

| Context | Variant | Decode ms/step | Prefill ms | Resident MiB | KV MiB | M0 frac | M3 frac | Shortlist | Selector ms | Selector us/inv |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 113.55 | 427.61 | 11.03 | 7.48 | 50.00 | 16.67 | - | 0.00 | - |
| 1024 | shortlist_base | 120.15 | 431.13 | 11.03 | 7.48 | 50.00 | 16.67 | 3120/3120 | 0.00 | - |
| 1024 | learned_selector | 81.83 | 424.57 | 13.66 | 9.25 | 39.19 | 60.81 | - | 41.57 | 27.1 |
| 2048 | exact | 153.21 | 19.28 | 21.84 | 14.79 | 50.00 | 16.67 | - | 0.00 | - |
| 2048 | shortlist_base | 115.63 | 26.93 | 21.60 | 14.79 | 50.00 | 16.67 | 4080/6192 | 0.00 | - |
| 2048 | learned_selector | 113.57 | 19.29 | 25.66 | 17.66 | 43.49 | 56.51 | - | 124.09 | 26.9 |

## Backend Breakdown

| Context | Variant | Prepare | Score | Mix | Softmax | Unpack | Chunk | Payload MiB/step | Metadata KiB/step |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 0.00 | 43.77 | 43.08 | 2.22 | 0.00 | 0.20 | 3.35 | 4344.60 |
| 1024 | shortlist_base | 0.00 | 45.08 | 44.04 | 2.16 | 0.00 | 0.20 | 3.35 | 4344.60 |
| 1024 | learned_selector | 0.00 | 28.60 | 25.07 | 2.19 | 0.00 | 0.21 | 8.50 | 952.36 |
| 2048 | exact | 0.00 | 72.55 | 59.88 | 0.17 | 0.00 | 0.20 | 6.66 | 8705.85 |
| 2048 | shortlist_base | 0.00 | 48.85 | 42.89 | 0.17 | 0.00 | 0.20 | 4.38 | 5732.53 |
| 2048 | learned_selector | 0.00 | 59.75 | 34.70 | 0.16 | 0.00 | 0.21 | 16.20 | 1962.36 |

## Speedups

| Context | Learned vs Exact | Learned vs Shortlist |
| ---: | ---: | ---: |
| 1024 | 1.39 | 1.47 |
| 2048 | 1.35 | 1.02 |
