# Qwen3.5 Backend Truth Report

## Decode And Memory

| Context | Variant | Decode ms/step | Prefill ms | Resident MiB | KV MiB | M0 frac | M3 frac | Shortlist | Selector ms | Selector us/inv |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 113.77 | 459.32 | 10.71 | 7.48 | 50.00 | 16.67 | - | 0.00 | - |
| 1024 | shortlist_base | 115.04 | 445.20 | 10.71 | 7.48 | 50.00 | 16.67 | 3120/3120 | 0.00 | - |
| 1024 | learned_selector | 78.97 | 447.30 | 13.65 | 9.25 | 39.19 | 60.81 | - | 41.65 | 27.1 |
| 2048 | exact | 142.72 | 19.48 | 21.21 | 14.79 | 50.00 | 16.67 | - | 0.00 | - |
| 2048 | shortlist_base | 111.32 | 19.34 | 21.07 | 14.79 | 50.00 | 16.67 | 4080/6192 | 0.00 | - |
| 2048 | learned_selector | 105.27 | 19.33 | 24.68 | 17.66 | 43.49 | 56.51 | - | 124.98 | 27.1 |

## Backend Breakdown

| Context | Variant | Prepare | Score | Mix | Softmax | Unpack | Chunk | Payload MiB/step | Metadata KiB/step |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 0.00 | 44.43 | 42.59 | 2.31 | 0.00 | 0.20 | 3.35 | 4344.60 |
| 1024 | shortlist_base | 0.00 | 43.04 | 42.55 | 2.09 | 0.00 | 0.20 | 3.35 | 4344.60 |
| 1024 | learned_selector | 0.00 | 27.17 | 23.43 | 2.16 | 1.69 | 0.21 | 8.50 | 952.36 |
| 2048 | exact | 0.00 | 68.86 | 56.57 | 0.15 | 0.00 | 0.19 | 6.66 | 8705.85 |
| 2048 | shortlist_base | 0.00 | 47.64 | 40.81 | 0.16 | 0.00 | 0.20 | 4.38 | 5732.53 |
| 2048 | learned_selector | 0.00 | 56.52 | 31.78 | 0.15 | 5.38 | 0.20 | 16.20 | 1962.36 |

## Speedups

| Context | Learned vs Exact | Learned vs Shortlist |
| ---: | ---: | ---: |
| 1024 | 1.44 | 1.46 |
| 2048 | 1.36 | 1.06 |
