# Qwen3.5 Backend Truth Report

## Decode And Memory

| Context | Variant | Decode ms/step | Prefill ms | Resident MiB | KV MiB | M0 frac | M3 frac | Shortlist | Selector ms | Selector us/inv |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 115.65 | 431.34 | 11.03 | 7.48 | 50.00 | 16.67 | - | 0.00 | - |
| 1024 | shortlist_base | 117.92 | 469.43 | 11.03 | 7.48 | 50.00 | 16.67 | 3120/3120 | 0.00 | - |
| 1024 | learned_selector | 78.42 | 431.48 | 13.57 | 9.25 | 39.19 | 60.81 | - | 41.01 | 26.7 |
| 2048 | exact | 144.64 | 19.35 | 21.84 | 14.79 | 50.00 | 16.67 | - | 0.00 | - |
| 2048 | shortlist_base | 111.70 | 19.43 | 21.60 | 14.79 | 50.00 | 16.67 | 4080/6192 | 0.00 | - |
| 2048 | learned_selector | 106.88 | 19.18 | 25.56 | 17.66 | 43.49 | 56.51 | - | 123.00 | 26.7 |

## Backend Breakdown

| Context | Variant | Prepare | Score | Mix | Softmax | Unpack | Chunk | Payload MiB/step | Metadata KiB/step |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 0.00 | 44.04 | 44.24 | 2.27 | 0.00 | 0.20 | 3.35 | 4344.60 |
| 1024 | shortlist_base | 0.00 | 44.15 | 43.72 | 2.21 | 0.00 | 0.20 | 3.35 | 4344.60 |
| 1024 | learned_selector | 0.00 | 27.69 | 23.33 | 2.08 | 1.76 | 0.21 | 8.50 | 952.36 |
| 2048 | exact | 0.00 | 69.56 | 57.66 | 0.16 | 0.00 | 0.20 | 6.66 | 8705.85 |
| 2048 | shortlist_base | 0.00 | 48.03 | 41.69 | 0.16 | 0.00 | 0.19 | 4.38 | 5732.53 |
| 2048 | learned_selector | 0.00 | 57.88 | 32.34 | 0.15 | 5.58 | 0.19 | 16.20 | 1962.36 |

## Speedups

| Context | Learned vs Exact | Learned vs Shortlist |
| ---: | ---: | ---: |
| 1024 | 1.47 | 1.50 |
| 2048 | 1.35 | 1.05 |
