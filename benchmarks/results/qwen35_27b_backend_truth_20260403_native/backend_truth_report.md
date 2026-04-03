# Qwen3.5 Backend Truth Report

## Decode And Memory

| Context | Variant | Decode ms/step | Decode p95 | Prefill ms | Resident MiB | KV MiB | M0 frac | M3 frac | Shortlist | Selector ms | Selector us/inv |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 504.45 | 504.45 | 218.71 | 36.88 | 25.00 | 100.00 | 0.00 | - | 0.00 | - |
| 1024 | shortlist_base | 519.36 | 519.36 | 218.13 | 36.88 | 25.00 | 100.00 | 0.00 | 33280/33280 | 0.00 | - |
| 1024 | learned_selector | 157.67 | 157.67 | 217.93 | 94.31 | 64.78 | 0.55 | 99.45 | - | 279.13 | 28.0 |
| 2048 | exact | 849.79 | 849.79 | 406.27 | 72.75 | 49.00 | 100.00 | 0.00 | - | 0.00 | - |
| 2048 | shortlist_base | 631.67 | 631.67 | 404.74 | 72.38 | 49.00 | 100.00 | 0.00 | 43520/66048 | 0.00 | - |
| 2048 | learned_selector | 240.21 | 240.21 | 401.46 | 180.71 | 128.64 | 0.45 | 99.55 | - | 707.10 | 26.8 |

## Backend Breakdown

| Context | Variant | Prepare | Score | Mix | Softmax | Unpack | Chunk | Payload MiB/step | Metadata KiB/step |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 0.00 | 219.11 | 217.92 | 0.83 | 0.00 | 0.89 | 16.28 | 7571.61 |
| 1024 | shortlist_base | 0.00 | 223.10 | 219.00 | 0.86 | 0.00 | 0.91 | 16.28 | 7571.61 |
| 1024 | learned_selector | 0.00 | 46.30 | 40.56 | 0.83 | 0.00 | 0.91 | 64.02 | 3498.11 |
| 2048 | exact | 0.00 | 387.30 | 388.77 | 0.80 | 0.00 | 0.94 | 32.28 | 15098.61 |
| 2048 | shortlist_base | 0.00 | 273.65 | 269.96 | 0.86 | 0.00 | 0.90 | 21.28 | 9929.18 |
| 2048 | learned_selector | 0.00 | 89.16 | 73.88 | 0.80 | 0.00 | 0.96 | 127.85 | 6943.61 |

## Speedups

| Context | Variant | Vs Exact | Vs Shortlist |
| ---: | --- | ---: | ---: |
| 1024 | learned_selector | 3.20 | 3.29 |
| 2048 | learned_selector | 3.54 | 2.63 |
