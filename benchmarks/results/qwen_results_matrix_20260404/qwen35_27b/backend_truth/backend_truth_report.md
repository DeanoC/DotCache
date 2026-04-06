# Qwen3.5 Backend Truth Report

## Decode And Memory

| Context | Variant | Decode ms/step | Decode p95 | Prefill ms | Resident MiB | KV MiB | M0 frac | M3 frac | Shortlist | Selector ms | Selector us/inv |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 485.96 | 492.32 | 989.97 | 36.88 | 25.00 | 100.00 | 0.00 | - | 0.00 | - |
| 1024 | shortlist_base | 504.26 | 521.54 | 980.14 | 36.88 | 25.00 | 100.00 | 0.00 | 33280/33280 | 0.00 | - |
| 1024 | learned_selector | 149.39 | 150.89 | 1100.20 | 95.30 | 64.79 | 0.54 | 99.46 | - | 813.61 | 24.8 |
| 2048 | exact | 821.41 | 821.96 | 1506.96 | 72.75 | 49.00 | 100.00 | 0.00 | - | 0.00 | - |
| 2048 | shortlist_base | 626.30 | 641.57 | 1397.84 | 72.38 | 49.00 | 100.00 | 0.00 | 43520/66048 | 0.00 | - |
| 2048 | learned_selector | 236.01 | 238.26 | 1418.58 | 180.71 | 128.64 | 0.45 | 99.55 | - | 2862.55 | 25.0 |

## Backend Breakdown

| Context | Variant | Prepare | Score | Mix | Softmax | Unpack | Chunk | Payload MiB/step | Metadata KiB/step |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 0.00 | 206.71 | 205.32 | 0.78 | 0.00 | 0.87 | 16.28 | 7571.61 |
| 1024 | shortlist_base | 0.00 | 211.76 | 209.31 | 0.81 | 0.00 | 0.87 | 16.28 | 7571.61 |
| 1024 | learned_selector | 0.00 | 39.71 | 35.91 | 0.76 | 0.00 | 0.86 | 64.02 | 3497.61 |
| 2048 | exact | 0.00 | 370.34 | 372.11 | 0.73 | 0.00 | 0.89 | 32.28 | 15098.61 |
| 2048 | shortlist_base | 0.00 | 267.55 | 263.61 | 0.84 | 0.00 | 0.89 | 21.28 | 9929.18 |
| 2048 | learned_selector | 0.00 | 81.96 | 68.36 | 0.76 | 0.00 | 0.90 | 127.85 | 6943.61 |

## Speedups

| Context | Variant | Vs Exact | Vs Shortlist |
| ---: | --- | ---: | ---: |
| 1024 | learned_selector | 3.25 | 3.38 |
| 2048 | learned_selector | 3.48 | 2.65 |
