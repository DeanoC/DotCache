# Qwen3.5 Backend Truth Report

## Decode And Memory

| Context | Variant | Decode ms/step | Decode p95 | Prefill ms | Resident MiB | KV MiB | M0 frac | M3 frac | Shortlist | Selector ms | Selector us/inv |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 520.86 | 520.86 | 1417.73 | 36.88 | 25.00 | 100.00 | 0.00 | - | 0.00 | - |
| 1024 | shortlist_base | 522.21 | 522.21 | 1430.70 | 36.88 | 25.00 | 100.00 | 0.00 | 33280/33280 | 0.00 | - |
| 1024 | learned_selector | 171.90 | 171.90 | 1311.09 | 95.30 | 64.79 | 0.54 | 99.46 | - | 204.43 | 25.0 |
| 2048 | exact | 838.43 | 838.43 | 1660.68 | 72.75 | 49.00 | 100.00 | 0.00 | - | 0.00 | - |
| 2048 | shortlist_base | 621.43 | 621.43 | 1550.75 | 72.38 | 49.00 | 100.00 | 0.00 | 43520/66048 | 0.00 | - |
| 2048 | learned_selector | 228.78 | 228.78 | 1585.29 | 180.71 | 128.64 | 0.45 | 99.55 | - | 610.03 | 24.8 |

## Backend Breakdown

| Context | Variant | Prepare | Score | Mix | Softmax | Unpack | Chunk | Payload MiB/step | Metadata KiB/step |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 0.00 | 214.76 | 213.07 | 1.71 | 0.00 | 0.87 | 16.28 | 7571.61 |
| 1024 | shortlist_base | 0.00 | 213.16 | 209.86 | 1.72 | 0.00 | 0.87 | 16.28 | 7571.61 |
| 1024 | learned_selector | 0.00 | 44.05 | 38.04 | 1.58 | 0.00 | 0.88 | 64.02 | 3497.61 |
| 2048 | exact | 0.00 | 378.61 | 378.50 | 0.79 | 0.00 | 0.89 | 32.28 | 15098.61 |
| 2048 | shortlist_base | 0.00 | 265.02 | 262.77 | 0.81 | 0.00 | 0.89 | 21.28 | 9929.18 |
| 2048 | learned_selector | 0.00 | 79.92 | 67.01 | 0.72 | 0.00 | 0.92 | 127.85 | 6943.61 |

## Speedups

| Context | Variant | Vs Exact | Vs Shortlist |
| ---: | --- | ---: | ---: |
| 1024 | learned_selector | 3.03 | 3.04 |
| 2048 | learned_selector | 3.66 | 2.72 |
