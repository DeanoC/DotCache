# Qwen3.5 Backend Truth Report

## Decode And Memory

| Context | Variant | Decode ms/step | Decode p95 | Prefill ms | Resident MiB | KV MiB | M0 frac | M3 frac | Shortlist | Selector ms | Selector us/inv |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 497.90 | 497.90 | 216.74 | 36.88 | 25.00 | 100.00 | 0.00 | - | 0.00 | - |
| 1024 | shortlist_base | 514.77 | 514.77 | 217.17 | 36.88 | 25.00 | 100.00 | 0.00 | 33280/33280 | 0.00 | - |
| 1024 | learned_selector | 158.67 | 158.67 | 216.69 | 87.68 | 64.44 | 1.40 | 98.60 | - | 263.01 | 26.3 |
| 2048 | exact | 840.85 | 840.85 | 403.41 | 72.75 | 49.00 | 100.00 | 0.00 | - | 0.00 | - |
| 2048 | shortlist_base | 634.79 | 634.79 | 404.66 | 72.38 | 49.00 | 100.00 | 0.00 | 43520/66048 | 0.00 | - |
| 2048 | learned_selector | 242.44 | 242.44 | 403.89 | 140.30 | 127.92 | 1.35 | 98.65 | - | 673.88 | 25.6 |

## Backend Breakdown

| Context | Variant | Prepare | Score | Mix | Softmax | Unpack | Chunk | Payload MiB/step | Metadata KiB/step |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 0.00 | 215.97 | 214.59 | 0.81 | 0.00 | 0.89 | 16.28 | 7571.61 |
| 1024 | shortlist_base | 0.00 | 220.09 | 217.75 | 0.85 | 0.00 | 0.90 | 16.28 | 7571.61 |
| 1024 | learned_selector | 0.00 | 38.79 | 51.94 | 0.78 | 0.00 | 1.00 | 63.61 | 3533.11 |
| 2048 | exact | 0.00 | 383.55 | 385.10 | 0.77 | 0.00 | 0.93 | 32.28 | 15098.61 |
| 2048 | shortlist_base | 0.00 | 274.90 | 270.31 | 0.87 | 0.00 | 0.91 | 21.28 | 9929.18 |
| 2048 | learned_selector | 0.00 | 69.70 | 97.48 | 0.74 | 0.00 | 1.23 | 126.98 | 7017.61 |

## Speedups

| Context | Variant | Vs Exact | Vs Shortlist |
| ---: | --- | ---: | ---: |
| 1024 | learned_selector | 3.14 | 3.24 |
| 2048 | learned_selector | 3.47 | 2.62 |
