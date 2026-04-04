# Qwen3.5 Backend Truth Report

## Decode And Memory

| Context | Variant | Decode ms/step | Decode p95 | Prefill ms | Resident MiB | KV MiB | M0 frac | M3 frac | Shortlist | Selector ms | Selector us/inv |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 507.44 | 507.44 | 1480.54 | 36.88 | 25.00 | 100.00 | 0.00 | - | 0.00 | - |
| 1024 | shortlist_base | 524.77 | 524.77 | 1437.93 | 36.88 | 25.00 | 100.00 | 0.00 | 33280/33280 | 0.00 | - |
| 1024 | learned_selector | 169.09 | 169.09 | 1516.46 | 95.30 | 64.79 | 0.54 | 99.46 | - | 201.25 | 24.6 |
| 2048 | exact | 824.98 | 824.98 | 1574.12 | 72.75 | 49.00 | 100.00 | 0.00 | - | 0.00 | - |
| 2048 | shortlist_base | 642.79 | 642.79 | 1353.73 | 72.38 | 49.00 | 100.00 | 0.00 | 43520/66048 | 0.00 | - |
| 2048 | learned_selector | 224.14 | 224.14 | 1529.95 | 180.71 | 128.64 | 0.45 | 99.55 | - | 608.53 | 24.8 |

## Backend Breakdown

| Context | Variant | Prepare | Score | Mix | Softmax | Unpack | Chunk | Payload MiB/step | Metadata KiB/step |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 0.00 | 208.94 | 207.97 | 1.77 | 0.00 | 0.86 | 16.28 | 7571.61 |
| 1024 | shortlist_base | 0.00 | 214.05 | 211.07 | 1.67 | 0.00 | 0.88 | 16.28 | 7571.61 |
| 1024 | learned_selector | 0.00 | 42.32 | 36.77 | 1.56 | 0.00 | 0.86 | 64.02 | 3497.61 |
| 2048 | exact | 0.00 | 372.01 | 373.79 | 0.72 | 0.00 | 0.87 | 32.28 | 15098.61 |
| 2048 | shortlist_base | 0.00 | 286.27 | 262.26 | 0.81 | 0.00 | 0.89 | 21.28 | 9929.18 |
| 2048 | learned_selector | 0.00 | 77.85 | 66.10 | 0.71 | 0.00 | 0.88 | 127.85 | 6943.61 |

## Speedups

| Context | Variant | Vs Exact | Vs Shortlist |
| ---: | --- | ---: | ---: |
| 1024 | learned_selector | 3.00 | 3.10 |
| 2048 | learned_selector | 3.68 | 2.87 |
