# Qwen3.5 Backend Truth Report

## Decode And Memory

| Context | Variant | Decode ms/step | Decode p95 | Prefill ms | Resident MiB | KV MiB | M0 frac | M3 frac | Shortlist | Selector ms | Selector us/inv |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 232.19 | 236.50 | 484.59 | 18.44 | 12.50 | 100.00 | 0.00 | - | 0.00 | - |
| 1024 | shortlist_base | 242.41 | 243.53 | 509.72 | 18.44 | 12.50 | 100.00 | 0.00 | 16640/16640 | 0.00 | - |
| 1024 | learned_selector | 71.47 | 73.92 | 439.12 | 47.84 | 32.15 | 1.76 | 98.24 | - | 412.94 | 25.2 |
| 2048 | exact | 400.16 | 401.94 | 549.96 | 36.38 | 24.50 | 100.00 | 0.00 | - | 0.00 | - |
| 2048 | shortlist_base | 300.46 | 316.82 | 531.65 | 35.41 | 24.50 | 100.00 | 0.00 | 21760/33024 | 0.00 | - |
| 2048 | learned_selector | 122.57 | 123.10 | 563.65 | 95.00 | 63.34 | 2.89 | 97.11 | - | 1444.67 | 25.2 |

## Backend Breakdown

| Context | Variant | Prepare | Score | Mix | Softmax | Unpack | Chunk | Payload MiB/step | Metadata KiB/step |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 0.00 | 103.19 | 102.67 | 0.39 | 0.00 | 0.43 | 8.14 | 3785.30 |
| 1024 | shortlist_base | 0.00 | 105.98 | 104.91 | 0.43 | 0.00 | 0.43 | 8.14 | 3785.30 |
| 1024 | learned_selector | 0.00 | 23.23 | 21.98 | 0.38 | 0.00 | 0.46 | 31.72 | 1773.30 |
| 2048 | exact | 0.00 | 185.30 | 186.12 | 0.36 | 0.00 | 0.44 | 16.14 | 7548.30 |
| 2048 | shortlist_base | 0.00 | 132.36 | 131.47 | 0.44 | 0.00 | 0.44 | 10.64 | 4963.92 |
| 2048 | learned_selector | 0.00 | 55.86 | 37.62 | 0.36 | 0.00 | 0.46 | 62.75 | 3570.80 |

## Speedups

| Context | Variant | Vs Exact | Vs Shortlist |
| ---: | --- | ---: | ---: |
| 1024 | learned_selector | 3.25 | 3.39 |
| 2048 | learned_selector | 3.26 | 2.45 |
