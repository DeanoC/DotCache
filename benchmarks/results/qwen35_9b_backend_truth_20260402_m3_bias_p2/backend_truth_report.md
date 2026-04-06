# Qwen3.5 Backend Truth Report

## Decode And Memory

| Context | Variant | Decode ms/step | Decode p95 | Prefill ms | Resident MiB | KV MiB | M0 frac | M3 frac | Shortlist | Selector ms | Selector us/inv |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 239.43 | 240.55 | 68.19 | 18.44 | 12.50 | 100.00 | 0.00 | - | 0.00 | - |
| 1024 | shortlist_base | 241.46 | 241.83 | 68.37 | 18.44 | 12.50 | 100.00 | 0.00 | 16640/16640 | 0.00 | - |
| 1024 | learned_selector | 74.73 | 76.29 | 68.19 | 48.29 | 32.27 | 1.17 | 98.83 | - | 558.70 | 25.7 |
| 2048 | exact | 412.26 | 413.70 | 124.23 | 36.38 | 24.50 | 100.00 | 0.00 | - | 0.00 | - |
| 2048 | shortlist_base | 303.69 | 304.65 | 123.50 | 35.41 | 24.50 | 100.00 | 0.00 | 21760/33024 | 0.00 | - |
| 2048 | learned_selector | 103.46 | 104.34 | 122.57 | 96.57 | 64.44 | 0.15 | 99.85 | - | 1609.04 | 25.7 |

## Backend Breakdown

| Context | Variant | Prepare | Score | Mix | Softmax | Unpack | Chunk | Payload MiB/step | Metadata KiB/step |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 0.00 | 106.66 | 106.02 | 0.41 | 0.00 | 0.43 | 8.14 | 3785.30 |
| 1024 | shortlist_base | 0.00 | 105.83 | 104.81 | 0.41 | 0.00 | 0.43 | 8.14 | 3785.30 |
| 1024 | learned_selector | 0.00 | 18.84 | 28.36 | 0.42 | 0.00 | 0.54 | 31.86 | 1761.30 |
| 2048 | exact | 0.00 | 191.05 | 191.78 | 0.39 | 0.00 | 0.44 | 16.14 | 7548.30 |
| 2048 | shortlist_base | 0.00 | 134.44 | 132.84 | 0.42 | 0.00 | 0.44 | 10.64 | 4963.92 |
| 2048 | learned_selector | 0.00 | 33.89 | 39.47 | 0.40 | 0.00 | 0.52 | 64.07 | 3458.30 |

## Speedups

| Context | Variant | Vs Exact | Vs Shortlist |
| ---: | --- | ---: | ---: |
| 1024 | learned_selector | 3.20 | 3.23 |
| 2048 | learned_selector | 3.98 | 2.94 |
