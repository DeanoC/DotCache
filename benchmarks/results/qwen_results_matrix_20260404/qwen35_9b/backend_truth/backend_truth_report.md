# Qwen3.5 Backend Truth Report

## Decode And Memory

| Context | Variant | Decode ms/step | Decode p95 | Prefill ms | Resident MiB | KV MiB | M0 frac | M3 frac | Shortlist | Selector ms | Selector us/inv |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 242.52 | 268.87 | 456.01 | 18.44 | 12.50 | 100.00 | 0.00 | - | 0.00 | - |
| 1024 | shortlist_base | 242.64 | 268.88 | 460.20 | 18.44 | 12.50 | 100.00 | 0.00 | 16640/16640 | 0.00 | - |
| 1024 | learned_selector | 74.78 | 77.41 | 446.63 | 48.30 | 32.26 | 1.22 | 98.78 | - | 423.95 | 25.9 |
| 2048 | exact | 404.70 | 419.47 | 525.69 | 36.38 | 24.50 | 100.00 | 0.00 | - | 0.00 | - |
| 2048 | shortlist_base | 307.73 | 317.27 | 538.05 | 35.41 | 24.50 | 100.00 | 0.00 | 21760/33024 | 0.00 | - |
| 2048 | learned_selector | 105.72 | 106.14 | 529.92 | 96.57 | 64.44 | 0.15 | 99.85 | - | 1463.62 | 25.5 |

## Backend Breakdown

| Context | Variant | Prepare | Score | Mix | Softmax | Unpack | Chunk | Payload MiB/step | Metadata KiB/step |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 0.00 | 105.61 | 104.94 | 0.39 | 0.00 | 0.44 | 8.14 | 3785.30 |
| 1024 | shortlist_base | 0.00 | 104.54 | 103.11 | 0.40 | 0.00 | 0.44 | 8.14 | 3785.30 |
| 1024 | learned_selector | 0.00 | 18.31 | 26.05 | 0.38 | 0.00 | 0.52 | 31.85 | 1762.30 |
| 2048 | exact | 0.00 | 184.83 | 186.14 | 0.37 | 0.00 | 0.45 | 16.14 | 7548.30 |
| 2048 | shortlist_base | 0.00 | 133.18 | 131.58 | 0.42 | 0.00 | 0.45 | 10.64 | 4963.92 |
| 2048 | learned_selector | 0.00 | 33.26 | 36.80 | 0.35 | 0.00 | 0.49 | 64.07 | 3458.30 |

## Speedups

| Context | Variant | Vs Exact | Vs Shortlist |
| ---: | --- | ---: | ---: |
| 1024 | learned_selector | 3.24 | 3.24 |
| 2048 | learned_selector | 3.83 | 2.91 |
