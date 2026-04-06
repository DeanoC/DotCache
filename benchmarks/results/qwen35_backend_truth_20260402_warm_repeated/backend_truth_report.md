# Qwen3.5 Backend Truth Report

## Decode And Memory

| Context | Variant | Decode ms/step | Decode p95 | Prefill ms | Resident MiB | KV MiB | M0 frac | M3 frac | Shortlist | Selector ms | Selector us/inv |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 91.12 | 93.41 | 15.83 | 11.03 | 7.48 | 50.00 | 16.67 | - | 0.00 | - |
| 1024 | shortlist_base | 95.28 | 97.11 | 16.11 | 11.03 | 7.48 | 50.00 | 16.67 | 3120/3120 | 0.00 | - |
| 1024 | learned_selector | 60.93 | 61.17 | 15.57 | 13.66 | 9.25 | 39.19 | 60.81 | - | 165.44 | 26.9 |
| 2048 | exact | 144.05 | 146.70 | 18.82 | 21.84 | 14.79 | 50.00 | 16.67 | - | 0.00 | - |
| 2048 | shortlist_base | 113.69 | 118.06 | 18.99 | 21.60 | 14.79 | 50.00 | 16.67 | 4080/6192 | 0.00 | - |
| 2048 | learned_selector | 104.87 | 105.97 | 18.92 | 25.66 | 17.66 | 43.49 | 56.51 | - | 572.02 | 26.6 |

## Backend Breakdown

| Context | Variant | Prepare | Score | Mix | Softmax | Unpack | Chunk | Payload MiB/step | Metadata KiB/step |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 0.00 | 40.64 | 34.46 | 0.16 | 0.00 | 0.19 | 3.35 | 4344.60 |
| 1024 | shortlist_base | 0.00 | 40.90 | 34.95 | 0.16 | 0.00 | 0.19 | 3.35 | 4344.60 |
| 1024 | learned_selector | 0.00 | 26.09 | 18.86 | 0.16 | 0.00 | 0.20 | 8.50 | 952.36 |
| 2048 | exact | 0.00 | 69.24 | 57.36 | 0.15 | 0.00 | 0.19 | 6.66 | 8705.85 |
| 2048 | shortlist_base | 0.00 | 48.54 | 42.11 | 0.17 | 0.00 | 0.19 | 4.38 | 5732.53 |
| 2048 | learned_selector | 0.00 | 56.25 | 32.06 | 0.15 | 0.00 | 0.19 | 16.20 | 1962.36 |

## Speedups

| Context | Learned vs Exact | Learned vs Shortlist |
| ---: | ---: | ---: |
| 1024 | 1.50 | 1.56 |
| 2048 | 1.37 | 1.08 |
