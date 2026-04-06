# Qwen3.5 Backend Truth Report

## Decode And Memory

| Context | Variant | Decode ms/step | Decode p95 | Prefill ms | Resident MiB | KV MiB | M0 frac | M3 frac | Shortlist | Selector ms | Selector us/inv |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 90.26 | 91.70 | 15.61 | 11.03 | 7.48 | 50.00 | 16.67 | - | 0.00 | - |
| 1024 | shortlist_base | 94.65 | 97.22 | 16.03 | 11.03 | 7.48 | 50.00 | 16.67 | 3120/3120 | 0.00 | - |
| 1024 | learned_selector | 62.13 | 67.89 | 15.73 | 13.66 | 9.25 | 39.19 | 60.81 | - | 171.33 | 27.9 |
| 2048 | exact | 145.66 | 147.10 | 18.99 | 21.84 | 14.79 | 50.00 | 16.67 | - | 0.00 | - |
| 2048 | shortlist_base | 112.53 | 114.65 | 18.93 | 21.60 | 14.79 | 50.00 | 16.67 | 4080/6192 | 0.00 | - |
| 2048 | learned_selector | 105.71 | 105.83 | 18.82 | 25.66 | 17.66 | 43.49 | 56.51 | - | 589.17 | 27.4 |

## Backend Breakdown

| Context | Variant | Prepare | Score | Mix | Softmax | Unpack | Chunk | Payload MiB/step | Metadata KiB/step |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 0.00 | 40.23 | 34.48 | 0.16 | 0.00 | 0.19 | 3.35 | 4344.60 |
| 1024 | shortlist_base | 0.00 | 40.84 | 35.25 | 0.16 | 0.00 | 0.20 | 3.35 | 4344.60 |
| 1024 | learned_selector | 0.00 | 26.61 | 19.41 | 0.16 | 0.00 | 0.21 | 8.50 | 952.36 |
| 2048 | exact | 0.00 | 69.97 | 58.11 | 0.15 | 0.00 | 0.20 | 6.66 | 8705.85 |
| 2048 | shortlist_base | 0.00 | 48.37 | 42.18 | 0.16 | 0.00 | 0.20 | 4.38 | 5732.53 |
| 2048 | learned_selector | 0.00 | 56.78 | 32.38 | 0.15 | 0.00 | 0.20 | 16.20 | 1962.36 |

## Speedups

| Context | Learned vs Exact | Learned vs Shortlist |
| ---: | ---: | ---: |
| 1024 | 1.45 | 1.52 |
| 2048 | 1.38 | 1.06 |
