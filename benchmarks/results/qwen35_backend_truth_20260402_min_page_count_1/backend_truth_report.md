# Qwen3.5 Backend Truth Report

## Decode And Memory

| Context | Variant | Decode ms/step | Prefill ms | Resident MiB | KV MiB | M0 frac | M3 frac | Shortlist | Selector ms | Selector us/inv |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 111.10 | 405.95 | 11.03 | 7.48 | 50.00 | 16.67 | - | 0.00 | - |
| 1024 | shortlist_base | 119.26 | 425.30 | 11.03 | 7.48 | 50.00 | 16.67 | 3120/3120 | 0.00 | - |
| 1024 | learned_selector | 78.33 | 440.03 | 13.66 | 9.25 | 39.19 | 60.81 | - | 40.75 | 26.5 |
| 2048 | exact | 144.42 | 19.28 | 21.84 | 14.79 | 50.00 | 16.67 | - | 0.00 | - |
| 2048 | shortlist_base | 114.44 | 19.66 | 21.60 | 14.79 | 50.00 | 16.67 | 4080/6192 | 0.00 | - |
| 2048 | learned_selector | 105.01 | 19.79 | 25.66 | 17.66 | 43.49 | 56.51 | - | 123.35 | 26.8 |

## Backend Breakdown

| Context | Variant | Prepare | Score | Mix | Softmax | Unpack | Chunk | Payload MiB/step | Metadata KiB/step |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | exact | 0.00 | 43.26 | 42.32 | 2.01 | 0.00 | 0.20 | 3.35 | 4344.60 |
| 1024 | shortlist_base | 0.00 | 43.89 | 42.95 | 2.23 | 0.00 | 0.20 | 3.35 | 4344.60 |
| 1024 | learned_selector | 0.00 | 27.22 | 23.24 | 2.04 | 0.00 | 0.21 | 8.50 | 952.36 |
| 2048 | exact | 0.00 | 69.63 | 57.52 | 0.16 | 0.00 | 0.19 | 6.66 | 8705.85 |
| 2048 | shortlist_base | 0.00 | 48.80 | 42.50 | 0.16 | 0.00 | 0.20 | 4.38 | 5732.53 |
| 2048 | learned_selector | 0.00 | 56.18 | 32.07 | 0.15 | 0.00 | 0.20 | 16.20 | 1962.36 |

## Speedups

| Context | Learned vs Exact | Learned vs Shortlist |
| ---: | ---: | ---: |
| 1024 | 1.42 | 1.52 |
| 2048 | 1.38 | 1.09 |
