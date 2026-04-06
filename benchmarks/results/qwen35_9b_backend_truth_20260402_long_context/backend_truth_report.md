# Qwen3.5 Backend Truth Report

## Decode And Memory

| Context | Variant | Decode ms/step | Decode p95 | Prefill ms | Resident MiB | KV MiB | M0 frac | M3 frac | Shortlist | Selector ms | Selector us/inv |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | exact | 796.45 | 906.19 | 290.88 | 72.25 | 48.50 | 100.00 | 0.00 | - | 0.00 | - |
| 4096 | shortlist_base | 345.04 | 350.21 | 294.56 | 71.88 | 48.50 | 100.00 | 0.00 | 21760/65792 | 0.00 | - |
| 4096 | learned_selector | 180.37 | 180.64 | 295.91 | 190.51 | 128.50 | 0.01 | 99.99 | - | 1787.58 | 25.2 |
| 8192 | exact | 1476.05 | 1573.00 | 576.77 | 144.00 | 96.50 | 100.00 | 0.00 | - | 0.00 | - |
| 8192 | shortlist_base | 367.03 | 478.28 | 569.25 | 143.26 | 96.50 | 100.00 | 0.00 | 21760/131328 | 0.00 | - |
| 8192 | learned_selector | 308.69 | 379.50 | 570.49 | 316.50 | 256.50 | 0.00 | 100.00 | - | 5868.00 | 25.0 |

## Backend Breakdown

| Context | Variant | Prepare | Score | Mix | Softmax | Unpack | Chunk | Payload MiB/step | Metadata KiB/step |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | exact | 0.00 | 368.63 | 375.09 | 1.10 | 0.00 | 0.45 | 32.14 | 15074.30 |
| 4096 | shortlist_base | 0.00 | 139.35 | 145.69 | 0.86 | 0.00 | 0.44 | 10.64 | 4964.04 |
| 4096 | learned_selector | 0.00 | 66.99 | 65.98 | 0.78 | 0.00 | 0.45 | 128.13 | 6882.80 |
| 8192 | exact | 0.00 | 689.78 | 720.60 | 1.13 | 0.00 | 0.47 | 64.14 | 30126.30 |
| 8192 | shortlist_base | 0.00 | 144.33 | 146.90 | 1.16 | 0.00 | 0.44 | 10.64 | 4963.99 |
| 8192 | learned_selector | 0.00 | 125.01 | 123.60 | 0.80 | 0.00 | 0.46 | 256.14 | 13742.30 |

## Speedups

| Context | Variant | Vs Exact | Vs Shortlist |
| ---: | --- | ---: | ---: |
| 4096 | learned_selector | 4.42 | 1.91 |
| 8192 | learned_selector | 4.78 | 1.19 |
