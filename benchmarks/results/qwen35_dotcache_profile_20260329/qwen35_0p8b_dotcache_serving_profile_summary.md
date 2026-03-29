# Qwen3.5 DotCache Serving Profile

## Breakdown

| Context | Step ms | DotCache decode ms | Append ms | QKV ms | Output ms | Other ms | Decode share | Resident MiB | Total Device MiB | K Pages | V Pages |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 401.33 | 380.79 | 0.74 | 2.24 | 0.38 | 17.17 | 94.88 | 42.22 | 3758.06 | 3072 | 3072 |
| 16384 | 1490.67 | 1469.07 | 0.79 | 2.30 | 0.44 | 18.06 | 98.55 | 164.91 | 9867.08 | 12288 | 12288 |
| 32768 | 2961.35 | 2939.84 | 0.80 | 2.30 | 0.44 | 17.98 | 99.27 | 252.33 | 17967.76 | 24576 | 24576 |

## Backend Decode Stages

| Context | Prepare ms | Score ms | Softmax ms | Mix ms | Chunk Assembly ms | Unpack ms | FWHT ms | Backend Other ms | H2D MiB/step | Payload MiB/step | Metadata KiB/step | Cache Hit % | Cache Resident MiB | Max Temporary MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 0.00 | 200.59 | 0.21 | 161.14 | 0.40 | 0.00 | 0.00 | 18.46 | 0.00 | 13.29 | 17422.34 | 0.00 | 0.00 | 0.00 |
| 16384 | 0.00 | 761.30 | 0.28 | 642.16 | 0.43 | 1.62 | 0.00 | 63.28 | 0.00 | 53.77 | 65254.85 | 0.00 | 0.00 | 0.00 |
| 32768 | 0.00 | 1662.27 | 0.32 | 1155.47 | 0.45 | 11.88 | 0.00 | 109.44 | 0.00 | 107.79 | 128798.71 | 0.00 | 0.00 | 0.00 |

## Scaling

| Context Range | Context x | Step x | DotCache decode x | Score x | Mix x | Prepare x | Resident x |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4096 -> 16384 | 4.00 | 3.71 | 3.86 | 3.80 | 3.99 | 0.00 | 3.91 |
| 16384 -> 32768 | 2.00 | 1.99 | 2.00 | 2.18 | 1.80 | 0.00 | 1.53 |
