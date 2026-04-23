# Overnight 2026-04-23 — summary

## Perf stage

### 1b. 64K cap sweep (per-KV-group)

| cap | mean ms/step | p50 ms/step | p95 ms/step | tok/s |
|---|---:|---:|---:|---:|
| 256 | 1091.02 | 1087.05 | 1140.31 | 0.92 |
| 512 | 1090.30 | 1088.38 | 1121.43 | 0.92 |
| 1024 | 1127.27 | 1111.77 | 1195.92 | 0.89 |
| 2048 | 897.69 | 899.70 | 937.65 | 1.11 |

### 1c. Context-scaling summary (tail of log)

```
  Dense:     42.6 ms/step, peak 13.46 GB
  Certified: 119.3 ms/step, peak 19.73 GB
  Speedup:   -180.3%
  Skip rate: 80.2%
  VRAM saved: -6.27 GB

============================================================
  48K context, 32 decode steps (INT8 model)
============================================================
  Prompt: 35457 tokens
  Prefill: 6618 ms, peak 15.56 GB (chunked)
  Tiered: 8556 MB VRAM, 2357 MB CPU (1536 ms)
  After free: 20.01 GB
  Dense:     52.2 ms/step, peak 13.82 GB
  Certified: 128.4 ms/step, peak 20.60 GB
  Speedup:   -145.9%
  Skip rate: 80.6%
  VRAM saved: -6.78 GB

============================================================
  64K context, 32 decode steps (INT8 model)
============================================================
  Prompt: 35457 tokens
  Prefill: 6838 ms, peak 15.56 GB (chunked)
  Tiered: 8556 MB VRAM, 2357 MB CPU (1431 ms)
  After free: 20.01 GB
  Dense:     42.8 ms/step, peak 13.82 GB
  Certified: 128.0 ms/step, peak 20.60 GB
  Speedup:   -198.6%
  Skip rate: 80.6%
  VRAM saved: -6.78 GB

============================================================
CONTEXT SCALING (INT8 model, chunked prefill)
============================================================
  Ctx  Dense ms  Cert ms  Ratio   Skip  Dense GB  Cert GB  Saved
   8K     36.4    60.9   1.7x 51.2%    10.19   11.88 -1.69
  16K     37.9    79.3   2.1x 71.4%    11.28   14.50 -3.22
  32K     42.6   119.3   2.8x 80.2%    13.46   19.73 -6.27
  48K     52.2   128.4   2.5x 80.6%    13.82   20.60 -6.78
  64K     42.8   128.0   3.0x 80.6%    13.82   20.60 -6.78

JSON -> benchmarks/results/certified_64k_int8model.json
```

## Value-error sweep at longer contexts

| context | loose mean | tight mean | tight/loose |
|---|---:|---:|---:|
| 16384 | — | — | (run failed) |
| 32768 | — | — | (run failed) |

## Quality stage

### PG-19 perplexity (with per-chunk CIs from PR #98)

- ctx=4096:
    ==================================================
    Context: 4096 tokens, 20 chunks
    Dense perplexity:     9.9416
    Certified perplexity: 9.9501
    Ratio (cert/dense):   1.000851
    Delta:                +0.0085
    Skip rate:            0.4056 (3272837138/8068792320 blocks)
    Concentration thr:    0.0
    
    JSON -> /workspace/DotCache/benchmarks/results/overnight_20260423/pg19_ctx4096.json
- ctx=8192:
      Certified [5/20]: chunk_ppl=6.45, suffix_ppl=5.38, running_ppl=6.65, skip=0.551 (overall 0.553)
      Certified [10/20]: chunk_ppl=17.98, suffix_ppl=17.10, running_ppl=10.06, skip=0.515 (overall 0.541)
- ctx=16384:
      Certified [5/20]: chunk_ppl=17.40, suffix_ppl=16.36, running_ppl=9.73, skip=0.727 (overall 0.735)

### RULER aggregated

```
-    | - | -    | 0.817               | 0.833 | +0.017 | 0    | -                    | -    | - | -    | 0.650               | 0.650 | +0.000 | 0   

## 95% CIs — calibrated 4K (n per subtask)

subtask         | n  | dense% | dense CI      | cert% | cert CI       | Δ (c-d) | Δ CI          | b(d>c) | c(c>d)
----------------+----+--------+---------------+-------+---------------+---------+---------------+--------+-------
niah_single     | 20 | 100.0  | [83.9, 100.0] | 100.0 | [83.9, 100.0] | +0.0    | [+0.0, +0.0]  | 0      | 0     
niah_multikey   | 20 | 100.0  | [83.9, 100.0] | 100.0 | [83.9, 100.0] | +0.0    | [+0.0, +0.0]  | 0      | 0     
niah_multivalue | 20 | 100.0  | [83.9, 100.0] | 100.0 | [83.9, 100.0] | +0.0    | [+0.0, +0.0]  | 0      | 0     
niah_multiquery | 20 | 100.0  | [83.9, 100.0] | 100.0 | [83.9, 100.0] | +0.0    | [+0.0, +0.0]  | 0      | 0     
vt              | 20 | 85.0   | [64.0, 94.8]  | 75.0  | [53.1, 88.8]  | -10.0   | [-29.1, +9.1] | 3      | 1     
cwe             | 20 | 100.0  | [83.9, 100.0] | 100.0 | [83.9, 100.0] | +0.0    | [+0.0, +0.0]  | 0      | 0     
fwe             | 20 | 45.0   | [25.8, 65.8]  | 50.0  | [29.9, 70.1]  | +5.0    | [-4.6, +14.6] | 0      | 1     

## 95% CIs — calibrated 8K (n per subtask)

subtask         | n  | dense% | dense CI      | cert% | cert CI       | Δ (c-d) | Δ CI          | b(d>c) | c(c>d)
----------------+----+--------+---------------+-------+---------------+---------+---------------+--------+-------
niah_single     | 20 | 100.0  | [83.9, 100.0] | 100.0 | [83.9, 100.0] | +0.0    | [+0.0, +0.0]  | 0      | 0     
niah_multikey   | 20 | 100.0  | [83.9, 100.0] | 100.0 | [83.9, 100.0] | +0.0    | [+0.0, +0.0]  | 0      | 0     
niah_multivalue | 20 | 100.0  | [83.9, 100.0] | 100.0 | [83.9, 100.0] | +0.0    | [+0.0, +0.0]  | 0      | 0     
niah_multiquery | 20 | 95.0   | [76.4, 99.1]  | 95.0  | [76.4, 99.1]  | +0.0    | [+0.0, +0.0]  | 0      | 0     
vt              | 20 | 75.0   | [53.1, 88.8]  | 65.0  | [43.3, 81.9]  | -10.0   | [-23.1, +3.1] | 2      | 0     
cwe             | 20 | 90.0   | [69.9, 97.2]  | 95.0  | [76.4, 99.1]  | +5.0    | [-4.6, +14.6] | 0      | 1     
fwe             | 20 | 20.0   | [8.1, 41.6]   | 20.0  | [8.1, 41.6]   | +0.0    | [+0.0, +0.0]  | 0      | 0     

## Overall (all subtasks pooled, with paired 95% CI)

config        | dense_mean | cert_mean | Δ_mean | crit  | Δ pass% (c-d) | paired 95% CI  | n  
--------------+------------+-----------+--------+-------+---------------+----------------+----
eps0_4k       | -          | -         | -      | -     | -             | -              | -  
calibrated_4k | 0.965      | 0.960     | -0.005 | 3/140 | -0.71         | [-3.84, +2.41] | 140
eps0_8k       | -          | -         | -      | -     | -             | -              | -  
calibrated_8k | 0.922      | 0.929     | +0.006 | 2/140 | -0.71         | [-3.14, +1.71] | 140

CSV -> /workspace/DotCache/benchmarks/results/overnight_20260423/ruler_summary.csv

```
