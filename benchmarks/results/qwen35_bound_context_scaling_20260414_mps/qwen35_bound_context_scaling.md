# Qwen3.5 Bound Mode — Context-Length Scaling

Measures how interval-bound win rate, certified-exit efficiency, and ms/step
scale with context length on Mac Mini MPS.

## Lane definitions

| Lane | enable_interval_bound | enable_ellipsoidal_bound |
|---|---|---|
| `spherical_only` | False | False |
| `interval` | True | False |

## `spherical_only` — scaling table

| tokens | blocks/layer | ms/step | exact | cert_stop_rate |
|---|---|---|---|---|
| 2,048 | 128 | 763.2 | 0.667 | 6.3% |
| 4,096 | 256 | 1353.4 | 1.000 | 6.2% |
| 8,192 | 512 | 3445.0 | 1.000 | 6.3% |

## `interval` — scaling table

| tokens | blocks/layer | ms/step | exact | cert_stop_rate | sph_win | int_win |
|---|---|---|---|---|----|------|
| 2,048 | 128 | 905.9 | 0.667 | 6.2% | 76.8% | 23.2% |
| 4,096 | 256 | 1422.8 | 1.000 | 6.2% | 83.9% | 16.1% |
| 8,192 | 512 | 4002.1 | 1.000 | 6.3% | 78.6% | 21.4% |

## Interval speedup vs spherical by context length

| tokens | spherical ms/step | interval ms/step | speedup | int_win_frac |
|---|---|---|---|---|
| 2,048 | 763.2 | 905.9 | -18.7% | 23.2% |
| 4,096 | 1353.4 | 1422.8 | -5.1% | 16.1% |
| 8,192 | 3445.0 | 4002.1 | -16.2% | 21.4% |

## Per-case results

### benchmark_report_2k

- `interval` (2048 tok, 128 blk/layer): 922.6 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=74%/int=26%
- `spherical_only` (2048 tok, 128 blk/layer): 756.2 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=100%/int=0%

### benchmark_report_4k

- `interval` (4096 tok, 256 blk/layer): 1351.3 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=83%/int=17%
- `spherical_only` (4096 tok, 256 blk/layer): 1389.3 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=100%/int=0%

### benchmark_report_8k

- `interval` (8192 tok, 512 blk/layer): 3876.6 ms/step, exact=True, cert_stop_rate=6.3%, bound: sph=82%/int=18%
- `spherical_only` (8192 tok, 512 blk/layer): 2592.5 ms/step, exact=True, cert_stop_rate=6.3%, bound: sph=100%/int=0%

### perf_journal_2k

- `interval` (2048 tok, 128 blk/layer): 973.4 ms/step, exact=False, cert_stop_rate=6.2%, bound: sph=79%/int=21%
- `spherical_only` (2048 tok, 128 blk/layer): 777.1 ms/step, exact=False, cert_stop_rate=6.2%, bound: sph=100%/int=0%

### perf_journal_4k

- `interval` (4096 tok, 256 blk/layer): 1395.7 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=83%/int=17%
- `spherical_only` (4096 tok, 256 blk/layer): 1345.6 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=100%/int=0%

### perf_journal_8k

- `interval` (8192 tok, 512 blk/layer): 4127.6 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=75%/int=25%
- `spherical_only` (8192 tok, 512 blk/layer): 4297.5 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=100%/int=0%

### qwen35_stage9_thesis_2k

- `interval` (2048 tok, 128 blk/layer): 821.5 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=78%/int=22%
- `spherical_only` (2048 tok, 128 blk/layer): 756.4 ms/step, exact=True, cert_stop_rate=6.3%, bound: sph=100%/int=0%

### qwen35_stage9_thesis_4k

- `interval` (4096 tok, 256 blk/layer): 1521.3 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=86%/int=14%
- `spherical_only` (4096 tok, 256 blk/layer): 1325.2 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=100%/int=0%

