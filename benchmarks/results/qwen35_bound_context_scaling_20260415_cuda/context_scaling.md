# Qwen3.5 Bound Mode — Context-Length Scaling

Measures how interval-bound win rate, certified-exit efficiency, and ms/step
scale with context length on Mac Mini MPS.

## Lane definitions

| Lane | enable_interval_bound | enable_ellipsoidal_bound |
|---|---|---|
| `spherical_only` | False | False |
| `interval` | True | False |
| `interval_ellip` | True | True |

## `spherical_only` — scaling table

| tokens | blocks/layer | ms/step | exact | cert_stop_rate |
|---|---|---|---|---|
| 2,048 | 128 | 373.1 | 1.000 | 6.3% |
| 4,096 | 256 | 699.9 | 1.000 | 6.2% |
| 8,192 | 512 | 1747.9 | 1.000 | 6.3% |

## `interval` — scaling table

| tokens | blocks/layer | ms/step | exact | cert_stop_rate | sph_win | int_win |
|---|---|---|---|---|----|------|
| 2,048 | 128 | 520.0 | 1.000 | 6.3% | 76.8% | 23.2% |
| 4,096 | 256 | 550.2 | 1.000 | 6.2% | 83.9% | 16.1% |
| 8,192 | 512 | 1020.1 | 1.000 | 6.3% | 78.6% | 21.4% |

## `interval_ellip` — scaling table

| tokens | blocks/layer | ms/step | exact | cert_stop_rate | sph_win | int_win |
|---|---|---|---|---|----|------|
| 2,048 | 128 | 323.5 | 1.000 | 6.3% | 40.4% | 20.6% |
| 4,096 | 256 | 560.8 | 1.000 | 6.2% | 42.3% | 14.7% |
| 8,192 | 512 | 1032.4 | 1.000 | 6.3% | 40.1% | 19.2% |

## Interval speedup vs spherical by context length

| tokens | spherical ms/step | interval ms/step | speedup | int_win_frac |
|---|---|---|---|---|
| 2,048 | 373.1 | 520.0 | -39.4% | 23.2% |
| 4,096 | 699.9 | 550.2 | +21.4% | 16.1% |
| 8,192 | 1747.9 | 1020.1 | +41.6% | 21.4% |

## Per-case results

### benchmark_report_2k

- `interval` (2048 tok, 128 blk/layer): 548.9 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=74%/int=26%
- `interval_ellip` (2048 tok, 128 blk/layer): 322.2 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=38%/int=23%
- `spherical_only` (2048 tok, 128 blk/layer): 316.5 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=100%/int=0%

### benchmark_report_4k

- `interval` (4096 tok, 256 blk/layer): 549.4 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=83%/int=17%
- `interval_ellip` (4096 tok, 256 blk/layer): 561.5 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=41%/int=16%
- `spherical_only` (4096 tok, 256 blk/layer): 547.0 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=100%/int=0%

### benchmark_report_8k

- `interval` (8192 tok, 512 blk/layer): 1019.8 ms/step, exact=True, cert_stop_rate=6.3%, bound: sph=82%/int=18%
- `interval_ellip` (8192 tok, 512 blk/layer): 1028.3 ms/step, exact=True, cert_stop_rate=6.3%, bound: sph=41%/int=16%
- `spherical_only` (8192 tok, 512 blk/layer): 1788.7 ms/step, exact=True, cert_stop_rate=6.3%, bound: sph=100%/int=0%

### perf_journal_2k

- `interval` (2048 tok, 128 blk/layer): 533.7 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=79%/int=21%
- `interval_ellip` (2048 tok, 128 blk/layer): 321.4 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=42%/int=19%
- `spherical_only` (2048 tok, 128 blk/layer): 495.9 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=100%/int=0%

### perf_journal_4k

- `interval` (4096 tok, 256 blk/layer): 551.2 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=83%/int=17%
- `interval_ellip` (4096 tok, 256 blk/layer): 558.3 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=41%/int=16%
- `spherical_only` (4096 tok, 256 blk/layer): 539.9 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=100%/int=0%

### perf_journal_8k

- `interval` (8192 tok, 512 blk/layer): 1020.4 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=75%/int=25%
- `interval_ellip` (8192 tok, 512 blk/layer): 1036.5 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=39%/int=22%
- `spherical_only` (8192 tok, 512 blk/layer): 1707.1 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=100%/int=0%

### qwen35_stage9_thesis_2k

- `interval` (2048 tok, 128 blk/layer): 477.5 ms/step, exact=True, cert_stop_rate=6.3%, bound: sph=78%/int=22%
- `interval_ellip` (2048 tok, 128 blk/layer): 327.0 ms/step, exact=True, cert_stop_rate=6.3%, bound: sph=41%/int=20%
- `spherical_only` (2048 tok, 128 blk/layer): 306.8 ms/step, exact=True, cert_stop_rate=6.3%, bound: sph=100%/int=0%

### qwen35_stage9_thesis_4k

- `interval` (4096 tok, 256 blk/layer): 550.2 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=86%/int=14%
- `interval_ellip` (4096 tok, 256 blk/layer): 562.6 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=44%/int=13%
- `spherical_only` (4096 tok, 256 blk/layer): 1013.0 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=100%/int=0%

