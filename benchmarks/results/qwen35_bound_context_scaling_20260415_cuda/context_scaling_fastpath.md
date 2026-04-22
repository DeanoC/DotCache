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
| 2,048 | 128 | 461.1 | 0.667 | 6.3% |
| 4,096 | 256 | 802.8 | 1.000 | 6.2% |
| 8,192 | 512 | 1066.3 | 1.000 | 6.3% |

## `interval` — scaling table

| tokens | blocks/layer | ms/step | exact | cert_stop_rate | sph_win | int_win |
|---|---|---|---|---|----|------|
| 2,048 | 128 | 249.3 | 0.667 | 6.3% | 76.8% | 23.2% |
| 4,096 | 256 | 434.1 | 1.000 | 6.2% | 83.9% | 16.1% |
| 8,192 | 512 | 792.1 | 1.000 | 6.3% | 78.6% | 21.4% |

## `interval_ellip` — scaling table

| tokens | blocks/layer | ms/step | exact | cert_stop_rate | sph_win | int_win |
|---|---|---|---|---|----|------|
| 2,048 | 128 | 257.7 | 0.667 | 6.3% | 40.4% | 20.6% |
| 4,096 | 256 | 440.7 | 1.000 | 6.2% | 42.3% | 14.7% |
| 8,192 | 512 | 800.0 | 1.000 | 6.3% | 40.1% | 19.2% |

## Interval speedup vs spherical by context length

| tokens | spherical ms/step | interval ms/step | speedup | int_win_frac |
|---|---|---|---|---|
| 2,048 | 461.1 | 249.3 | +45.9% | 23.2% |
| 4,096 | 802.8 | 434.1 | +45.9% | 16.1% |
| 8,192 | 1066.3 | 792.1 | +25.7% | 21.4% |

## Per-case results

### benchmark_report_2k

- `interval` (2048 tok, 128 blk/layer): 249.7 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=74%/int=26%
- `interval_ellip` (2048 tok, 128 blk/layer): 256.0 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=38%/int=23%
- `spherical_only` (2048 tok, 128 blk/layer): 483.8 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=100%/int=0%

### benchmark_report_4k

- `interval` (4096 tok, 256 blk/layer): 433.1 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=83%/int=17%
- `interval_ellip` (4096 tok, 256 blk/layer): 441.5 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=41%/int=16%
- `spherical_only` (4096 tok, 256 blk/layer): 820.5 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=100%/int=0%

### benchmark_report_8k

- `interval` (8192 tok, 512 blk/layer): 791.5 ms/step, exact=True, cert_stop_rate=6.3%, bound: sph=82%/int=18%
- `interval_ellip` (8192 tok, 512 blk/layer): 800.3 ms/step, exact=True, cert_stop_rate=6.3%, bound: sph=41%/int=16%
- `spherical_only` (8192 tok, 512 blk/layer): 805.7 ms/step, exact=True, cert_stop_rate=6.3%, bound: sph=100%/int=0%

### perf_journal_2k

- `interval` (2048 tok, 128 blk/layer): 249.4 ms/step, exact=False, cert_stop_rate=6.2%, bound: sph=79%/int=21%
- `interval_ellip` (2048 tok, 128 blk/layer): 260.6 ms/step, exact=False, cert_stop_rate=6.2%, bound: sph=42%/int=19%
- `spherical_only` (2048 tok, 128 blk/layer): 442.0 ms/step, exact=False, cert_stop_rate=6.2%, bound: sph=100%/int=0%

### perf_journal_4k

- `interval` (4096 tok, 256 blk/layer): 434.5 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=83%/int=17%
- `interval_ellip` (4096 tok, 256 blk/layer): 441.8 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=41%/int=16%
- `spherical_only` (4096 tok, 256 blk/layer): 796.9 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=100%/int=0%

### perf_journal_8k

- `interval` (8192 tok, 512 blk/layer): 792.7 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=75%/int=25%
- `interval_ellip` (8192 tok, 512 blk/layer): 799.7 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=39%/int=22%
- `spherical_only` (8192 tok, 512 blk/layer): 1327.0 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=100%/int=0%

### qwen35_stage9_thesis_2k

- `interval` (2048 tok, 128 blk/layer): 248.9 ms/step, exact=True, cert_stop_rate=6.3%, bound: sph=78%/int=22%
- `interval_ellip` (2048 tok, 128 blk/layer): 256.3 ms/step, exact=True, cert_stop_rate=6.3%, bound: sph=41%/int=20%
- `spherical_only` (2048 tok, 128 blk/layer): 457.5 ms/step, exact=True, cert_stop_rate=6.3%, bound: sph=100%/int=0%

### qwen35_stage9_thesis_4k

- `interval` (4096 tok, 256 blk/layer): 434.8 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=86%/int=14%
- `interval_ellip` (4096 tok, 256 blk/layer): 438.8 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=44%/int=13%
- `spherical_only` (4096 tok, 256 blk/layer): 791.0 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=100%/int=0%

