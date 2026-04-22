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
| 2,048 | 128 | 256.8 | 1.000 | 6.3% |
| 4,096 | 256 | 449.2 | 1.000 | 6.2% |
| 8,192 | 512 | 811.8 | 1.000 | 6.3% |

## `interval` — scaling table

| tokens | blocks/layer | ms/step | exact | cert_stop_rate | sph_win | int_win |
|---|---|---|---|---|----|------|
| 2,048 | 128 | 255.4 | 1.000 | 6.3% | 76.8% | 23.2% |
| 4,096 | 256 | 441.4 | 1.000 | 6.2% | 83.9% | 16.1% |
| 8,192 | 512 | 812.0 | 1.000 | 6.3% | 78.6% | 21.4% |

## `interval_ellip` — scaling table

| tokens | blocks/layer | ms/step | exact | cert_stop_rate | sph_win | int_win |
|---|---|---|---|---|----|------|
| 2,048 | 128 | 262.2 | 1.000 | 6.3% | 40.4% | 20.6% |
| 4,096 | 256 | 450.0 | 1.000 | 6.2% | 42.3% | 14.7% |
| 8,192 | 512 | 822.9 | 1.000 | 6.3% | 40.1% | 19.2% |

## Interval speedup vs spherical by context length

| tokens | spherical ms/step | interval ms/step | speedup | int_win_frac |
|---|---|---|---|---|
| 2,048 | 256.8 | 255.4 | +0.6% | 23.2% |
| 4,096 | 449.2 | 441.4 | +1.7% | 16.1% |
| 8,192 | 811.8 | 812.0 | -0.0% | 21.4% |

## Per-case results

### benchmark_report_2k

- `interval` (2048 tok, 128 blk/layer): 256.1 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=74%/int=26%
- `interval_ellip` (2048 tok, 128 blk/layer): 262.0 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=38%/int=23%
- `spherical_only` (2048 tok, 128 blk/layer): 262.1 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=100%/int=0%

### benchmark_report_4k

- `interval` (4096 tok, 256 blk/layer): 441.6 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=83%/int=17%
- `interval_ellip` (4096 tok, 256 blk/layer): 450.9 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=41%/int=16%
- `spherical_only` (4096 tok, 256 blk/layer): 445.1 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=100%/int=0%

### benchmark_report_8k

- `interval` (8192 tok, 512 blk/layer): 814.3 ms/step, exact=True, cert_stop_rate=6.3%, bound: sph=82%/int=18%
- `interval_ellip` (8192 tok, 512 blk/layer): 822.2 ms/step, exact=True, cert_stop_rate=6.3%, bound: sph=41%/int=16%
- `spherical_only` (8192 tok, 512 blk/layer): 814.2 ms/step, exact=True, cert_stop_rate=6.3%, bound: sph=100%/int=0%

### perf_journal_2k

- `interval` (2048 tok, 128 blk/layer): 257.1 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=79%/int=21%
- `interval_ellip` (2048 tok, 128 blk/layer): 262.6 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=42%/int=19%
- `spherical_only` (2048 tok, 128 blk/layer): 252.1 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=100%/int=0%

### perf_journal_4k

- `interval` (4096 tok, 256 blk/layer): 441.8 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=83%/int=17%
- `interval_ellip` (4096 tok, 256 blk/layer): 449.5 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=41%/int=16%
- `spherical_only` (4096 tok, 256 blk/layer): 459.2 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=100%/int=0%

### perf_journal_8k

- `interval` (8192 tok, 512 blk/layer): 809.8 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=75%/int=25%
- `interval_ellip` (8192 tok, 512 blk/layer): 823.6 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=39%/int=22%
- `spherical_only` (8192 tok, 512 blk/layer): 809.4 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=100%/int=0%

### qwen35_stage9_thesis_2k

- `interval` (2048 tok, 128 blk/layer): 253.0 ms/step, exact=True, cert_stop_rate=6.3%, bound: sph=78%/int=22%
- `interval_ellip` (2048 tok, 128 blk/layer): 262.2 ms/step, exact=True, cert_stop_rate=6.3%, bound: sph=41%/int=20%
- `spherical_only` (2048 tok, 128 blk/layer): 256.2 ms/step, exact=True, cert_stop_rate=6.3%, bound: sph=100%/int=0%

### qwen35_stage9_thesis_4k

- `interval` (4096 tok, 256 blk/layer): 440.9 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=86%/int=14%
- `interval_ellip` (4096 tok, 256 blk/layer): 449.6 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=44%/int=13%
- `spherical_only` (4096 tok, 256 blk/layer): 443.4 ms/step, exact=True, cert_stop_rate=6.2%, bound: sph=100%/int=0%

