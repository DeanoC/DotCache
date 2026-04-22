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
| 16,384 | 1024 | 1827.3 | 0.000 | 6.3% |

## `interval` — scaling table

| tokens | blocks/layer | ms/step | exact | cert_stop_rate | sph_win | int_win |
|---|---|---|---|---|----|------|
| 16,384 | 1024 | 1899.2 | 0.000 | 6.3% | 80.6% | 19.4% |

## `interval_ellip` — scaling table

| tokens | blocks/layer | ms/step | exact | cert_stop_rate | sph_win | int_win |
|---|---|---|---|---|----|------|
| 16,384 | 1024 | 1889.9 | 0.000 | 6.3% | 39.6% | 17.4% |

## Interval speedup vs spherical by context length

| tokens | spherical ms/step | interval ms/step | speedup | int_win_frac |
|---|---|---|---|---|
| 16,384 | 1827.3 | 1899.2 | -3.9% | 19.4% |

## Per-case results

### perf_journal_16k

- `interval` (16384 tok, 1024 blk/layer): 1899.2 ms/step, exact=skipped, cert_stop_rate=6.3%, bound: sph=81%/int=19%
- `interval_ellip` (16384 tok, 1024 blk/layer): 1889.9 ms/step, exact=skipped, cert_stop_rate=6.3%, bound: sph=40%/int=17%
- `spherical_only` (16384 tok, 1024 blk/layer): 1827.3 ms/step, exact=skipped, cert_stop_rate=6.3%, bound: sph=100%/int=0%

