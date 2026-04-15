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
| 32,768 | 2048 | 4050.4 | 0.000 | 12.5% |

## `interval` — scaling table

| tokens | blocks/layer | ms/step | exact | cert_stop_rate | sph_win | int_win |
|---|---|---|---|---|----|------|
| 32,768 | 2048 | 6224.1 | 0.000 | 12.5% | 82.9% | 17.1% |

## `interval_ellip` — scaling table

| tokens | blocks/layer | ms/step | exact | cert_stop_rate | sph_win | int_win |
|---|---|---|---|---|----|------|
| 32,768 | 2048 | 3978.9 | 0.000 | 12.5% | 38.4% | 15.3% |

## Interval speedup vs spherical by context length

| tokens | spherical ms/step | interval ms/step | speedup | int_win_frac |
|---|---|---|---|---|
| 32,768 | 4050.4 | 6224.1 | -53.7% | 17.1% |

## Per-case results

### perf_journal_32k

- `interval` (32768 tok, 2048 blk/layer): 6224.1 ms/step, exact=skipped, cert_stop_rate=12.5%, bound: sph=83%/int=17%
- `interval_ellip` (32768 tok, 2048 blk/layer): 3978.9 ms/step, exact=skipped, cert_stop_rate=12.5%, bound: sph=38%/int=15%
- `spherical_only` (32768 tok, 2048 blk/layer): 4050.4 ms/step, exact=skipped, cert_stop_rate=12.5%, bound: sph=100%/int=0%

