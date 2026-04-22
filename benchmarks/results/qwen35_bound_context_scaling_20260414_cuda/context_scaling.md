# Qwen3.5 Bound Mode Context-Length Scaling

Sweeps context lengths [1K–32K] for certified upper-bound modes on CUDA.
16K and 32K are included (excluded from MPS due to dense-SDPA OOM at prefill).

## Lane definitions

| Lane | enable_interval_bound | enable_ellipsoidal_bound |
|---|---|---|
| `spherical_only` | False | False |
| `interval` | True | False |
| `interval_ellip` | True | True |

## ms/step by context length

| context | `spherical_only` | `interval` | `interval_ellip` |
|---|---|---|---|
| 1,024 | 253.51 | 260.18 | 277.87 |
| 2,048 | 398.90 | 405.19 | 429.71 |
| 4,096 | 684.47 | 696.68 | 719.85 |
| 8,192 | 1411.94 | 1259.72 | 1276.74 |
| 16,384 | 2851.12 | 2856.22 | 2866.70 |
| 32,768 | 5847.41 | 5887.33 | 5907.40 |

## Δ ms/step vs spherical_only (negative = faster)

| context | `interval` | `interval_ellip` |
|---|---|---|
| 1,024 | +2.6% | +9.6% |
| 2,048 | +1.6% | +7.7% |
| 4,096 | +1.8% | +5.2% |
| 8,192 | -10.8% | -9.6% |
| 16,384 | +0.2% | +0.5% |
| 32,768 | +0.7% | +1.0% |

## cert_stop_rate by context length

| context | `spherical_only` | `interval` | `interval_ellip` |
|---|---|---|---|
| 1,024 | 1.000 | 1.000 | 1.000 |
| 2,048 | 1.000 | 1.000 | 1.000 |
| 4,096 | 1.000 | 1.000 | 1.000 |
| 8,192 | 1.000 | 1.000 | 1.000 |
| 16,384 | 1.000 | 1.000 | 1.000 |
| 32,768 | 1.000 | 1.000 | 1.000 |

## score_ms/step by context length

| context | `spherical_only` | `interval` | `interval_ellip` |
|---|---|---|---|
| 1,024 | 19.71 | 26.07 | 37.05 |
| 2,048 | 22.24 | 28.86 | 39.94 |
| 4,096 | 27.17 | 34.32 | 45.12 |
| 8,192 | 36.71 | 43.22 | 54.15 |
| 16,384 | 55.80 | 61.85 | 72.70 |
| 32,768 | 91.71 | 98.37 | 110.08 |

## Notes

- cert_stop_rate is measured by running single-step harness calls (N=decode_steps)
  and checking whether any FA layer issued a certified exit in each step.
- ms/step is from a separate N-step run to amortise CUDA sync overhead.
- Dense reference is skipped for contexts above --max-dense-length (default: none).
