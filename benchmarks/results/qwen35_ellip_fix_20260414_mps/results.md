# Qwen3.5 Bound Mode Compare

Compares three certified upper-bound modes under the real mixed Stage 9 serving config.

## Lane definitions

| Lane | enable_interval_bound | enable_ellipsoidal_bound |
|---|---|---|
| `spherical_only` | False | False |
| `interval` | True | False |
| `interval_ellip` | True | True |

## Summary

| Lane | avg ms/step | exact_match_vs_dense | score_ms/case | cert_stop_blocks/case | checkpoints/case |
|---|---|---|---|---|---|
| `spherical_only` | 557.10 | 0.875 | 324.24 | 384.0 | 29.2 |
| `interval` | 483.50 | 0.875 | 407.92 | 384.0 | 29.2 |
| `interval_ellip` | 881.78 | 0.875 | 709.35 | 384.0 | 29.2 |

### Speed-up vs spherical_only

- `interval`: +13.2% (483.50 vs 557.10 ms/step), cert_stop_blocks 384.0 vs 384.0 (0.0 fewer)
- `interval_ellip`: -58.3% (881.78 vs 557.10 ms/step), cert_stop_blocks 384.0 vs 384.0 (0.0 fewer)

## Bound winner fractions

Which certified upper-bound method provides the tightest value per (block, q_head) evaluation.

| Lane | spherical | interval | ellipsoidal | total evals |
|---|---|---|---|---|
| `spherical_only` | 100.0% | 0.0% | 0.0% | 196,608 |
| `interval` | 76.6% | 23.4% | 0.0% | 196,608 |
| `interval_ellip` | 40.2% | 20.9% | 38.9% | 219,983 |

## Per-case results

### aae_stage_summary

- `spherical_only`: 728.99 ms/step, exact=True, cert_stop=582 blocks, chkpts=42, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 685.56 ms/step, exact=True, cert_stop=582 blocks, chkpts=42, bound_wins: sph=79%/int=21%/ellip=0%
- `interval_ellip`: 968.81 ms/step, exact=True, cert_stop=582 blocks, chkpts=42, bound_wins: sph=40%/int=19%/ellip=41%

### bench_decode_code

- `spherical_only`: 413.04 ms/step, exact=True, cert_stop=240 blocks, chkpts=18, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 315.46 ms/step, exact=True, cert_stop=240 blocks, chkpts=18, bound_wins: sph=87%/int=13%/ellip=0%
- `interval_ellip`: 656.68 ms/step, exact=True, cert_stop=240 blocks, chkpts=18, bound_wins: sph=43%/int=12%/ellip=45%

### benchmark_report

- `spherical_only`: 643.00 ms/step, exact=True, cert_stop=582 blocks, chkpts=42, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 600.69 ms/step, exact=True, cert_stop=582 blocks, chkpts=42, bound_wins: sph=71%/int=29%/ellip=0%
- `interval_ellip`: 1031.16 ms/step, exact=True, cert_stop=582 blocks, chkpts=42, bound_wins: sph=38%/int=25%/ellip=37%

### compressed_page_rfc

- `spherical_only`: 617.69 ms/step, exact=True, cert_stop=390 blocks, chkpts=30, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 486.60 ms/step, exact=True, cert_stop=390 blocks, chkpts=30, bound_wins: sph=79%/int=21%/ellip=0%
- `interval_ellip`: 809.82 ms/step, exact=True, cert_stop=390 blocks, chkpts=30, bound_wins: sph=40%/int=19%/ellip=41%

### hip_call_flow

- `spherical_only`: 554.61 ms/step, exact=True, cert_stop=390 blocks, chkpts=30, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 556.20 ms/step, exact=True, cert_stop=390 blocks, chkpts=30, bound_wins: sph=81%/int=19%/ellip=0%
- `interval_ellip`: 922.44 ms/step, exact=True, cert_stop=390 blocks, chkpts=30, bound_wins: sph=45%/int=18%/ellip=37%

### local_layer_profiles

- `spherical_only`: 546.39 ms/step, exact=True, cert_stop=390 blocks, chkpts=30, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 491.19 ms/step, exact=True, cert_stop=390 blocks, chkpts=30, bound_wins: sph=73%/int=27%/ellip=0%
- `interval_ellip`: 875.93 ms/step, exact=True, cert_stop=390 blocks, chkpts=30, bound_wins: sph=41%/int=24%/ellip=35%

### test_attention_vs_dense

- `spherical_only`: 426.57 ms/step, exact=True, cert_stop=108 blocks, chkpts=12, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 245.79 ms/step, exact=True, cert_stop=108 blocks, chkpts=12, bound_wins: sph=74%/int=26%/ellip=0%
- `interval_ellip`: 818.30 ms/step, exact=True, cert_stop=108 blocks, chkpts=12, bound_wins: sph=40%/int=24%/ellip=37%

### turboquant_comparison_plan

- `spherical_only`: 526.51 ms/step, exact=False, cert_stop=390 blocks, chkpts=30, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 486.56 ms/step, exact=False, cert_stop=390 blocks, chkpts=30, bound_wins: sph=73%/int=27%/ellip=0%
- `interval_ellip`: 971.09 ms/step, exact=False, cert_stop=390 blocks, chkpts=30, bound_wins: sph=38%/int=23%/ellip=39%

