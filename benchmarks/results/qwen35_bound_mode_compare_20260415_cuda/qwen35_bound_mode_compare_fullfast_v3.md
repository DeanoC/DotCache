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
| `spherical_only` | 153.43 | 0.875 | 107.70 | 384.0 | 0.0 |
| `interval` | 156.23 | 0.875 | 129.37 | 384.0 | 0.0 |
| `interval_ellip` | 164.44 | 0.875 | 156.27 | 384.0 | 0.0 |

### Speed-up vs spherical_only

- `interval`: -1.8% (156.23 vs 153.43 ms/step), cert_stop_blocks 384.0 vs 384.0 (0.0 fewer)
- `interval_ellip`: -7.2% (164.44 vs 153.43 ms/step), cert_stop_blocks 384.0 vs 384.0 (0.0 fewer)

## Bound winner fractions

Which certified upper-bound method provides the tightest value per (block, q_head) evaluation.

| Lane | spherical | interval | ellipsoidal | total evals |
|---|---|---|---|---|
| `spherical_only` | 100.0% | 0.0% | 0.0% | 196,608 |
| `interval` | 76.6% | 23.4% | 0.0% | 196,608 |
| `interval_ellip` | 40.2% | 20.9% | 38.9% | 219,956 |

## Per-case results

### aae_stage_summary

- `spherical_only`: 202.04 ms/step, exact=True, cert_stop=582 blocks, chkpts=0, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 202.99 ms/step, exact=True, cert_stop=582 blocks, chkpts=0, bound_wins: sph=79%/int=21%/ellip=0%
- `interval_ellip`: 212.42 ms/step, exact=True, cert_stop=582 blocks, chkpts=0, bound_wins: sph=40%/int=19%/ellip=41%

### bench_decode_code

- `spherical_only`: 111.39 ms/step, exact=True, cert_stop=240 blocks, chkpts=0, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 115.01 ms/step, exact=True, cert_stop=240 blocks, chkpts=0, bound_wins: sph=87%/int=13%/ellip=0%
- `interval_ellip`: 122.33 ms/step, exact=True, cert_stop=240 blocks, chkpts=0, bound_wins: sph=43%/int=12%/ellip=45%

### benchmark_report

- `spherical_only`: 200.00 ms/step, exact=True, cert_stop=582 blocks, chkpts=0, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 204.99 ms/step, exact=True, cert_stop=582 blocks, chkpts=0, bound_wins: sph=71%/int=29%/ellip=0%
- `interval_ellip`: 209.98 ms/step, exact=True, cert_stop=582 blocks, chkpts=0, bound_wins: sph=38%/int=25%/ellip=37%

### compressed_page_rfc

- `spherical_only`: 155.37 ms/step, exact=True, cert_stop=390 blocks, chkpts=0, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 157.53 ms/step, exact=True, cert_stop=390 blocks, chkpts=0, bound_wins: sph=79%/int=21%/ellip=0%
- `interval_ellip`: 169.41 ms/step, exact=True, cert_stop=390 blocks, chkpts=0, bound_wins: sph=40%/int=19%/ellip=41%

### hip_call_flow

- `spherical_only`: 154.85 ms/step, exact=True, cert_stop=390 blocks, chkpts=0, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 158.18 ms/step, exact=True, cert_stop=390 blocks, chkpts=0, bound_wins: sph=80%/int=20%/ellip=0%
- `interval_ellip`: 166.07 ms/step, exact=True, cert_stop=390 blocks, chkpts=0, bound_wins: sph=45%/int=18%/ellip=37%

### local_layer_profiles

- `spherical_only`: 154.54 ms/step, exact=True, cert_stop=390 blocks, chkpts=0, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 158.09 ms/step, exact=True, cert_stop=390 blocks, chkpts=0, bound_wins: sph=73%/int=27%/ellip=0%
- `interval_ellip`: 166.58 ms/step, exact=True, cert_stop=390 blocks, chkpts=0, bound_wins: sph=41%/int=24%/ellip=35%

### test_attention_vs_dense

- `spherical_only`: 90.44 ms/step, exact=True, cert_stop=108 blocks, chkpts=0, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 91.72 ms/step, exact=True, cert_stop=108 blocks, chkpts=0, bound_wins: sph=74%/int=26%/ellip=0%
- `interval_ellip`: 99.31 ms/step, exact=True, cert_stop=108 blocks, chkpts=0, bound_wins: sph=40%/int=24%/ellip=37%

### turboquant_comparison_plan

- `spherical_only`: 158.83 ms/step, exact=False, cert_stop=390 blocks, chkpts=0, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 161.37 ms/step, exact=False, cert_stop=390 blocks, chkpts=0, bound_wins: sph=73%/int=27%/ellip=0%
- `interval_ellip`: 169.39 ms/step, exact=False, cert_stop=390 blocks, chkpts=0, bound_wins: sph=38%/int=23%/ellip=39%

