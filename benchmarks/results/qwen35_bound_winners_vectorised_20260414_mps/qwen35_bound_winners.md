# Qwen3.5 Bound Mode Compare

Compares three certified upper-bound modes under the real mixed Stage 9 serving config.

## Lane definitions

| Lane | enable_interval_bound | enable_ellipsoidal_bound |
|---|---|---|
| `interval` | True | False |
| `interval_ellip` | True | True |

## Summary

| Lane | avg ms/step | exact_match_vs_dense | score_ms/case | cert_stop_blocks/case | checkpoints/case |
|---|---|---|---|---|---|
| `interval` | 570.68 | 0.875 | 482.36 | 384.0 | 29.2 |
| `interval_ellip` | 832.69 | 0.875 | 604.64 | 384.0 | 29.2 |

## Bound winner fractions

Which certified upper-bound method provides the tightest value per (block, q_head) evaluation.

| Lane | spherical | interval | ellipsoidal | total evals |
|---|---|---|---|---|
| `interval` | 76.6% | 23.4% | 0.0% | 196,608 |
| `interval_ellip` | 40.2% | 20.9% | 38.9% | 219,983 |

## Per-case results

### aae_stage_summary

- `interval`: 740.50 ms/step, exact=True, cert_stop=582 blocks, chkpts=42, bound_wins: sph=79%/int=21%/ellip=0%
- `interval_ellip`: 1063.41 ms/step, exact=True, cert_stop=582 blocks, chkpts=42, bound_wins: sph=40%/int=19%/ellip=41%

### bench_decode_code

- `interval`: 382.62 ms/step, exact=True, cert_stop=240 blocks, chkpts=18, bound_wins: sph=87%/int=13%/ellip=0%
- `interval_ellip`: 652.86 ms/step, exact=True, cert_stop=240 blocks, chkpts=18, bound_wins: sph=43%/int=12%/ellip=45%

### benchmark_report

- `interval`: 605.01 ms/step, exact=True, cert_stop=582 blocks, chkpts=42, bound_wins: sph=71%/int=29%/ellip=0%
- `interval_ellip`: 884.16 ms/step, exact=True, cert_stop=582 blocks, chkpts=42, bound_wins: sph=38%/int=25%/ellip=37%

### compressed_page_rfc

- `interval`: 580.82 ms/step, exact=True, cert_stop=390 blocks, chkpts=30, bound_wins: sph=79%/int=21%/ellip=0%
- `interval_ellip`: 749.96 ms/step, exact=True, cert_stop=390 blocks, chkpts=30, bound_wins: sph=40%/int=19%/ellip=41%

### hip_call_flow

- `interval`: 528.12 ms/step, exact=True, cert_stop=390 blocks, chkpts=30, bound_wins: sph=81%/int=19%/ellip=0%
- `interval_ellip`: 757.49 ms/step, exact=True, cert_stop=390 blocks, chkpts=30, bound_wins: sph=45%/int=18%/ellip=37%

### local_layer_profiles

- `interval`: 588.44 ms/step, exact=True, cert_stop=390 blocks, chkpts=30, bound_wins: sph=73%/int=27%/ellip=0%
- `interval_ellip`: 830.68 ms/step, exact=True, cert_stop=390 blocks, chkpts=30, bound_wins: sph=41%/int=24%/ellip=35%

### test_attention_vs_dense

- `interval`: 451.23 ms/step, exact=True, cert_stop=108 blocks, chkpts=12, bound_wins: sph=74%/int=26%/ellip=0%
- `interval_ellip`: 783.94 ms/step, exact=True, cert_stop=108 blocks, chkpts=12, bound_wins: sph=40%/int=24%/ellip=37%

### turboquant_comparison_plan

- `interval`: 688.70 ms/step, exact=False, cert_stop=390 blocks, chkpts=30, bound_wins: sph=73%/int=27%/ellip=0%
- `interval_ellip`: 939.01 ms/step, exact=False, cert_stop=390 blocks, chkpts=30, bound_wins: sph=38%/int=23%/ellip=39%

