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
| `spherical_only` | 155.22 | 0.875 | 426.08 | 386.2 | 0.0 |
| `interval` | 157.52 | 0.875 | 510.23 | 386.2 | 0.0 |
| `interval_ellip` | 165.58 | 0.875 | 616.77 | 391.5 | 0.0 |

### Speed-up vs spherical_only

- `interval`: -1.5% (157.52 vs 155.22 ms/step), cert_stop_blocks 386.2 vs 386.2 (0.0 fewer)
- `interval_ellip`: -6.7% (165.58 vs 155.22 ms/step), cert_stop_blocks 391.5 vs 386.2 (-5.2 fewer)

## Bound winner fractions

Which certified upper-bound method provides the tightest value per (block, q_head) evaluation.

| Lane | spherical | interval | ellipsoidal | total evals |
|---|---|---|---|---|
| `spherical_only` | 100.0% | 0.0% | 0.0% | 793,440 |
| `interval` | 76.7% | 23.3% | 0.0% | 793,440 |
| `interval_ellip` | 40.5% | 20.9% | 38.6% | 883,600 |

## Per-case results

### aae_stage_summary

- `spherical_only`: 200.24 ms/step, exact=True, cert_stop=588 blocks, chkpts=0, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 203.01 ms/step, exact=True, cert_stop=588 blocks, chkpts=0, bound_wins: sph=77%/int=23%/ellip=0%
- `interval_ellip`: 209.81 ms/step, exact=True, cert_stop=588 blocks, chkpts=0, bound_wins: sph=39%/int=21%/ellip=40%

### bench_decode_code

- `spherical_only`: 116.87 ms/step, exact=True, cert_stop=210 blocks, chkpts=0, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 118.62 ms/step, exact=True, cert_stop=210 blocks, chkpts=0, bound_wins: sph=85%/int=15%/ellip=0%
- `interval_ellip`: 125.49 ms/step, exact=True, cert_stop=252 blocks, chkpts=0, bound_wins: sph=44%/int=14%/ellip=42%

### benchmark_report

- `spherical_only`: 199.74 ms/step, exact=True, cert_stop=588 blocks, chkpts=0, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 202.82 ms/step, exact=True, cert_stop=588 blocks, chkpts=0, bound_wins: sph=72%/int=28%/ellip=0%
- `interval_ellip`: 209.62 ms/step, exact=True, cert_stop=588 blocks, chkpts=0, bound_wins: sph=38%/int=25%/ellip=37%

### compressed_page_rfc

- `spherical_only`: 157.35 ms/step, exact=True, cert_stop=396 blocks, chkpts=0, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 161.59 ms/step, exact=True, cert_stop=396 blocks, chkpts=0, bound_wins: sph=78%/int=22%/ellip=0%
- `interval_ellip`: 168.29 ms/step, exact=True, cert_stop=396 blocks, chkpts=0, bound_wins: sph=40%/int=20%/ellip=40%

### hip_call_flow

- `spherical_only`: 158.05 ms/step, exact=True, cert_stop=396 blocks, chkpts=0, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 158.32 ms/step, exact=True, cert_stop=396 blocks, chkpts=0, bound_wins: sph=79%/int=21%/ellip=0%
- `interval_ellip`: 168.78 ms/step, exact=True, cert_stop=396 blocks, chkpts=0, bound_wins: sph=45%/int=19%/ellip=36%

### local_layer_profiles

- `spherical_only`: 158.24 ms/step, exact=True, cert_stop=396 blocks, chkpts=0, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 160.27 ms/step, exact=True, cert_stop=396 blocks, chkpts=0, bound_wins: sph=76%/int=24%/ellip=0%
- `interval_ellip`: 169.58 ms/step, exact=True, cert_stop=396 blocks, chkpts=0, bound_wins: sph=42%/int=22%/ellip=36%

### test_attention_vs_dense

- `spherical_only`: 93.54 ms/step, exact=True, cert_stop=120 blocks, chkpts=0, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 96.13 ms/step, exact=True, cert_stop=120 blocks, chkpts=0, bound_wins: sph=77%/int=23%/ellip=0%
- `interval_ellip`: 104.65 ms/step, exact=True, cert_stop=120 blocks, chkpts=0, bound_wins: sph=40%/int=21%/ellip=39%

### turboquant_comparison_plan

- `spherical_only`: 157.77 ms/step, exact=False, cert_stop=396 blocks, chkpts=0, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 159.37 ms/step, exact=False, cert_stop=396 blocks, chkpts=0, bound_wins: sph=76%/int=24%/ellip=0%
- `interval_ellip`: 168.45 ms/step, exact=False, cert_stop=396 blocks, chkpts=0, bound_wins: sph=38%/int=21%/ellip=40%

