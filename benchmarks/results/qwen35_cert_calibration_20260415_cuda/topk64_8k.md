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
| `spherical_only` | 307.81 | 0.889 | 135.08 | 683.3 | 0.0 |
| `interval` | 418.41 | 0.889 | 165.03 | 683.3 | 0.0 |
| `interval_ellip` | 428.65 | 0.889 | 191.35 | 683.3 | 0.0 |

### Speed-up vs spherical_only

- `interval`: -35.9% (418.41 vs 307.81 ms/step), cert_stop_blocks 683.3 vs 683.3 (0.0 fewer)
- `interval_ellip`: -39.3% (428.65 vs 307.81 ms/step), cert_stop_blocks 683.3 vs 683.3 (0.0 fewer)

## Bound winner fractions

Which certified upper-bound method provides the tightest value per (block, q_head) evaluation.

| Lane | spherical | interval | ellipsoidal | total evals |
|---|---|---|---|---|
| `spherical_only` | 100.0% | 0.0% | 0.0% | 393,600 |
| `interval` | 75.6% | 24.4% | 0.0% | 393,600 |
| `interval_ellip` | 39.9% | 21.7% | 38.3% | 441,002 |

## Per-case results

### aae_stage_summary

- `spherical_only`: 423.94 ms/step, exact=True, cert_stop=582 blocks, chkpts=0, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 388.19 ms/step, exact=True, cert_stop=582 blocks, chkpts=0, bound_wins: sph=79%/int=21%/ellip=0%
- `interval_ellip`: 349.28 ms/step, exact=True, cert_stop=582 blocks, chkpts=0, bound_wins: sph=40%/int=19%/ellip=41%

### bench_decode_code

- `spherical_only`: 162.35 ms/step, exact=True, cert_stop=240 blocks, chkpts=0, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 170.81 ms/step, exact=True, cert_stop=240 blocks, chkpts=0, bound_wins: sph=87%/int=13%/ellip=0%
- `interval_ellip`: 195.81 ms/step, exact=True, cert_stop=240 blocks, chkpts=0, bound_wins: sph=43%/int=12%/ellip=45%

### benchmark_report

- `spherical_only`: 200.65 ms/step, exact=True, cert_stop=582 blocks, chkpts=0, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 203.91 ms/step, exact=True, cert_stop=582 blocks, chkpts=0, bound_wins: sph=71%/int=29%/ellip=0%
- `interval_ellip`: 212.31 ms/step, exact=True, cert_stop=582 blocks, chkpts=0, bound_wins: sph=38%/int=25%/ellip=37%

### compressed_page_rfc

- `spherical_only`: 156.71 ms/step, exact=True, cert_stop=390 blocks, chkpts=0, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 308.57 ms/step, exact=True, cert_stop=390 blocks, chkpts=0, bound_wins: sph=79%/int=21%/ellip=0%
- `interval_ellip`: 312.86 ms/step, exact=True, cert_stop=390 blocks, chkpts=0, bound_wins: sph=40%/int=19%/ellip=41%

### hip_call_flow

- `spherical_only`: 293.80 ms/step, exact=True, cert_stop=390 blocks, chkpts=0, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 296.18 ms/step, exact=True, cert_stop=390 blocks, chkpts=0, bound_wins: sph=80%/int=20%/ellip=0%
- `interval_ellip`: 332.82 ms/step, exact=True, cert_stop=390 blocks, chkpts=0, bound_wins: sph=45%/int=18%/ellip=37%

### local_layer_profiles

- `spherical_only`: 293.49 ms/step, exact=True, cert_stop=390 blocks, chkpts=0, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 301.28 ms/step, exact=True, cert_stop=390 blocks, chkpts=0, bound_wins: sph=73%/int=27%/ellip=0%
- `interval_ellip`: 310.50 ms/step, exact=True, cert_stop=390 blocks, chkpts=0, bound_wins: sph=41%/int=24%/ellip=35%

### performance_journal

- `spherical_only`: 788.23 ms/step, exact=True, cert_stop=3078 blocks, chkpts=0, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 1643.26 ms/step, exact=True, cert_stop=3078 blocks, chkpts=0, bound_wins: sph=75%/int=25%/ellip=0%
- `interval_ellip`: 1610.05 ms/step, exact=True, cert_stop=3078 blocks, chkpts=0, bound_wins: sph=40%/int=23%/ellip=38%

### test_attention_vs_dense

- `spherical_only`: 152.01 ms/step, exact=True, cert_stop=108 blocks, chkpts=0, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 154.53 ms/step, exact=True, cert_stop=108 blocks, chkpts=0, bound_wins: sph=74%/int=26%/ellip=0%
- `interval_ellip`: 188.09 ms/step, exact=True, cert_stop=108 blocks, chkpts=0, bound_wins: sph=40%/int=24%/ellip=37%

### turboquant_comparison_plan

- `spherical_only`: 299.12 ms/step, exact=False, cert_stop=390 blocks, chkpts=0, bound_wins: sph=100%/int=0%/ellip=0%
- `interval`: 298.94 ms/step, exact=False, cert_stop=390 blocks, chkpts=0, bound_wins: sph=73%/int=27%/ellip=0%
- `interval_ellip`: 346.18 ms/step, exact=False, cert_stop=390 blocks, chkpts=0, bound_wins: sph=38%/int=23%/ellip=39%

