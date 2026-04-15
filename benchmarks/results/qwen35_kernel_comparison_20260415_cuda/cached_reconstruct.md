# Qwen3.5 Bound Mode Compare

Compares three certified upper-bound modes under the real mixed Stage 9 serving config.

## Lane definitions

| Lane | enable_interval_bound | enable_ellipsoidal_bound |
|---|---|---|
| `interval` | True | False |

## Summary

| Lane | avg ms/step | exact_match_vs_dense | score_ms/case | cert_stop_blocks/case | checkpoints/case |
|---|---|---|---|---|---|
| `interval` | 174.57 | 0.875 | 129.86 | 384.0 | 0.0 |

## Bound winner fractions

Which certified upper-bound method provides the tightest value per (block, q_head) evaluation.

| Lane | spherical | interval | ellipsoidal | total evals |
|---|---|---|---|---|
| `interval` | 76.6% | 23.4% | 0.0% | 196,608 |

## Per-case results

### aae_stage_summary

- `interval`: 228.48 ms/step, exact=True, cert_stop=582 blocks, chkpts=0, bound_wins: sph=79%/int=21%/ellip=0%

### bench_decode_code

- `interval`: 121.29 ms/step, exact=True, cert_stop=240 blocks, chkpts=0, bound_wins: sph=87%/int=13%/ellip=0%

### benchmark_report

- `interval`: 225.18 ms/step, exact=True, cert_stop=582 blocks, chkpts=0, bound_wins: sph=71%/int=29%/ellip=0%

### compressed_page_rfc

- `interval`: 179.63 ms/step, exact=True, cert_stop=390 blocks, chkpts=0, bound_wins: sph=79%/int=21%/ellip=0%

### hip_call_flow

- `interval`: 173.33 ms/step, exact=True, cert_stop=390 blocks, chkpts=0, bound_wins: sph=80%/int=20%/ellip=0%

### local_layer_profiles

- `interval`: 175.83 ms/step, exact=True, cert_stop=390 blocks, chkpts=0, bound_wins: sph=73%/int=27%/ellip=0%

### test_attention_vs_dense

- `interval`: 87.51 ms/step, exact=True, cert_stop=108 blocks, chkpts=0, bound_wins: sph=74%/int=26%/ellip=0%

### turboquant_comparison_plan

- `interval`: 205.33 ms/step, exact=False, cert_stop=390 blocks, chkpts=0, bound_wins: sph=73%/int=27%/ellip=0%

