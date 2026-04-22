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
| `spherical_only` | 265.88 | 0.875 | 136.27 | 384.0 | 0.0 |
| `interval` | 266.56 | 0.875 | 162.86 | 384.0 | 0.0 |
| `interval_ellip` | 294.84 | 0.875 | 190.47 | 384.0 | 0.0 |

### Speed-up vs spherical_only

- `interval`: -0.3% (266.56 vs 265.88 ms/step), cert_stop_blocks 384.0 vs 384.0 (0.0 fewer)
- `interval_ellip`: -10.9% (294.84 vs 265.88 ms/step), cert_stop_blocks 384.0 vs 384.0 (0.0 fewer)

## Per-case results

### aae_stage_summary

- `spherical_only`: 258.67 ms/step, exact=True, cert_stop=582 blocks, chkpts=0
- `interval`: 254.91 ms/step, exact=True, cert_stop=582 blocks, chkpts=0
- `interval_ellip`: 406.92 ms/step, exact=True, cert_stop=582 blocks, chkpts=0

### bench_decode_code

- `spherical_only`: 205.30 ms/step, exact=True, cert_stop=240 blocks, chkpts=0
- `interval`: 211.39 ms/step, exact=True, cert_stop=240 blocks, chkpts=0
- `interval_ellip`: 219.17 ms/step, exact=True, cert_stop=240 blocks, chkpts=0

### benchmark_report

- `spherical_only`: 384.02 ms/step, exact=True, cert_stop=582 blocks, chkpts=0
- `interval`: 388.60 ms/step, exact=True, cert_stop=582 blocks, chkpts=0
- `interval_ellip`: 417.72 ms/step, exact=True, cert_stop=582 blocks, chkpts=0

### compressed_page_rfc

- `spherical_only`: 290.12 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval`: 266.87 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval_ellip`: 323.67 ms/step, exact=True, cert_stop=390 blocks, chkpts=0

### hip_call_flow

- `spherical_only`: 304.54 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval`: 233.56 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval_ellip`: 333.30 ms/step, exact=True, cert_stop=390 blocks, chkpts=0

### local_layer_profiles

- `spherical_only`: 292.59 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval`: 294.19 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval_ellip`: 173.73 ms/step, exact=True, cert_stop=390 blocks, chkpts=0

### test_attention_vs_dense

- `spherical_only`: 91.19 ms/step, exact=True, cert_stop=108 blocks, chkpts=0
- `interval`: 158.41 ms/step, exact=True, cert_stop=108 blocks, chkpts=0
- `interval_ellip`: 180.94 ms/step, exact=True, cert_stop=108 blocks, chkpts=0

### turboquant_comparison_plan

- `spherical_only`: 300.60 ms/step, exact=False, cert_stop=390 blocks, chkpts=0
- `interval`: 324.54 ms/step, exact=False, cert_stop=390 blocks, chkpts=0
- `interval_ellip`: 303.28 ms/step, exact=False, cert_stop=390 blocks, chkpts=0

