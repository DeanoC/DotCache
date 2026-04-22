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
| `spherical_only` | 306.65 | 0.875 | 137.61 | 384.0 | 0.0 |
| `interval` | 298.34 | 0.875 | 159.69 | 384.0 | 0.0 |
| `interval_ellip` | 336.35 | 0.875 | 190.85 | 384.0 | 0.0 |

### Speed-up vs spherical_only

- `interval`: +2.7% (298.34 vs 306.65 ms/step), cert_stop_blocks 384.0 vs 384.0 (0.0 fewer)
- `interval_ellip`: -9.7% (336.35 vs 306.65 ms/step), cert_stop_blocks 384.0 vs 384.0 (0.0 fewer)

## Per-case results

### aae_stage_summary

- `spherical_only`: 271.68 ms/step, exact=True, cert_stop=582 blocks, chkpts=0
- `interval`: 445.00 ms/step, exact=True, cert_stop=582 blocks, chkpts=0
- `interval_ellip`: 459.51 ms/step, exact=True, cert_stop=582 blocks, chkpts=0

### bench_decode_code

- `spherical_only`: 237.29 ms/step, exact=True, cert_stop=240 blocks, chkpts=0
- `interval`: 197.70 ms/step, exact=True, cert_stop=240 blocks, chkpts=0
- `interval_ellip`: 224.23 ms/step, exact=True, cert_stop=240 blocks, chkpts=0

### benchmark_report

- `spherical_only`: 439.60 ms/step, exact=True, cert_stop=582 blocks, chkpts=0
- `interval`: 305.18 ms/step, exact=True, cert_stop=582 blocks, chkpts=0
- `interval_ellip`: 460.26 ms/step, exact=True, cert_stop=582 blocks, chkpts=0

### compressed_page_rfc

- `spherical_only`: 358.11 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval`: 338.09 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval_ellip`: 301.23 ms/step, exact=True, cert_stop=390 blocks, chkpts=0

### hip_call_flow

- `spherical_only`: 286.74 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval`: 276.07 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval_ellip`: 358.57 ms/step, exact=True, cert_stop=390 blocks, chkpts=0

### local_layer_profiles

- `spherical_only`: 342.42 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval`: 284.21 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval_ellip`: 355.49 ms/step, exact=True, cert_stop=390 blocks, chkpts=0

### test_attention_vs_dense

- `spherical_only`: 179.47 ms/step, exact=True, cert_stop=108 blocks, chkpts=0
- `interval`: 161.27 ms/step, exact=True, cert_stop=108 blocks, chkpts=0
- `interval_ellip`: 175.49 ms/step, exact=True, cert_stop=108 blocks, chkpts=0

### turboquant_comparison_plan

- `spherical_only`: 337.86 ms/step, exact=False, cert_stop=390 blocks, chkpts=0
- `interval`: 379.24 ms/step, exact=False, cert_stop=390 blocks, chkpts=0
- `interval_ellip`: 356.06 ms/step, exact=False, cert_stop=390 blocks, chkpts=0

