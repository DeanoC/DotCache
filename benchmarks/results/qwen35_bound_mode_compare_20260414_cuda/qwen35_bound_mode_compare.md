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
| `spherical_only` | 261.52 | 0.750 | 159.97 | 384.0 | 0.0 |
| `interval` | 253.91 | 0.750 | 211.05 | 384.0 | 0.0 |
| `interval_ellip` | 272.42 | 0.875 | 299.43 | 384.0 | 0.0 |

### Speed-up vs spherical_only

- `interval`: +2.9% (253.91 vs 261.52 ms/step), cert_stop_blocks 384.0 vs 384.0 (0.0 fewer)
- `interval_ellip`: -4.2% (272.42 vs 261.52 ms/step), cert_stop_blocks 384.0 vs 384.0 (0.0 fewer)

## Per-case results

### aae_stage_summary

- `spherical_only`: 430.52 ms/step, exact=True, cert_stop=582 blocks, chkpts=0
- `interval`: 331.04 ms/step, exact=True, cert_stop=582 blocks, chkpts=0
- `interval_ellip`: 350.19 ms/step, exact=True, cert_stop=582 blocks, chkpts=0

### bench_decode_code

- `spherical_only`: 180.21 ms/step, exact=False, cert_stop=240 blocks, chkpts=0
- `interval`: 188.23 ms/step, exact=False, cert_stop=240 blocks, chkpts=0
- `interval_ellip`: 206.69 ms/step, exact=True, cert_stop=240 blocks, chkpts=0

### benchmark_report

- `spherical_only`: 326.83 ms/step, exact=True, cert_stop=582 blocks, chkpts=0
- `interval`: 329.40 ms/step, exact=True, cert_stop=582 blocks, chkpts=0
- `interval_ellip`: 348.96 ms/step, exact=True, cert_stop=582 blocks, chkpts=0

### compressed_page_rfc

- `spherical_only`: 253.45 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval`: 257.83 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval_ellip`: 276.67 ms/step, exact=True, cert_stop=390 blocks, chkpts=0

### hip_call_flow

- `spherical_only`: 254.26 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval`: 258.97 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval_ellip`: 276.58 ms/step, exact=True, cert_stop=390 blocks, chkpts=0

### local_layer_profiles

- `spherical_only`: 252.01 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval`: 259.06 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval_ellip`: 275.89 ms/step, exact=True, cert_stop=390 blocks, chkpts=0

### test_attention_vs_dense

- `spherical_only`: 140.68 ms/step, exact=True, cert_stop=108 blocks, chkpts=0
- `interval`: 148.42 ms/step, exact=True, cert_stop=108 blocks, chkpts=0
- `interval_ellip`: 166.46 ms/step, exact=True, cert_stop=108 blocks, chkpts=0

### turboquant_comparison_plan

- `spherical_only`: 254.22 ms/step, exact=False, cert_stop=390 blocks, chkpts=0
- `interval`: 258.32 ms/step, exact=False, cert_stop=390 blocks, chkpts=0
- `interval_ellip`: 277.92 ms/step, exact=False, cert_stop=390 blocks, chkpts=0

