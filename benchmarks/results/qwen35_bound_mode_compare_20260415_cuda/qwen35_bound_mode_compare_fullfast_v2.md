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
| `spherical_only` | 153.40 | 0.875 | 106.39 | 384.0 | 0.0 |
| `interval` | 156.60 | 0.875 | 128.97 | 384.0 | 0.0 |
| `interval_ellip` | 163.67 | 0.875 | 154.43 | 384.0 | 0.0 |

### Speed-up vs spherical_only

- `interval`: -2.1% (156.60 vs 153.40 ms/step), cert_stop_blocks 384.0 vs 384.0 (0.0 fewer)
- `interval_ellip`: -6.7% (163.67 vs 153.40 ms/step), cert_stop_blocks 384.0 vs 384.0 (0.0 fewer)

## Per-case results

### aae_stage_summary

- `spherical_only`: 204.18 ms/step, exact=True, cert_stop=582 blocks, chkpts=0
- `interval`: 203.62 ms/step, exact=True, cert_stop=582 blocks, chkpts=0
- `interval_ellip`: 209.68 ms/step, exact=True, cert_stop=582 blocks, chkpts=0

### bench_decode_code

- `spherical_only`: 110.44 ms/step, exact=True, cert_stop=240 blocks, chkpts=0
- `interval`: 113.51 ms/step, exact=True, cert_stop=240 blocks, chkpts=0
- `interval_ellip`: 121.18 ms/step, exact=True, cert_stop=240 blocks, chkpts=0

### benchmark_report

- `spherical_only`: 201.22 ms/step, exact=True, cert_stop=582 blocks, chkpts=0
- `interval`: 204.99 ms/step, exact=True, cert_stop=582 blocks, chkpts=0
- `interval_ellip`: 212.54 ms/step, exact=True, cert_stop=582 blocks, chkpts=0

### compressed_page_rfc

- `spherical_only`: 156.11 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval`: 158.94 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval_ellip`: 166.37 ms/step, exact=True, cert_stop=390 blocks, chkpts=0

### hip_call_flow

- `spherical_only`: 155.80 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval`: 161.03 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval_ellip`: 167.50 ms/step, exact=True, cert_stop=390 blocks, chkpts=0

### local_layer_profiles

- `spherical_only`: 155.36 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval`: 159.66 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval_ellip`: 165.82 ms/step, exact=True, cert_stop=390 blocks, chkpts=0

### test_attention_vs_dense

- `spherical_only`: 87.17 ms/step, exact=True, cert_stop=108 blocks, chkpts=0
- `interval`: 92.25 ms/step, exact=True, cert_stop=108 blocks, chkpts=0
- `interval_ellip`: 100.04 ms/step, exact=True, cert_stop=108 blocks, chkpts=0

### turboquant_comparison_plan

- `spherical_only`: 156.92 ms/step, exact=False, cert_stop=390 blocks, chkpts=0
- `interval`: 158.84 ms/step, exact=False, cert_stop=390 blocks, chkpts=0
- `interval_ellip`: 166.24 ms/step, exact=False, cert_stop=390 blocks, chkpts=0

