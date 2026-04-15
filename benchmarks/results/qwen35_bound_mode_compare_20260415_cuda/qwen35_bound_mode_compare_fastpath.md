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
| `spherical_only` | 279.80 | 1.000 | 140.34 | 384.0 | 0.0 |
| `interval` | 288.81 | 1.000 | 166.39 | 384.0 | 0.0 |
| `interval_ellip` | 286.26 | 1.000 | 190.33 | 384.0 | 0.0 |

### Speed-up vs spherical_only

- `interval`: -3.2% (288.81 vs 279.80 ms/step), cert_stop_blocks 384.0 vs 384.0 (0.0 fewer)
- `interval_ellip`: -2.3% (286.26 vs 279.80 ms/step), cert_stop_blocks 384.0 vs 384.0 (0.0 fewer)

## Per-case results

### aae_stage_summary

- `spherical_only`: 413.33 ms/step, exact=True, cert_stop=582 blocks, chkpts=0
- `interval`: 354.85 ms/step, exact=True, cert_stop=582 blocks, chkpts=0
- `interval_ellip`: 421.55 ms/step, exact=True, cert_stop=582 blocks, chkpts=0

### bench_decode_code

- `spherical_only`: 206.69 ms/step, exact=True, cert_stop=240 blocks, chkpts=0
- `interval`: 211.99 ms/step, exact=True, cert_stop=240 blocks, chkpts=0
- `interval_ellip`: 224.33 ms/step, exact=True, cert_stop=240 blocks, chkpts=0

### benchmark_report

- `spherical_only`: 395.83 ms/step, exact=True, cert_stop=582 blocks, chkpts=0
- `interval`: 378.44 ms/step, exact=True, cert_stop=582 blocks, chkpts=0
- `interval_ellip`: 222.27 ms/step, exact=True, cert_stop=582 blocks, chkpts=0

### compressed_page_rfc

- `spherical_only`: 311.57 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval`: 294.04 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval_ellip`: 304.98 ms/step, exact=True, cert_stop=390 blocks, chkpts=0

### hip_call_flow

- `spherical_only`: 303.67 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval`: 310.69 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval_ellip`: 310.94 ms/step, exact=True, cert_stop=390 blocks, chkpts=0

### local_layer_profiles

- `spherical_only`: 162.45 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval`: 303.60 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval_ellip`: 323.40 ms/step, exact=True, cert_stop=390 blocks, chkpts=0

### test_attention_vs_dense

- `spherical_only`: 157.34 ms/step, exact=True, cert_stop=108 blocks, chkpts=0
- `interval`: 163.33 ms/step, exact=True, cert_stop=108 blocks, chkpts=0
- `interval_ellip`: 178.11 ms/step, exact=True, cert_stop=108 blocks, chkpts=0

### turboquant_comparison_plan

- `spherical_only`: 287.47 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval`: 293.55 ms/step, exact=True, cert_stop=390 blocks, chkpts=0
- `interval_ellip`: 304.50 ms/step, exact=True, cert_stop=390 blocks, chkpts=0

