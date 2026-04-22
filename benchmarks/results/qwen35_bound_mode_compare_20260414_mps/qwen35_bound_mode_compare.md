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
| `spherical_only` | 538.36 | 0.875 | 203.69 | 384.0 | 29.2 |
| `interval` | 507.82 | 0.875 | 254.80 | 384.0 | 29.2 |
| `interval_ellip` | 830.22 | 0.875 | 419.87 | 384.0 | 29.2 |

### Speed-up vs spherical_only

- `interval`: +5.7% (507.82 vs 538.36 ms/step), cert_stop_blocks 384.0 vs 384.0 (0.0 fewer)
- `interval_ellip`: -54.2% (830.22 vs 538.36 ms/step), cert_stop_blocks 384.0 vs 384.0 (0.0 fewer)

## Per-case results

### aae_stage_summary

- `spherical_only`: 727.66 ms/step, exact=True, cert_stop=582 blocks, chkpts=42
- `interval`: 660.09 ms/step, exact=True, cert_stop=582 blocks, chkpts=42
- `interval_ellip`: 887.54 ms/step, exact=True, cert_stop=582 blocks, chkpts=42

### bench_decode_code

- `spherical_only`: 393.33 ms/step, exact=True, cert_stop=240 blocks, chkpts=18
- `interval`: 322.85 ms/step, exact=True, cert_stop=240 blocks, chkpts=18
- `interval_ellip`: 644.20 ms/step, exact=True, cert_stop=240 blocks, chkpts=18

### benchmark_report

- `spherical_only`: 638.79 ms/step, exact=True, cert_stop=582 blocks, chkpts=42
- `interval`: 696.24 ms/step, exact=True, cert_stop=582 blocks, chkpts=42
- `interval_ellip`: 961.53 ms/step, exact=True, cert_stop=582 blocks, chkpts=42

### compressed_page_rfc

- `spherical_only`: 557.90 ms/step, exact=True, cert_stop=390 blocks, chkpts=30
- `interval`: 575.59 ms/step, exact=True, cert_stop=390 blocks, chkpts=30
- `interval_ellip`: 827.09 ms/step, exact=True, cert_stop=390 blocks, chkpts=30

### hip_call_flow

- `spherical_only`: 530.71 ms/step, exact=True, cert_stop=390 blocks, chkpts=30
- `interval`: 521.19 ms/step, exact=True, cert_stop=390 blocks, chkpts=30
- `interval_ellip`: 835.18 ms/step, exact=True, cert_stop=390 blocks, chkpts=30

### local_layer_profiles

- `spherical_only`: 517.64 ms/step, exact=True, cert_stop=390 blocks, chkpts=30
- `interval`: 507.90 ms/step, exact=True, cert_stop=390 blocks, chkpts=30
- `interval_ellip`: 786.85 ms/step, exact=True, cert_stop=390 blocks, chkpts=30

### test_attention_vs_dense

- `spherical_only`: 389.68 ms/step, exact=True, cert_stop=108 blocks, chkpts=12
- `interval`: 257.69 ms/step, exact=True, cert_stop=108 blocks, chkpts=12
- `interval_ellip`: 771.05 ms/step, exact=True, cert_stop=108 blocks, chkpts=12

### turboquant_comparison_plan

- `spherical_only`: 551.16 ms/step, exact=False, cert_stop=390 blocks, chkpts=30
- `interval`: 521.02 ms/step, exact=False, cert_stop=390 blocks, chkpts=30
- `interval_ellip`: 928.33 ms/step, exact=False, cert_stop=390 blocks, chkpts=30

