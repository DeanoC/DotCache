# Certificate Calibration Results

Model: Qwen/Qwen3.5-0.8B, Device: cuda, Steps: 8

| Lane | Case | cert_stop% | beta_med | beta_p90 | beta_max | delta_med | delta_p90 | delta_max | 1st_cert_blks |
|---|---|---|---|---|---|---|---|---|---|
| spherical_only | aae_stage_summary | 100.0% | 0.0000 | 0.0000 | 0.0000 | 0.000000 | 0.000000 | 0.000000 | 97.0 |
| spherical_only | bench_decode_code | 97.9% | 0.0000 | 0.0000 | 0.0000 | 0.000000 | 0.000000 | 0.000000 | 40.0 |
| spherical_only | benchmark_report | 100.0% | 0.0000 | 0.0000 | 0.0000 | 0.000000 | 0.000000 | 0.000000 | 97.0 |
| spherical_only | compressed_page_rfc | 100.0% | 0.0000 | 0.0000 | 0.0000 | 0.000000 | 0.000000 | 0.000000 | 65.0 |
| spherical_only | hip_call_flow | 100.0% | 0.0000 | 0.0000 | 0.0000 | 0.000000 | 0.000000 | 0.000000 | 65.0 |
| spherical_only | local_layer_profiles | 100.0% | 0.0000 | 0.0000 | 0.0000 | 0.000000 | 0.000000 | 0.000000 | 65.0 |
| spherical_only | test_attention_vs_dense | 95.8% | 0.0000 | 0.0000 | 0.0000 | 0.000000 | 0.000000 | 0.000000 | 18.0 |
| spherical_only | turboquant_comparison_plan | 100.0% | 0.0000 | 0.0000 | 0.0000 | 0.000000 | 0.000000 | 0.000000 | 65.0 |
| interval | aae_stage_summary | 100.0% | 0.0000 | 0.0000 | 0.0000 | 0.000000 | 0.000000 | 0.000000 | 97.0 |
| interval | bench_decode_code | 97.9% | 0.0000 | 0.0000 | 0.0000 | 0.000000 | 0.000000 | 0.000000 | 40.0 |
| interval | benchmark_report | 100.0% | 0.0000 | 0.0000 | 0.0000 | 0.000000 | 0.000000 | 0.000000 | 97.0 |
| interval | compressed_page_rfc | 100.0% | 0.0000 | 0.0000 | 0.0000 | 0.000000 | 0.000000 | 0.000000 | 65.0 |
| interval | hip_call_flow | 100.0% | 0.0000 | 0.0000 | 0.0000 | 0.000000 | 0.000000 | 0.000000 | 65.0 |
| interval | local_layer_profiles | 100.0% | 0.0000 | 0.0000 | 0.0000 | 0.000000 | 0.000000 | 0.000000 | 65.0 |
| interval | test_attention_vs_dense | 95.8% | 0.0000 | 0.0000 | 0.0000 | 0.000000 | 0.000000 | 0.000000 | 18.0 |
| interval | turboquant_comparison_plan | 100.0% | 0.0000 | 0.0000 | 0.0000 | 0.000000 | 0.000000 | 0.000000 | 65.0 |
| interval_ellip | aae_stage_summary | 100.0% | 0.0000 | 0.0000 | 0.0000 | 0.000000 | 0.000000 | 0.000000 | 97.0 |
| interval_ellip | bench_decode_code | 100.0% | 0.0000 | 0.0000 | 0.0000 | 0.000000 | 0.000000 | 0.000000 | 40.0 |
| interval_ellip | benchmark_report | 100.0% | 0.0000 | 0.0000 | 0.0000 | 0.000000 | 0.000000 | 0.000000 | 97.0 |
| interval_ellip | compressed_page_rfc | 100.0% | 0.0000 | 0.0000 | 0.0000 | 0.000000 | 0.000000 | 0.000000 | 65.0 |
| interval_ellip | hip_call_flow | 100.0% | 0.0000 | 0.0000 | 0.0000 | 0.000000 | 0.000000 | 0.000000 | 65.0 |
| interval_ellip | local_layer_profiles | 100.0% | 0.0000 | 0.0000 | 0.0000 | 0.000000 | 0.000000 | 0.000000 | 65.0 |
| interval_ellip | test_attention_vs_dense | 95.8% | 0.0000 | 0.0000 | 0.0000 | 0.000000 | 0.000000 | 0.000000 | 18.0 |
| interval_ellip | turboquant_comparison_plan | 100.0% | 0.0000 | 0.0000 | 0.0000 | 0.000000 | 0.000000 | 0.000000 | 65.0 |
