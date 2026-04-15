# Certificate Calibration: ℓ₂ Error vs Bound

Model: Qwen/Qwen3.5-0.8B, Steps: 8

ℓ₂ error: median=1.9980, max=3.1323

Cosine similarity: median=0.999999, min=0.999996

Token match rate: 100.0%

| Case | ℓ₂ median | ℓ₂ max | cos min | token match |
|---|---|---|---|---|
| aae_stage_summary | 1.9796 | 2.8176 | 0.999998 | 100% |
| bench_decode_code | 2.7361 | 3.1323 | 0.999996 | 100% |
| benchmark_report | 1.9263 | 2.7058 | 0.999998 | 100% |
| compressed_page_rfc | 1.7486 | 2.3349 | 0.999998 | 100% |
| hip_call_flow | 2.2378 | 2.8779 | 0.999998 | 100% |
| local_layer_profiles | 1.8773 | 2.4960 | 0.999998 | 100% |
| test_attention_vs_dense | 2.0559 | 2.4149 | 0.999999 | 100% |
| turboquant_comparison_plan | 1.9476 | 2.0890 | 0.999999 | 100% |
