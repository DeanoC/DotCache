# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 8
- dense avg ms/step: 61.1958
- hand-tuned avg ms/step: 945.1134
- bias avg ms/step: 913.9570
- hand-tuned vs dense exact match rate: 1.000
- bias vs dense exact match rate: 1.000
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 1.000
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 7.2380
- hand-tuned score ms/case: 73.1319
- bias score ms/case: 56.7974
- hand-tuned selection ms/case: 473.3758
- bias selection ms/case: 482.9444
- hand-tuned optional-selection ms/case: 18.7025
- bias optional-selection ms/case: 19.0457
- hand-tuned diverse-selection ms/case: 3.1274
- bias diverse-selection ms/case: 3.0731
- hand-tuned compression-selection ms/case: 0.0000
- bias compression-selection ms/case: 0.0000
- hand-tuned policy-bias ms/case: 0.0577
- bias policy-bias ms/case: 0.0722
- hand-tuned direct-M0 assembly ms/case: 0.0000
- bias direct-M0 assembly ms/case: 0.0000
- hand-tuned direct-M0 query-prep ms/case: 0.0000
- bias direct-M0 query-prep ms/case: 0.0000
- hand-tuned direct-M0 gather ms/case: 0.0000
- bias direct-M0 gather ms/case: 0.0000
- hand-tuned direct-M0 score ms/case: 0.0000
- bias direct-M0 score ms/case: 0.0000
- hand-tuned executed M0 blocks/case: 0.00
- bias executed M0 blocks/case: 0.00
- hand-tuned executed M3 blocks/case: 0.00
- bias executed M3 blocks/case: 0.00
- hand-tuned exact-M3 score ms/case: 0.0000
- bias exact-M3 score ms/case: 0.0000
- hand-tuned final-mix ms/case: 0.0000
- bias final-mix ms/case: 0.0000

## Cases

- aae_stage_summary: dense 64.2245 ms/step, hand 1382.1729, bias 1340.5946, hand=dense True, bias=dense True, bias=hand True, hand select 700.53 ms, bias select 753.57 ms
- bench_decode_code: dense 63.5960 ms/step, hand 608.9026, bias 595.1633, hand=dense True, bias=dense True, bias=hand True, hand select 307.90 ms, bias select 294.39 ms
- benchmark_report: dense 65.3929 ms/step, hand 1350.5411, bias 1347.3284, hand=dense True, bias=dense True, bias=hand True, hand select 732.00 ms, bias select 756.61 ms
- compressed_page_rfc: dense 62.4926 ms/step, hand 945.4824, bias 927.5660, hand=dense True, bias=dense True, bias=hand True, hand select 492.51 ms, bias select 495.62 ms
- hip_call_flow: dense 57.0536 ms/step, hand 950.7977, bias 930.5364, hand=dense True, bias=dense True, bias=hand True, hand select 474.85 ms, bias select 498.00 ms
- local_layer_profiles: dense 56.6914 ms/step, hand 971.6309, bias 923.7835, hand=dense True, bias=dense True, bias=hand True, hand select 466.29 ms, bias select 485.80 ms
- test_attention_vs_dense: dense 60.9421 ms/step, hand 382.9532, bias 315.8488, hand=dense True, bias=dense True, bias=hand True, hand select 134.96 ms, bias select 80.25 ms
- turboquant_comparison_plan: dense 59.1731 ms/step, hand 968.4264, bias 930.8352, hand=dense True, bias=dense True, bias=hand True, hand select 477.96 ms, bias select 499.31 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
