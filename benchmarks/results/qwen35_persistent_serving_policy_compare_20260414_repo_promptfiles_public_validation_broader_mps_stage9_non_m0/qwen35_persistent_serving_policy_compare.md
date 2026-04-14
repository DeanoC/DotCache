# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 8
- dense avg ms/step: 60.1904
- hand-tuned avg ms/step: 1392.0537
- bias avg ms/step: 1352.1261
- hand-tuned vs dense exact match rate: 1.000
- bias vs dense exact match rate: 1.000
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.875
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 7.4404
- hand-tuned score ms/case: 57.3009
- bias score ms/case: 46.5814
- hand-tuned selection ms/case: 448.5864
- bias selection ms/case: 469.5860
- hand-tuned optional-selection ms/case: 18.5218
- bias optional-selection ms/case: 19.2466
- hand-tuned diverse-selection ms/case: 3.1209
- bias diverse-selection ms/case: 3.0761
- hand-tuned compression-selection ms/case: 0.0000
- bias compression-selection ms/case: 0.0000
- hand-tuned policy-bias ms/case: 0.0599
- bias policy-bias ms/case: 0.0783
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
- hand-tuned executed M3 blocks/case: 3072.00
- bias executed M3 blocks/case: 3072.00
- hand-tuned exact-M3 score ms/case: 0.0000
- bias exact-M3 score ms/case: 0.0000
- hand-tuned final-mix ms/case: 0.0000
- bias final-mix ms/case: 0.0000

## Cases

- aae_stage_summary: dense 62.9896 ms/step, hand 2040.1258, bias 1989.1206, hand=dense True, bias=dense True, bias=hand True, hand select 691.83 ms, bias select 732.71 ms
- bench_decode_code: dense 59.6506 ms/step, hand 916.8687, bias 888.3018, hand=dense True, bias=dense True, bias=hand True, hand select 289.28 ms, bias select 281.03 ms
- benchmark_report: dense 63.4349 ms/step, hand 2012.7570, bias 2000.7066, hand=dense True, bias=dense True, bias=hand True, hand select 708.13 ms, bias select 733.66 ms
- compressed_page_rfc: dense 61.5388 ms/step, hand 1443.5844, bias 1354.4993, hand=dense True, bias=dense True, bias=hand True, hand select 463.18 ms, bias select 479.97 ms
- hip_call_flow: dense 55.9421 ms/step, hand 1327.6097, bias 1346.3467, hand=dense True, bias=dense True, bias=hand True, hand select 425.24 ms, bias select 476.09 ms
- local_layer_profiles: dense 57.9130 ms/step, hand 1428.5540, bias 1390.9548, hand=dense True, bias=dense True, bias=hand True, hand select 449.68 ms, bias select 489.18 ms
- test_attention_vs_dense: dense 60.9025 ms/step, hand 548.5623, bias 455.9028, hand=dense True, bias=dense True, bias=hand True, hand select 103.33 ms, bias select 70.16 ms
- turboquant_comparison_plan: dense 59.1515 ms/step, hand 1418.3681, bias 1391.1765, hand=dense True, bias=dense True, bias=hand True, hand select 458.02 ms, bias select 493.89 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
