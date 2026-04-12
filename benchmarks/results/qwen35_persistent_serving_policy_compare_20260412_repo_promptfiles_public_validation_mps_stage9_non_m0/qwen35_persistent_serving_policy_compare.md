# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 6
- dense avg ms/step: 70.8964
- hand-tuned avg ms/step: 2330.8272
- bias avg ms/step: 2273.0113
- hand-tuned vs dense exact match rate: 0.833
- bias vs dense exact match rate: 0.833
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.833
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 9.1375
- hand-tuned score ms/case: 66.3018
- bias score ms/case: 50.3631
- hand-tuned selection ms/case: 812.9233
- bias selection ms/case: 839.0900
- hand-tuned optional-selection ms/case: 25.3981
- bias optional-selection ms/case: 25.9921
- hand-tuned diverse-selection ms/case: 4.5461
- bias diverse-selection ms/case: 4.5964
- hand-tuned compression-selection ms/case: 0.0000
- bias compression-selection ms/case: 0.0000
- hand-tuned policy-bias ms/case: 0.0835
- bias policy-bias ms/case: 0.1103
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
- hand-tuned executed M3 blocks/case: 4912.00
- bias executed M3 blocks/case: 4912.00
- hand-tuned exact-M3 score ms/case: 0.0000
- bias exact-M3 score ms/case: 0.0000
- hand-tuned final-mix ms/case: 0.0000
- bias final-mix ms/case: 0.0000

## Cases

- cuda_shortlist_paper_table: dense 86.7324 ms/step, hand 2936.1636, bias 2842.8791, hand=dense True, bias=dense True, bias=hand True, hand select 1032.93 ms, bias select 1065.30 ms
- model_roadmap: dense 69.6393 ms/step, hand 2221.4667, bias 2203.6435, hand=dense True, bias=dense True, bias=hand True, hand select 782.82 ms, bias select 817.35 ms
- real_mixed_probe_code: dense 61.3475 ms/step, hand 2175.4331, bias 2182.2395, hand=dense True, bias=dense True, bias=hand True, hand select 743.99 ms, bias select 803.14 ms
- serving_policy_compare_code: dense 69.0008 ms/step, hand 2953.5372, bias 2824.5156, hand=dense True, bias=dense True, bias=hand True, hand select 1068.28 ms, bias select 1057.38 ms
- stage9_backend_comparison: dense 74.9721 ms/step, hand 1588.7649, bias 1500.8359, hand=dense True, bias=dense True, bias=hand True, hand select 520.98 ms, bias select 531.37 ms
- submission_execution_plan: dense 63.6866 ms/step, hand 2109.5977, bias 2083.9542, hand=dense False, bias=dense False, bias=hand True, hand select 728.54 ms, bias select 759.99 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
