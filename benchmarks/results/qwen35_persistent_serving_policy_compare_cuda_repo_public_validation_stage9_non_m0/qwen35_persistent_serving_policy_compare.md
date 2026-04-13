# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 6
- dense avg ms/step: 26.3672
- hand-tuned avg ms/step: 338.9078
- bias avg ms/step: 339.6559
- hand-tuned vs dense exact match rate: 0.833
- bias vs dense exact match rate: 0.833
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.167
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 17.2708
- hand-tuned score ms/case: 117.3877
- bias score ms/case: 115.3505
- hand-tuned selection ms/case: 140.0318
- bias selection ms/case: 132.3343
- hand-tuned optional-selection ms/case: 0.0000
- bias optional-selection ms/case: 0.0000
- hand-tuned diverse-selection ms/case: 0.0000
- bias diverse-selection ms/case: 0.0000
- hand-tuned compression-selection ms/case: 0.0000
- bias compression-selection ms/case: 0.0000
- hand-tuned policy-bias ms/case: 0.0000
- bias policy-bias ms/case: 0.0000
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

- cuda_shortlist_paper_table: dense 36.7888 ms/step, hand 419.2420, bias 411.0873, hand=dense True, bias=dense True, bias=hand True, hand select 182.57 ms, bias select 132.18 ms
- model_roadmap: dense 24.1465 ms/step, hand 324.0345, bias 325.6528, hand=dense True, bias=dense True, bias=hand True, hand select 132.28 ms, bias select 132.74 ms
- real_mixed_probe_code: dense 24.1351 ms/step, hand 323.3318, bias 324.6108, hand=dense True, bias=dense True, bias=hand True, hand select 131.00 ms, bias select 132.62 ms
- serving_policy_compare_code: dense 24.2984 ms/step, hand 406.6033, bias 410.5805, hand=dense True, bias=dense True, bias=hand True, hand select 131.48 ms, bias select 132.50 ms
- stage9_backend_comparison: dense 24.2782 ms/step, hand 238.0929, bias 240.9378, hand=dense True, bias=dense True, bias=hand True, hand select 131.51 ms, bias select 131.75 ms
- submission_execution_plan: dense 24.5565 ms/step, hand 322.1424, bias 325.0660, hand=dense False, bias=dense False, bias=hand True, hand select 131.35 ms, bias select 132.21 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
