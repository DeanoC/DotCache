# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 6
- dense avg ms/step: 101.8691
- hand-tuned avg ms/step: 2484.6003
- bias avg ms/step: 2828.7758
- hand-tuned vs dense exact match rate: 0.167
- bias vs dense exact match rate: 0.167
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.667
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 13.9351
- hand-tuned score ms/case: 137.7746
- bias score ms/case: 103.1989
- hand-tuned selection ms/case: 1320.1957
- bias selection ms/case: 1481.8547
- hand-tuned optional-selection ms/case: 29.6396
- bias optional-selection ms/case: 34.6236
- hand-tuned diverse-selection ms/case: 5.7263
- bias diverse-selection ms/case: 6.8023
- hand-tuned compression-selection ms/case: 0.0000
- bias compression-selection ms/case: 0.0000
- hand-tuned policy-bias ms/case: 0.1151
- bias policy-bias ms/case: 0.1717
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

- local_layer_profiles: dense 67.4952 ms/step, hand 1768.7865, bias 3854.8160, hand=dense False, bias=dense False, bias=hand True, hand select 863.47 ms, bias select 2101.40 ms
- model_roadmap: dense 64.9107 ms/step, hand 1903.2389, bias 1725.8048, hand=dense True, bias=dense True, bias=hand True, hand select 1023.76 ms, bias select 885.16 ms
- page_selection_eval: dense 78.6405 ms/step, hand 2618.6019, bias 2556.4417, hand=dense False, bias=dense False, bias=hand True, hand select 1380.96 ms, bias select 1324.60 ms
- performance_journal: dense 88.8225 ms/step, hand 3460.9153, bias 3377.8148, hand=dense False, bias=dense False, bias=hand True, hand select 1885.60 ms, bias select 1799.21 ms
- repo_readme: dense 95.0958 ms/step, hand 2313.6701, bias 2917.7089, hand=dense False, bias=dense False, bias=hand True, hand select 1177.26 ms, bias select 1501.75 ms
- submission_execution_plan: dense 216.2497 ms/step, hand 2842.3892, bias 2540.0687, hand=dense False, bias=dense False, bias=hand True, hand select 1590.13 ms, bias select 1279.00 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
