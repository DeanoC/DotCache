# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 6
- dense avg ms/step: 81.9753
- hand-tuned avg ms/step: 4289.7923
- bias avg ms/step: 4234.4017
- hand-tuned vs dense exact match rate: 0.167
- bias vs dense exact match rate: 0.167
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.833
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 10.4287
- hand-tuned score ms/case: 88.4735
- bias score ms/case: 52.0360
- hand-tuned selection ms/case: 1627.4693
- bias selection ms/case: 1174.7160
- hand-tuned optional-selection ms/case: 26.8146
- bias optional-selection ms/case: 26.2934
- hand-tuned diverse-selection ms/case: 4.7629
- bias diverse-selection ms/case: 4.7778
- hand-tuned compression-selection ms/case: 0.0000
- bias compression-selection ms/case: 0.0000
- hand-tuned policy-bias ms/case: 0.1095
- bias policy-bias ms/case: 0.1324
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
- hand-tuned executed M3 blocks/case: 6224.00
- bias executed M3 blocks/case: 6224.00
- hand-tuned exact-M3 score ms/case: 0.0000
- bias exact-M3 score ms/case: 0.0000
- hand-tuned final-mix ms/case: 0.0000
- bias final-mix ms/case: 0.0000

## Cases

- local_layer_profiles: dense 81.2535 ms/step, hand 2936.8071, bias 2801.2703, hand=dense False, bias=dense False, bias=hand True, hand select 1023.52 ms, bias select 1012.96 ms
- model_roadmap: dense 64.1436 ms/step, hand 2846.0654, bias 2810.5057, hand=dense True, bias=dense True, bias=hand True, hand select 994.78 ms, bias select 1015.03 ms
- page_selection_eval: dense 93.2517 ms/step, hand 6159.6699, bias 5950.4083, hand=dense False, bias=dense False, bias=hand True, hand select 1564.89 ms, bias select 1489.06 ms
- performance_journal: dense 108.8726 ms/step, hand 9094.6308, bias 9299.1114, hand=dense False, bias=dense False, bias=hand True, hand select 4495.33 ms, bias select 1822.78 ms
- repo_readme: dense 67.6557 ms/step, hand 2535.3632, bias 2518.5532, hand=dense False, bias=dense False, bias=hand True, hand select 895.65 ms, bias select 955.20 ms
- submission_execution_plan: dense 76.6749 ms/step, hand 2166.2177, bias 2026.5612, hand=dense False, bias=dense False, bias=hand True, hand select 790.64 ms, bias select 753.27 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
