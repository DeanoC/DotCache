# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 6
- dense avg ms/step: 26.0325
- hand-tuned avg ms/step: 632.5052
- bias avg ms/step: 632.1238
- hand-tuned vs dense exact match rate: 0.167
- bias vs dense exact match rate: 0.167
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.333
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 12.4990
- hand-tuned score ms/case: 99.9685
- bias score ms/case: 95.1194
- hand-tuned selection ms/case: 202.5111
- bias selection ms/case: 197.6511
- hand-tuned optional-selection ms/case: 18.8229
- bias optional-selection ms/case: 18.7893
- hand-tuned diverse-selection ms/case: 7.0711
- bias diverse-selection ms/case: 7.0628
- hand-tuned compression-selection ms/case: 0.0000
- bias compression-selection ms/case: 0.0000
- hand-tuned policy-bias ms/case: 0.0773
- bias policy-bias ms/case: 0.1107
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

- local_layer_profiles: dense 35.2330 ms/step, hand 326.7908, bias 318.7448, hand=dense False, bias=dense False, bias=hand True, hand select 213.98 ms, bias select 182.44 ms
- model_roadmap: dense 24.6456 ms/step, hand 316.5852, bias 318.5226, hand=dense True, bias=dense True, bias=hand True, hand select 181.76 ms, bias select 182.36 ms
- page_selection_eval: dense 24.2778 ms/step, hand 913.1120, bias 912.0698, hand=dense False, bias=dense False, bias=hand True, hand select 220.96 ms, bias select 220.82 ms
- performance_journal: dense 24.0737 ms/step, hand 1669.5210, bias 1671.3408, hand=dense False, bias=dense False, bias=hand True, hand select 258.09 ms, bias select 259.13 ms
- repo_readme: dense 23.9515 ms/step, hand 316.2257, bias 317.5073, hand=dense False, bias=dense False, bias=hand True, hand select 180.78 ms, bias select 181.27 ms
- submission_execution_plan: dense 24.0132 ms/step, hand 252.7964, bias 254.5576, hand=dense False, bias=dense False, bias=hand True, hand select 159.49 ms, bias select 159.89 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
