# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 6
- dense avg ms/step: 25.4554
- hand-tuned avg ms/step: 464.8833
- bias avg ms/step: 465.3962
- hand-tuned vs dense exact match rate: 0.167
- bias vs dense exact match rate: 0.167
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.167
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 19.0069
- hand-tuned score ms/case: 115.5392
- bias score ms/case: 113.3100
- hand-tuned selection ms/case: 137.6000
- bias selection ms/case: 130.8526
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

- local_layer_profiles: dense 34.0990 ms/step, hand 407.0556, bias 400.2540, hand=dense False, bias=dense False, bias=hand True, hand select 175.30 ms, bias select 131.55 ms
- model_roadmap: dense 23.5179 ms/step, hand 399.8339, bias 401.2912, hand=dense True, bias=dense True, bias=hand True, hand select 129.35 ms, bias select 130.05 ms
- page_selection_eval: dense 24.1266 ms/step, hand 558.3685, bias 559.7330, hand=dense False, bias=dense False, bias=hand True, hand select 129.98 ms, bias select 130.54 ms
- performance_journal: dense 23.7505 ms/step, hand 716.5840, bias 718.5920, hand=dense False, bias=dense False, bias=hand True, hand select 131.09 ms, bias select 131.64 ms
- repo_readme: dense 23.5490 ms/step, hand 394.1367, bias 396.6360, hand=dense False, bias=dense False, bias=hand True, hand select 130.02 ms, bias select 130.77 ms
- submission_execution_plan: dense 23.6895 ms/step, hand 313.3210, bias 315.8707, hand=dense False, bias=dense False, bias=hand True, hand select 129.86 ms, bias select 130.57 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
