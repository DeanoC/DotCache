# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 6
- dense avg ms/step: 25.8235
- hand-tuned avg ms/step: 791.3715
- bias avg ms/step: 792.5957
- hand-tuned vs dense exact match rate: 0.167
- bias vs dense exact match rate: 0.167
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.167
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 17.5748
- hand-tuned score ms/case: 120.1184
- bias score ms/case: 114.9119
- hand-tuned selection ms/case: 222.0833
- bias selection ms/case: 216.9104
- hand-tuned optional-selection ms/case: 19.2884
- bias optional-selection ms/case: 19.3247
- hand-tuned diverse-selection ms/case: 7.2352
- bias diverse-selection ms/case: 7.2568
- hand-tuned compression-selection ms/case: 0.0000
- bias compression-selection ms/case: 0.0000
- hand-tuned policy-bias ms/case: 0.1023
- bias policy-bias ms/case: 0.1388
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

- local_layer_profiles: dense 35.5522 ms/step, hand 679.0479, bias 670.7464, hand=dense False, bias=dense False, bias=hand True, hand select 236.85 ms, bias select 203.20 ms
- model_roadmap: dense 23.6747 ms/step, hand 667.1302, bias 670.1198, hand=dense True, bias=dense True, bias=hand True, hand select 201.00 ms, bias select 201.41 ms
- page_selection_eval: dense 23.8183 ms/step, hand 958.7452, bias 962.9326, hand=dense False, bias=dense False, bias=hand True, hand select 239.26 ms, bias select 239.59 ms
- performance_journal: dense 23.8011 ms/step, hand 1257.1399, bias 1260.5316, hand=dense False, bias=dense False, bias=hand True, hand select 276.43 ms, bias select 276.65 ms
- repo_readme: dense 23.6972 ms/step, hand 666.5459, bias 669.8021, hand=dense False, bias=dense False, bias=hand True, hand select 200.41 ms, bias select 201.76 ms
- submission_execution_plan: dense 24.3973 ms/step, hand 519.6202, bias 521.4415, hand=dense False, bias=dense False, bias=hand True, hand select 178.56 ms, bias select 178.87 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
