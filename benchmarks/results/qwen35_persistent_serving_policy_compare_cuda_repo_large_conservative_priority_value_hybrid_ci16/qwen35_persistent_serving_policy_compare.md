# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 10
- dense avg ms/step: 24.7842
- hand-tuned avg ms/step: 652.1505
- bias avg ms/step: 652.9069
- hand-tuned vs dense exact match rate: 0.500
- bias vs dense exact match rate: 0.500
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.200
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 16.8000
- hand-tuned score ms/case: 115.5739
- bias score ms/case: 113.0670
- hand-tuned selection ms/case: 197.2841
- bias selection ms/case: 194.6796
- hand-tuned optional-selection ms/case: 16.1501
- bias optional-selection ms/case: 16.1806
- hand-tuned diverse-selection ms/case: 6.3027
- bias diverse-selection ms/case: 6.3062
- hand-tuned compression-selection ms/case: 0.0000
- bias compression-selection ms/case: 0.0000
- hand-tuned policy-bias ms/case: 0.1039
- bias policy-bias ms/case: 0.1406
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

- decode_pseudocode: dense 35.3509 ms/step, hand 387.7715, bias 379.5477, hand=dense True, bias=dense True, bias=hand True, hand select 183.64 ms, bias select 154.74 ms
- layer_revision_tracker: dense 23.7670 ms/step, hand 665.8452, bias 667.5665, hand=dense True, bias=dense True, bias=hand True, hand select 200.89 ms, bias select 199.96 ms
- local_layer_profiles: dense 23.5411 ms/step, hand 664.2155, bias 665.6877, hand=dense False, bias=dense False, bias=hand True, hand select 198.62 ms, bias select 198.64 ms
- model_roadmap: dense 23.5266 ms/step, hand 663.5367, bias 667.6949, hand=dense True, bias=dense True, bias=hand True, hand select 198.65 ms, bias select 199.43 ms
- page_selection_eval: dense 23.6708 ms/step, hand 954.9067, bias 958.6044, hand=dense False, bias=dense False, bias=hand True, hand select 235.58 ms, bias select 236.99 ms
- performance_journal: dense 23.6554 ms/step, hand 1257.1692, bias 1254.7832, hand=dense False, bias=dense False, bias=hand True, hand select 273.04 ms, bias select 272.53 ms
- repo_readme: dense 23.6620 ms/step, hand 663.4662, bias 666.0369, hand=dense False, bias=dense False, bias=hand True, hand select 198.31 ms, bias select 198.73 ms
- state_cache_roadmap: dense 23.5784 ms/step, hand 372.8444, bias 374.2992, hand=dense True, bias=dense True, bias=hand True, hand select 153.70 ms, bias select 154.39 ms
- statecache_showcase: dense 23.5725 ms/step, hand 373.4400, bias 374.3320, hand=dense True, bias=dense True, bias=hand True, hand select 154.32 ms, bias select 154.63 ms
- submission_execution_plan: dense 23.5177 ms/step, hand 518.3102, bias 520.5165, hand=dense False, bias=dense False, bias=hand True, hand select 176.09 ms, bias select 176.77 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
