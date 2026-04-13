# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 10
- dense avg ms/step: 24.7524
- hand-tuned avg ms/step: 388.2736
- bias avg ms/step: 388.7825
- hand-tuned vs dense exact match rate: 0.500
- bias vs dense exact match rate: 0.500
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.300
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 18.1194
- hand-tuned score ms/case: 114.5203
- bias score ms/case: 113.1044
- hand-tuned selection ms/case: 134.5133
- bias selection ms/case: 130.5413
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

- decode_pseudocode: dense 35.6338 ms/step, hand 240.1246, bias 235.7711, hand=dense True, bias=dense True, bias=hand True, hand select 173.13 ms, bias select 130.03 ms
- layer_revision_tracker: dense 23.6487 ms/step, hand 401.9284, bias 395.9761, hand=dense True, bias=dense True, bias=hand True, hand select 131.76 ms, bias select 130.21 ms
- local_layer_profiles: dense 23.7824 ms/step, hand 394.8945, bias 397.7231, hand=dense False, bias=dense False, bias=hand True, hand select 129.94 ms, bias select 130.47 ms
- model_roadmap: dense 23.4443 ms/step, hand 397.9831, bias 398.1829, hand=dense True, bias=dense True, bias=hand True, hand select 129.74 ms, bias select 130.24 ms
- page_selection_eval: dense 23.4485 ms/step, hand 558.4097, bias 561.7572, hand=dense False, bias=dense False, bias=hand True, hand select 130.81 ms, bias select 131.21 ms
- performance_journal: dense 23.5375 ms/step, hand 720.0668, bias 719.5336, hand=dense False, bias=dense False, bias=hand True, hand select 131.48 ms, bias select 131.98 ms
- repo_readme: dense 23.4196 ms/step, hand 394.0854, bias 397.3076, hand=dense False, bias=dense False, bias=hand True, hand select 130.22 ms, bias select 131.10 ms
- state_cache_roadmap: dense 23.4919 ms/step, hand 231.5812, bias 233.1012, hand=dense True, bias=dense True, bias=hand True, hand select 129.18 ms, bias select 129.83 ms
- statecache_showcase: dense 23.5860 ms/step, hand 230.1232, bias 232.8729, hand=dense True, bias=dense True, bias=hand True, hand select 129.14 ms, bias select 130.11 ms
- submission_execution_plan: dense 23.5317 ms/step, hand 313.5390, bias 315.5991, hand=dense False, bias=dense False, bias=hand True, hand select 129.74 ms, bias select 130.24 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
