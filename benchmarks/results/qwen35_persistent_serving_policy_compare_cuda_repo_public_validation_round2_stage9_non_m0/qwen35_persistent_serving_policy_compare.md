# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 8
- dense avg ms/step: 25.7194
- hand-tuned avg ms/step: 304.2815
- bias avg ms/step: 305.1092
- hand-tuned vs dense exact match rate: 0.750
- bias vs dense exact match rate: 0.750
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.125
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 18.1884
- hand-tuned score ms/case: 116.2731
- bias score ms/case: 114.9231
- hand-tuned selection ms/case: 136.6463
- bias selection ms/case: 131.9661
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

- bench_gguf_external_code: dense 35.5336 ms/step, hand 328.1159, bias 324.3985, hand=dense True, bias=dense True, bias=hand True, hand select 170.09 ms, bias select 132.55 ms
- gemma4_apple_compatibility: dense 24.1593 ms/step, hand 134.8085, bias 136.3494, hand=dense True, bias=dense True, bias=hand True, hand select 129.79 ms, bias select 130.62 ms
- inspect_policy_prefill_code: dense 24.1040 ms/step, hand 236.1500, bias 237.1046, hand=dense True, bias=dense True, bias=hand True, hand select 130.26 ms, bias select 130.76 ms
- page_selection_standardized_eval: dense 24.3079 ms/step, hand 405.3633, bias 405.5225, hand=dense True, bias=dense True, bias=hand True, hand select 133.03 ms, bias select 131.45 ms
- performance_journal: dense 24.6712 ms/step, hand 405.4406, bias 406.5898, hand=dense False, bias=dense False, bias=hand True, hand select 131.86 ms, bias select 132.19 ms
- real_mixed_probe_test_code: dense 24.1569 ms/step, hand 320.7492, bias 323.9125, hand=dense True, bias=dense True, bias=hand True, hand select 131.07 ms, bias select 132.00 ms
- state_cache_roadmap: dense 24.5565 ms/step, hand 283.6598, bias 286.5285, hand=dense False, bias=dense False, bias=hand True, hand select 133.47 ms, bias select 134.54 ms
- statecache_showcase: dense 24.2660 ms/step, hand 319.9650, bias 320.4675, hand=dense True, bias=dense True, bias=hand True, hand select 133.59 ms, bias select 131.62 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
