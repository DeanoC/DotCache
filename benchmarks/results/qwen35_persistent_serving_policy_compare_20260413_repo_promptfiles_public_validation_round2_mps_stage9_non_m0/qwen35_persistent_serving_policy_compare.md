# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 8
- dense avg ms/step: 79.7974
- hand-tuned avg ms/step: 2108.4205
- bias avg ms/step: 1984.4225
- hand-tuned vs dense exact match rate: 0.750
- bias vs dense exact match rate: 0.750
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.875
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 12.9415
- hand-tuned score ms/case: 92.2444
- bias score ms/case: 61.5498
- hand-tuned selection ms/case: 764.4219
- bias selection ms/case: 725.7398
- hand-tuned optional-selection ms/case: 27.6948
- bias optional-selection ms/case: 24.2603
- hand-tuned diverse-selection ms/case: 6.4041
- bias diverse-selection ms/case: 4.3415
- hand-tuned compression-selection ms/case: 0.0000
- bias compression-selection ms/case: 0.0000
- hand-tuned policy-bias ms/case: 0.1753
- bias policy-bias ms/case: 0.1637
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
- hand-tuned executed M3 blocks/case: 4368.00
- bias executed M3 blocks/case: 4368.00
- hand-tuned exact-M3 score ms/case: 0.0000
- bias exact-M3 score ms/case: 0.0000
- hand-tuned final-mix ms/case: 0.0000
- bias final-mix ms/case: 0.0000

## Cases

- bench_gguf_external_code: dense 97.8986 ms/step, hand 2226.9635, bias 2106.4700, hand=dense True, bias=dense True, bias=hand True, hand select 804.82 ms, bias select 779.70 ms
- gemma4_apple_compatibility: dense 60.6600 ms/step, hand 912.7291, bias 699.3750, hand=dense True, bias=dense True, bias=hand True, hand select 274.89 ms, bias select 212.60 ms
- inspect_policy_prefill_code: dense 65.4703 ms/step, hand 1780.2240, bias 1365.8021, hand=dense True, bias=dense True, bias=hand True, hand select 613.29 ms, bias select 477.51 ms
- page_selection_standardized_eval: dense 74.0318 ms/step, hand 2840.2762, bias 2879.3170, hand=dense True, bias=dense True, bias=hand True, hand select 1079.89 ms, bias select 1070.89 ms
- performance_journal: dense 62.4574 ms/step, hand 2850.3748, bias 2838.6567, hand=dense False, bias=dense False, bias=hand True, hand select 1047.48 ms, bias select 1040.68 ms
- real_mixed_probe_test_code: dense 133.0642 ms/step, hand 2220.6601, bias 2196.0152, hand=dense True, bias=dense True, bias=hand True, hand select 806.58 ms, bias select 804.34 ms
- state_cache_roadmap: dense 79.3845 ms/step, hand 1821.4468, bias 1646.7866, hand=dense False, bias=dense False, bias=hand True, hand select 691.43 ms, bias select 622.31 ms
- statecache_showcase: dense 65.4122 ms/step, hand 2214.6894, bias 2142.9576, hand=dense True, bias=dense True, bias=hand True, hand select 797.00 ms, bias select 797.88 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
