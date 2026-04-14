# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 8
- dense avg ms/step: 79.5970
- hand-tuned avg ms/step: 1524.7338
- bias avg ms/step: 1387.6871
- hand-tuned vs dense exact match rate: 0.750
- bias vs dense exact match rate: 0.750
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.625
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 16.6720
- hand-tuned score ms/case: 133.7778
- bias score ms/case: 75.5344
- hand-tuned selection ms/case: 815.9999
- bias selection ms/case: 723.5120
- hand-tuned optional-selection ms/case: 25.5469
- bias optional-selection ms/case: 25.2024
- hand-tuned diverse-selection ms/case: 5.7358
- bias diverse-selection ms/case: 4.4237
- hand-tuned compression-selection ms/case: 0.0000
- bias compression-selection ms/case: 0.0000
- hand-tuned policy-bias ms/case: 0.1611
- bias policy-bias ms/case: 0.1456
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

- bench_gguf_external_code: dense 85.6299 ms/step, hand 1482.2870, bias 1569.4021, hand=dense True, bias=dense True, bias=hand True, hand select 782.40 ms, bias select 853.79 ms
- gemma4_apple_compatibility: dense 60.5303 ms/step, hand 544.0564, bias 556.1106, hand=dense True, bias=dense True, bias=hand True, hand select 262.13 ms, bias select 244.93 ms
- inspect_policy_prefill_code: dense 62.9986 ms/step, hand 1169.6817, bias 981.8819, hand=dense True, bias=dense True, bias=hand True, hand select 641.11 ms, bias select 516.84 ms
- page_selection_standardized_eval: dense 83.7763 ms/step, hand 2300.4031, bias 1813.2604, hand=dense True, bias=dense True, bias=hand True, hand select 1259.71 ms, bias select 971.65 ms
- performance_journal: dense 64.1645 ms/step, hand 2192.2442, bias 1974.2510, hand=dense False, bias=dense False, bias=hand True, hand select 1086.29 ms, bias select 1022.38 ms
- real_mixed_probe_test_code: dense 71.7814 ms/step, hand 1561.8500, bias 1321.5206, hand=dense True, bias=dense True, bias=hand True, hand select 914.19 ms, bias select 642.59 ms
- state_cache_roadmap: dense 141.4973 ms/step, hand 1285.4080, bias 1399.3675, hand=dense False, bias=dense False, bias=hand True, hand select 750.39 ms, bias select 754.61 ms
- statecache_showcase: dense 66.3976 ms/step, hand 1661.9401, bias 1485.7025, hand=dense True, bias=dense True, bias=hand True, hand select 831.78 ms, bias select 781.30 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
