# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 8
- dense avg ms/step: 25.5815
- hand-tuned avg ms/step: 302.7460
- bias avg ms/step: 304.2355
- hand-tuned vs dense exact match rate: 0.750
- bias vs dense exact match rate: 0.750
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.125
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 18.2862
- hand-tuned score ms/case: 116.7496
- bias score ms/case: 115.3711
- hand-tuned selection ms/case: 136.8010
- bias selection ms/case: 132.6475
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

- bench_gguf_external_code: dense 34.2774 ms/step, hand 327.7549, bias 323.0196, hand=dense True, bias=dense True, bias=hand True, hand select 168.22 ms, bias select 133.19 ms
- gemma4_apple_compatibility: dense 24.6421 ms/step, hand 134.8364, bias 138.1186, hand=dense True, bias=dense True, bias=hand True, hand select 131.14 ms, bias select 132.32 ms
- inspect_policy_prefill_code: dense 24.2374 ms/step, hand 235.9198, bias 238.6266, hand=dense True, bias=dense True, bias=hand True, hand select 132.13 ms, bias select 131.10 ms
- page_selection_standardized_eval: dense 24.2008 ms/step, hand 403.0387, bias 403.1582, hand=dense True, bias=dense True, bias=hand True, hand select 134.26 ms, bias select 132.10 ms
- performance_journal: dense 24.0944 ms/step, hand 402.3423, bias 403.8823, hand=dense False, bias=dense False, bias=hand True, hand select 132.20 ms, bias select 133.68 ms
- real_mixed_probe_test_code: dense 24.5637 ms/step, hand 317.7379, bias 319.8113, hand=dense True, bias=dense True, bias=hand True, hand select 131.69 ms, bias select 131.94 ms
- state_cache_roadmap: dense 24.3653 ms/step, hand 282.4351, bias 286.9216, hand=dense False, bias=dense False, bias=hand True, hand select 133.40 ms, bias select 134.45 ms
- statecache_showcase: dense 24.2705 ms/step, hand 317.9029, bias 320.3459, hand=dense True, bias=dense True, bias=hand True, hand select 131.36 ms, bias select 132.40 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
