# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 4
- dense avg ms/step: 69.9623
- hand-tuned avg ms/step: 646.8884
- bias avg ms/step: 569.0128
- hand-tuned vs dense exact match rate: 1.000
- bias vs dense exact match rate: 1.000
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.750
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 10.4667
- hand-tuned score ms/case: 218.8800
- bias score ms/case: 207.5510
- hand-tuned selection ms/case: 869.9452
- bias selection ms/case: 836.5389
- hand-tuned optional-selection ms/case: 26.7673
- bias optional-selection ms/case: 24.7096
- hand-tuned diverse-selection ms/case: 4.7234
- bias diverse-selection ms/case: 3.5290
- hand-tuned compression-selection ms/case: 3.6320
- bias compression-selection ms/case: 3.6371
- hand-tuned policy-bias ms/case: 0.0887
- bias policy-bias ms/case: 0.1162
- hand-tuned direct-M0 assembly ms/case: 0.0000
- bias direct-M0 assembly ms/case: 0.0000
- hand-tuned direct-M0 query-prep ms/case: 0.0000
- bias direct-M0 query-prep ms/case: 0.0000
- hand-tuned direct-M0 gather ms/case: 0.0000
- bias direct-M0 gather ms/case: 0.0000
- hand-tuned direct-M0 score ms/case: 0.0000
- bias direct-M0 score ms/case: 0.0000
- hand-tuned executed M0 blocks/case: 3828.00
- bias executed M0 blocks/case: 3828.00
- hand-tuned executed M3 blocks/case: 12.00
- bias executed M3 blocks/case: 12.00
- hand-tuned exact-M3 score ms/case: 0.0000
- bias exact-M3 score ms/case: 0.0000
- hand-tuned final-mix ms/case: 0.0000
- bias final-mix ms/case: 0.0000

## Cases

- gemma4_apple_compatibility: dense 71.2104 ms/step, hand 415.6466, bias 322.1127, hand=dense True, bias=dense True, bias=hand True, hand select 425.60 ms, bias select 397.27 ms
- model_roadmap: dense 77.1572 ms/step, hand 768.6825, bias 642.5137, hand=dense True, bias=dense True, bias=hand True, hand select 1024.27 ms, bias select 978.47 ms
- statecache_showcase: dense 61.4135 ms/step, hand 674.6809, bias 690.1170, hand=dense True, bias=dense True, bias=hand True, hand select 1002.33 ms, bias select 1021.73 ms
- submission_execution_plan: dense 70.0682 ms/step, hand 728.5436, bias 621.3077, hand=dense True, bias=dense True, bias=hand True, hand select 1027.59 ms, bias select 948.69 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
