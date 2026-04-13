# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 4
- dense avg ms/step: 28.2191
- hand-tuned avg ms/step: 290.7519
- bias avg ms/step: 303.2387
- hand-tuned vs dense exact match rate: 1.000
- bias vs dense exact match rate: 1.000
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.250
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 19.9433
- hand-tuned score ms/case: 155.1305
- bias score ms/case: 160.2237
- hand-tuned selection ms/case: 180.5745
- bias selection ms/case: 179.1269
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
- hand-tuned executed M0 blocks/case: 3828.00
- bias executed M0 blocks/case: 3828.00
- hand-tuned executed M3 blocks/case: 12.00
- bias executed M3 blocks/case: 12.00
- hand-tuned exact-M3 score ms/case: 0.0000
- bias exact-M3 score ms/case: 0.0000
- hand-tuned final-mix ms/case: 0.0000
- bias final-mix ms/case: 0.0000

## Cases

- gemma4_apple_compatibility: dense 34.7249 ms/step, hand 154.1586, bias 143.7441, hand=dense True, bias=dense True, bias=hand True, hand select 190.17 ms, bias select 152.15 ms
- model_roadmap: dense 24.5821 ms/step, hand 317.9874, bias 319.6849, hand=dense True, bias=dense True, bias=hand True, hand select 171.39 ms, bias select 170.45 ms
- statecache_showcase: dense 24.7741 ms/step, hand 317.3150, bias 367.5569, hand=dense True, bias=dense True, bias=hand True, hand select 170.11 ms, bias select 194.30 ms
- submission_execution_plan: dense 28.7952 ms/step, hand 373.5468, bias 381.9686, hand=dense True, bias=dense True, bias=hand True, hand select 190.63 ms, bias select 199.61 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
