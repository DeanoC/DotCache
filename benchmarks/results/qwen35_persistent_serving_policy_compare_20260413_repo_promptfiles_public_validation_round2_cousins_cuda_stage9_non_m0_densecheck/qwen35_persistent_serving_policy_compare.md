# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 4
- dense avg ms/step: 28.7134
- hand-tuned avg ms/step: 302.2015
- bias avg ms/step: 308.2500
- hand-tuned vs dense exact match rate: 1.000
- bias vs dense exact match rate: 1.000
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.500
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 20.2479
- hand-tuned score ms/case: 129.4564
- bias score ms/case: 131.4251
- hand-tuned selection ms/case: 155.3281
- bias selection ms/case: 150.1414
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

- gemma4_apple_compatibility: dense 37.2508 ms/step, hand 161.5253, bias 160.7561, hand=dense True, bias=dense True, bias=hand True, hand select 196.63 ms, bias select 141.04 ms
- model_roadmap: dense 26.3120 ms/step, hand 378.2594, bias 342.9351, hand=dense True, bias=dense True, bias=hand True, hand select 150.75 ms, bias select 140.79 ms
- statecache_showcase: dense 25.8876 ms/step, hand 333.0168, bias 336.2990, hand=dense True, bias=dense True, bias=hand True, hand select 135.72 ms, bias select 137.92 ms
- submission_execution_plan: dense 25.4033 ms/step, hand 336.0046, bias 393.0097, hand=dense True, bias=dense True, bias=hand True, hand select 138.21 ms, bias select 180.81 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
