# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 4
- dense avg ms/step: 29.5300
- hand-tuned avg ms/step: 293.0329
- bias avg ms/step: 284.2208
- hand-tuned vs dense exact match rate: 1.000
- bias vs dense exact match rate: 1.000
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.500
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 19.5799
- hand-tuned score ms/case: 129.2489
- bias score ms/case: 116.3897
- hand-tuned selection ms/case: 155.7990
- bias selection ms/case: 133.5846
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

- gemma4_apple_compatibility: dense 39.0557 ms/step, hand 168.2144, bias 166.7633, hand=dense True, bias=dense True, bias=hand True, hand select 200.26 ms, bias select 137.55 ms
- model_roadmap: dense 29.7462 ms/step, hand 362.6406, bias 323.6073, hand=dense True, bias=dense True, bias=hand True, hand select 160.59 ms, bias select 132.29 ms
- statecache_showcase: dense 24.5865 ms/step, hand 320.3399, bias 323.5286, hand=dense True, bias=dense True, bias=hand True, hand select 131.00 ms, bias select 131.84 ms
- submission_execution_plan: dense 24.7314 ms/step, hand 320.9369, bias 322.9839, hand=dense True, bias=dense True, bias=hand True, hand select 131.35 ms, bias select 132.66 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
