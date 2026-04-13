# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 4
- dense avg ms/step: 69.0005
- hand-tuned avg ms/step: 630.0259
- bias avg ms/step: 528.5846
- hand-tuned vs dense exact match rate: 0.750
- bias vs dense exact match rate: 0.750
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 1.000
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 9.4827
- hand-tuned score ms/case: 213.8867
- bias score ms/case: 188.8112
- hand-tuned selection ms/case: 862.5307
- bias selection ms/case: 800.1676
- hand-tuned optional-selection ms/case: 24.9477
- bias optional-selection ms/case: 24.9152
- hand-tuned diverse-selection ms/case: 3.7460
- bias diverse-selection ms/case: 3.7100
- hand-tuned compression-selection ms/case: 3.5044
- bias compression-selection ms/case: 3.4262
- hand-tuned policy-bias ms/case: 0.0754
- bias policy-bias ms/case: 0.1009
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

- gemma4_apple_compatibility: dense 74.7860 ms/step, hand 410.3673, bias 294.0617, hand=dense True, bias=dense True, bias=hand True, hand select 417.14 ms, bias select 358.29 ms
- model_roadmap: dense 71.1659 ms/step, hand 750.1032, bias 598.2067, hand=dense True, bias=dense True, bias=hand True, hand select 1042.24 ms, bias select 930.62 ms
- statecache_showcase: dense 65.8811 ms/step, hand 705.4817, bias 588.6415, hand=dense True, bias=dense True, bias=hand True, hand select 1014.86 ms, bias select 938.53 ms
- submission_execution_plan: dense 64.1690 ms/step, hand 654.1513, bias 633.4284, hand=dense False, bias=dense False, bias=hand True, hand select 975.88 ms, bias select 973.24 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
