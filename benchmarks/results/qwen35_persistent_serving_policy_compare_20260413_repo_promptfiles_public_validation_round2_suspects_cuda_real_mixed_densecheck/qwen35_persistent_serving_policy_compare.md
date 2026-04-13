# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 2
- dense avg ms/step: 29.9411
- hand-tuned avg ms/step: 334.2413
- bias avg ms/step: 328.8473
- hand-tuned vs dense exact match rate: 0.500
- bias vs dense exact match rate: 0.500
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.500
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 19.7051
- hand-tuned score ms/case: 160.2416
- bias score ms/case: 151.3782
- hand-tuned selection ms/case: 190.1030
- bias selection ms/case: 168.5041
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
- hand-tuned executed M0 blocks/case: 5112.00
- bias executed M0 blocks/case: 5112.00
- hand-tuned executed M3 blocks/case: 24.00
- bias executed M3 blocks/case: 24.00
- hand-tuned exact-M3 score ms/case: 0.0000
- bias exact-M3 score ms/case: 0.0000
- hand-tuned final-mix ms/case: 0.0000
- bias final-mix ms/case: 0.0000

## Cases

- performance_journal: dense 36.0149 ms/step, hand 396.3810, bias 383.8854, hand=dense False, bias=dense False, bias=hand True, hand select 214.94 ms, bias select 174.03 ms
- state_cache_roadmap: dense 23.8673 ms/step, hand 272.1017, bias 273.8092, hand=dense True, bias=dense True, bias=hand True, hand select 165.26 ms, bias select 162.98 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
