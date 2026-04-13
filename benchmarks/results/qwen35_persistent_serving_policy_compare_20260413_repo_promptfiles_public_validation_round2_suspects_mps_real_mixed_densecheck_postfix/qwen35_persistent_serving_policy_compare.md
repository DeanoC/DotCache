# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 2
- dense avg ms/step: 82.5625
- hand-tuned avg ms/step: 808.1390
- bias avg ms/step: 632.3670
- hand-tuned vs dense exact match rate: 0.500
- bias vs dense exact match rate: 0.500
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 1.000
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 9.0694
- hand-tuned score ms/case: 269.4670
- bias score ms/case: 187.3424
- hand-tuned selection ms/case: 1184.9782
- bias selection ms/case: 1037.1609
- hand-tuned optional-selection ms/case: 29.4389
- bias optional-selection ms/case: 25.6690
- hand-tuned diverse-selection ms/case: 4.3670
- bias diverse-selection ms/case: 4.5750
- hand-tuned compression-selection ms/case: 4.5555
- bias compression-selection ms/case: 4.5928
- hand-tuned policy-bias ms/case: 0.0891
- bias policy-bias ms/case: 0.1123
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

- performance_journal: dense 82.1001 ms/step, hand 920.1928, bias 761.9172, hand=dense False, bias=dense False, bias=hand True, hand select 1369.16 ms, bias select 1261.66 ms
- state_cache_roadmap: dense 83.0249 ms/step, hand 696.0852, bias 502.8168, hand=dense True, bias=dense True, bias=hand True, hand select 1000.79 ms, bias select 812.66 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
