# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 2
- dense avg ms/step: 29.7774
- hand-tuned avg ms/step: 345.3505
- bias avg ms/step: 344.3261
- hand-tuned vs dense exact match rate: 0.000
- bias vs dense exact match rate: 0.000
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.500
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 18.5451
- hand-tuned score ms/case: 121.3219
- bias score ms/case: 115.2457
- hand-tuned selection ms/case: 151.6148
- bias selection ms/case: 132.7970
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

- performance_journal: dense 35.4158 ms/step, hand 408.6209, bias 403.3293, hand=dense False, bias=dense False, bias=hand True, hand select 170.42 ms, bias select 133.09 ms
- state_cache_roadmap: dense 24.1390 ms/step, hand 282.0802, bias 285.3229, hand=dense False, bias=dense False, bias=hand True, hand select 132.81 ms, bias select 132.50 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
