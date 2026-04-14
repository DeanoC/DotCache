# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 2
- dense avg ms/step: 74.3601
- hand-tuned avg ms/step: 820.3605
- bias avg ms/step: 669.1027
- hand-tuned vs dense exact match rate: 0.000
- bias vs dense exact match rate: 0.000
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 1.000
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 9.5598
- hand-tuned score ms/case: 269.9471
- bias score ms/case: 206.2382
- hand-tuned selection ms/case: 1167.2708
- bias selection ms/case: 1029.8874
- hand-tuned optional-selection ms/case: 26.8165
- bias optional-selection ms/case: 26.5971
- hand-tuned diverse-selection ms/case: 4.6834
- bias diverse-selection ms/case: 4.7889
- hand-tuned compression-selection ms/case: 4.6874
- bias compression-selection ms/case: 4.7400
- hand-tuned policy-bias ms/case: 0.0971
- bias policy-bias ms/case: 0.1350
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

- performance_journal: dense 78.0444 ms/step, hand 925.1482, bias 732.1756, hand=dense False, bias=dense False, bias=hand True, hand select 1328.13 ms, bias select 1159.29 ms
- state_cache_roadmap: dense 70.6759 ms/step, hand 715.5728, bias 606.0298, hand=dense False, bias=dense False, bias=hand True, hand select 1006.41 ms, bias select 900.49 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
