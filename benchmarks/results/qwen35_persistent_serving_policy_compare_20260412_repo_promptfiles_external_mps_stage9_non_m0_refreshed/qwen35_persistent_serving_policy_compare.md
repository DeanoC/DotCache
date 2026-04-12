# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 3
- dense avg ms/step: 69.9718
- hand-tuned avg ms/step: 1609.8176
- bias avg ms/step: 1700.5052
- hand-tuned vs dense exact match rate: 1.000
- bias vs dense exact match rate: 1.000
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.333
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 9.3829
- hand-tuned score ms/case: 60.6789
- bias score ms/case: 54.1944
- hand-tuned selection ms/case: 523.1015
- bias selection ms/case: 591.2766
- hand-tuned optional-selection ms/case: 24.4248
- bias optional-selection ms/case: 27.1783
- hand-tuned diverse-selection ms/case: 5.0356
- bias diverse-selection ms/case: 3.8605
- hand-tuned compression-selection ms/case: 0.0000
- bias compression-selection ms/case: 0.0000
- hand-tuned policy-bias ms/case: 0.0766
- bias policy-bias ms/case: 0.1188
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
- hand-tuned executed M3 blocks/case: 3120.00
- bias executed M3 blocks/case: 3120.00
- hand-tuned exact-M3 score ms/case: 0.0000
- bias exact-M3 score ms/case: 0.0000
- hand-tuned final-mix ms/case: 0.0000
- bias final-mix ms/case: 0.0000

## Cases

- decode_pseudocode: dense 77.1755 ms/step, hand 1677.9311, bias 1982.4075, hand=dense True, bias=dense True, bias=hand True, hand select 545.33 ms, bias select 683.64 ms
- state_cache_roadmap: dense 66.2536 ms/step, hand 1572.7228, bias 1531.0586, hand=dense True, bias=dense True, bias=hand True, hand select 502.94 ms, bias select 523.36 ms
- statecache_showcase: dense 66.4862 ms/step, hand 1578.7989, bias 1588.0494, hand=dense True, bias=dense True, bias=hand True, hand select 521.03 ms, bias select 566.83 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
