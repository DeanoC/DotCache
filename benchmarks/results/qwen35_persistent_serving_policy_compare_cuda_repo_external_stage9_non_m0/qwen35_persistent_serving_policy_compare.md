# Qwen3.5 Persistent Serving Policy Compare

## Summary

- case count: 3
- dense avg ms/step: 28.3445
- hand-tuned avg ms/step: 193.6261
- bias avg ms/step: 192.5384
- hand-tuned vs dense exact match rate: 1.000
- bias vs dense exact match rate: 1.000
- bias vs hand-tuned exact match rate: 1.000
- bias faster than hand-tuned rate: 0.333
- hand-tuned policy resolve ms/case: 0.0000
- bias policy resolve ms/case: 12.4255
- hand-tuned score ms/case: 104.8845
- bias score ms/case: 96.5888
- hand-tuned selection ms/case: 148.3594
- bias selection ms/case: 140.1980
- hand-tuned optional-selection ms/case: 9.1799
- bias optional-selection ms/case: 9.2830
- hand-tuned diverse-selection ms/case: 3.9576
- bias diverse-selection ms/case: 3.9521
- hand-tuned compression-selection ms/case: 0.0000
- bias compression-selection ms/case: 0.0000
- hand-tuned policy-bias ms/case: 0.0765
- bias policy-bias ms/case: 0.1069
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

- decode_pseudocode: dense 36.1152 ms/step, hand 199.1901, bias 192.8510, hand=dense True, bias=dense True, bias=hand True, hand select 167.05 ms, bias select 139.58 ms
- state_cache_roadmap: dense 24.1438 ms/step, hand 191.2833, bias 192.0756, hand=dense True, bias=dense True, bias=hand True, hand select 138.90 ms, bias select 140.96 ms
- statecache_showcase: dense 24.7745 ms/step, hand 190.4048, bias 192.6887, hand=dense True, bias=dense True, bias=hand True, hand select 139.12 ms, bias select 140.06 ms

## Read

This is an end-to-end serving comparison through the real Qwen persistent harness, using the same unseen prompt families as the external replay validation.
It compares dense generation, persistent hand-tuned selection, and persistent bias-guided selection on generated ids and per-step decode latency.
